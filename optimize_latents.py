import torch
import argparse
import os
from torch.optim import Adam
from tqdm import tqdm
import pickle

from sft.models.latent_cot_model import LatentCOTModel, LossType
from sft.models.latent_tokenizer import LatentTokenizer
from data.multiplication_dataset import get_4x4_dataset
from data.gsm8k_dataset import get_gsm8k_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["4x4","gsm8k"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--latent_pool", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tie_weights", action='store_true')
    parser.add_argument("--output_dir", type=str, default="optimized_latents")
    parser.add_argument("--log_every", type=int, default=20, help="Log progress every N examples")
    parser.add_argument("--test_examples", type=int, default=5, help="Number of examples to test after optimization")
    return parser.parse_args()

def optimize_latent_for_example(
    model, 
    tokenizer, 
    example, 
    initial_latent, 
    device, 
    steps=100, 
    lr=1e-3,
    verbose=False
):
    """Optimize a latent representation for a specific example"""
    question_str = example["question"]
    reasoning_str = example["reasoning"]
    answer_str = example["answer"]
    
    embedding_layer = model.embedding
    
    Z = initial_latent.clone().detach().to(device).requires_grad_(True)
    
    optimizer = Adam([Z], lr=lr)
    
    question_ids = tokenizer.encode(question_str, return_tensors="pt").to(device)[0]
    gt_str = reasoning_str + " #### " + answer_str
    gt_ids = tokenizer.encode(gt_str, return_tensors="pt").to(device)[0]
    
    bos_id = torch.tensor([tokenizer.bos_token_id], device=device)
    start_latent_id = torch.tensor([tokenizer.start_latent_id], device=device)
    end_latent_id = torch.tensor([tokenizer.end_latent_id], device=device)
    
    bos_embed = embedding_layer(bos_id).view(1, 1, -1)
    question_embeds = embedding_layer(question_ids.unsqueeze(0))
    start_latent_embed = embedding_layer(start_latent_id).view(1, 1, -1)
    end_latent_embed = embedding_layer(end_latent_id).view(1, 1, -1)
    gt_embeds = embedding_layer(gt_ids.unsqueeze(0))
    
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        
        Z_batched = Z.unsqueeze(0)
        
        context_embeds = torch.cat([
            bos_embed,
            question_embeds,
            start_latent_embed,
            Z_batched,
            end_latent_embed
        ], dim=1)
        
        inputs_embeds = torch.cat([context_embeds, gt_embeds], dim=1)
        
        attention_mask = torch.ones(inputs_embeds.shape[:-1], device=device)
        
        context_len = context_embeds.shape[1]
        seq_len = inputs_embeds.shape[1]
        
        labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)  # -100 is the ignore index
        labels[:, context_len:] = gt_ids.unsqueeze(0)
        
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{steps}, Loss: {loss.item():.4f}")
    
    return Z.detach().cpu(), losses

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_latents_pool{args.latent_pool}.pkl")
    log_file = os.path.join(args.output_dir, f"{args.dataset}_optimization_log.txt")
    
    if args.dataset == "4x4":
        ds = get_4x4_dataset()
    else:  # gsm8k
        ds = get_gsm8k_dataset()
    
    train_data = ds["train"]
    num_examples = len(train_data)
    print(f"Loaded {num_examples} examples from {args.dataset}")
    
    print(f"Loading model from {args.model}")
    tokenizer = LatentTokenizer(args.tokenizer)
    model = LatentCOTModel(args.model, tokenizer, tie_weights=args.tie_weights).to(device)
    model.eval()  

    for param in model.parameters():
        param.requires_grad = False
    
    embedding_layer = model.embedding
    embed_dim = embedding_layer.weight.shape[1]

    optimized_latents = {}
    all_losses = {}
    
    print("Initializing mean-pooled latents for all examples...")
    mean_pooled_latents = {}
    
    for idx in tqdm(range(num_examples)):
        with torch.no_grad():
            example = train_data[idx]
            reasoning_str = example["reasoning"]

            reasoning_ids = tokenizer.encode(reasoning_str, return_tensors="pt").to(device)[0]
            reasoning_embeds = embedding_layer(reasoning_ids.unsqueeze(0))[0]
            
            seq_len = reasoning_embeds.shape[0]
            latent_size = args.latent_pool
            pooled_size = max(1, seq_len // latent_size)
            
            latent_vec = torch.zeros(latent_size, embed_dim, device=device)
            for i in range(latent_size):
                start_idx = i * pooled_size
                end_idx = min((i + 1) * pooled_size, seq_len)
                if start_idx < end_idx:
                    latent_vec[i] = torch.mean(reasoning_embeds[start_idx:end_idx], dim=0)
            
            mean_pooled_latents[idx] = latent_vec.cpu()

    print(f"Optimizing latents for all {num_examples} examples...")
    
    with open(log_file, 'w') as log:
        log.write(f"Optimizing latents for {args.dataset} dataset\n")
        log.write(f"Parameters: latent_pool={args.latent_pool}, steps={args.steps}, lr={args.lr}\n\n")
        
        for idx in tqdm(range(num_examples)):
            example = train_data[idx]
            initial_latent = mean_pooled_latents[idx]
            
            verbose = (idx % args.log_every == 0)
            if verbose:
                print(f"\nOptimizing example {idx}/{num_examples}")
                
            optimized_latent, losses = optimize_latent_for_example(
                model=model,
                tokenizer=tokenizer,
                example=example,
                initial_latent=initial_latent,
                device=device,
                steps=args.steps,
                lr=args.lr,
                verbose=verbose
            )
            
            optimized_latents[idx] = optimized_latent
            all_losses[idx] = losses
            
            if verbose or idx == num_examples - 1:
                initial_loss = losses[0]
                final_loss = losses[-1]
                improvement = (initial_loss - final_loss) / initial_loss * 100
                
                log_message = f"Example {idx}: Initial loss = {initial_loss:.4f}, Final loss = {final_loss:.4f}, Improvement = {improvement:.2f}%"
                print(log_message)
                log.write(log_message + "\n")
                
                if (idx + 1) % 100 == 0 or idx == num_examples - 1:
                    temp_output = os.path.join(args.output_dir, f"{args.dataset}_latents_pool{args.latent_pool}_temp_{idx}.pkl")
                    with open(temp_output, 'wb') as f:
                        pickle.dump(optimized_latents, f)
                    print(f"Saved intermediate results to {temp_output}")
    
    print(f"Saving all optimized latents to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(optimized_latents, f)
    
    """
    print(f"\nTesting {args.test_examples} examples with optimized latents:")
    test_indices = list(range(min(args.test_examples, len(ds["test"]))))
    
    with torch.no_grad():
        for test_idx in test_indices:
            example = ds["test"][test_idx]
            question_str = example["question"]
            
            first_train_latent = next(iter(optimized_latents.values()))

            question_ids = tokenizer.encode(question_str, return_tensors="pt").to(device)[0]

            print(f"\nQuestion: {question_str}")
            
            generated_text = model.generate(
                question_ids, 
                output_cot=False,
                max_new_latents=0,
                max_new_tokens=128,
                provided_latents=first_train_latent.to(device)
            )
            
            print(f"Generated: {generated_text}")
    """

if __name__ == "__main__":
    main()