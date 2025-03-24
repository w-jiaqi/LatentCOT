from datasets import load_dataset
import torch

TRAIN_4x4_PATH = "data/multiplication/4x4/train.txt"
TEST_4x4_PATH = "data/multiplication/4x4/valid.txt"

def get_4x4_multiplication_dataset(tokenizer, num_train=None, num_proc=None):
    def preprocess_fn(examples):
        questions = [example.split("||")[0] + '||' for example in examples['text']] # TODO change to \n and then use same base dataset of QRA
        answers = [example.split("||")[1] for example in examples['text']]

        questions_tokenized = tokenizer(questions, add_special_tokens=True) # add begin of seq token
        answers_tokenized = tokenizer(answers, add_special_tokens=False)

        labels = []
        input_ids = []
        attention_mask = []

        for i in range(len(questions)):
            question_input_ids = questions_tokenized['input_ids'][i]
            answer_input_ids = answers_tokenized['input_ids'][i] + [tokenizer.eos_token_id]
            
            question_attention_mask = questions_tokenized['attention_mask'][i]
            answer_attention_mask = answers_tokenized['attention_mask'][i] + [1]
            
            labels.append([-100] * len(question_input_ids) + answer_input_ids) # -100 is the ignore token for cross entropy loss
            input_ids.append(question_input_ids + answer_input_ids)
            attention_mask.append(question_attention_mask + answer_attention_mask)

        return {'labels': labels, 'input_ids': input_ids, 'attention_mask': attention_mask}

    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_4x4_PATH,
            "test": TEST_4x4_PATH,
        },
    )

    if num_train != None:
        ds["train"] = ds["train"].select(range(num_train))

    ds = ds.map(preprocess_fn, batched=True, num_proc=num_proc, remove_columns=['text'])

    return ds

# note: i dont think the way we preprocess here will work for gsm8k because 
# we can't batch process the latents (they will all be different lengths)
def get_text_to_latent_dataset(tokenizer, embedding, start_latent_id, 
                                          end_latent_id, num_proc=None, latent_pool=10, num_train=None):
    start_latent_embedding = embedding(torch.tensor(start_latent_id))
    end_latent_embedding = embedding(torch.tensor(end_latent_id))

    def preprocess_fn(examples):
        texts = examples['text']
        questions = [example.split("||")[0] + '||' for example in texts]
        full_answers = [example.split("||")[1] for example in texts]
        
        reasonings = [full_answer.split("####")[0].strip() for full_answer in full_answers]

        question_tokens = tokenizer(questions, return_tensors="pt")
        reasoning_tokens = tokenizer(reasonings, return_tensors="pt")

        question_embeddings = embedding(question_tokens['input_ids'])
        reasoning_embeddings = embedding(reasoning_tokens['input_ids'])

        batch_size, reasoning_length, latent_dim = reasoning_embeddings.shape
        latent_reasoning_length = (reasoning_length // latent_pool) + 1

        latent_reasoning_embeddings = torch.zeros(batch_size, latent_reasoning_length, latent_dim)
        
        for i in range(0, reasoning_length, latent_pool):
            latent_reasoning_embeddings[:, i // latent_pool, :] = reasoning_embeddings[:, i:i+latent_pool, :].mean(dim=1)

        start_latent_column = start_latent_embedding.expand(batch_size, 1, latent_dim)
        end_latent_column = end_latent_embedding.expand(batch_size, 1, latent_dim)

        assert start_latent_column.shape == end_latent_column.shape == torch.Size([batch_size, 1, latent_dim])

        input_embeds = torch.cat((
            question_embeddings, 
            start_latent_column, 
            latent_reasoning_embeddings, 
            end_latent_column
        ), dim=1)

        attention_mask = torch.ones(input_embeds.shape[:-1])
        
        # we mask the loss on the start_latent and don't mask on the end latent
        label_mask = torch.cat((
            torch.zeros(batch_size, question_embeddings.shape[1] + 1), 
            torch.ones(batch_size, latent_reasoning_length + 1)
        ), dim=1)

        assert input_embeds.shape[:-1] == label_mask.shape, f"input_embeds: {input_embeds.shape}, label_mask: {label_mask.shape}"

        return {
            'input_embeds': input_embeds,
            'attention_mask': attention_mask,
            'label_mask': label_mask
        }

    
    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_4x4_PATH,
            "test": TEST_4x4_PATH
        }
    )

    if num_train != None:
        ds["train"] = ds["train"].select(range(num_train))

    ds = ds.map(preprocess_fn, batched=True, num_proc=num_proc, remove_columns=['text'])
    ds.set_format("pt")

    return ds


# dont have the base model follow the reverse digit instructions
def format_multiplication_example_base_model(example):
    text = example["text"]

    question = text.split("||")[0].strip()
    num_1 = question.split("*")[0].strip().replace(" ", "")
    num_2 = question.split("*")[1].strip().replace(" ", "")

    # reversing the digits so it's more fair for the base model
    num_1 = int(num_1[::-1])
    num_2 = int(num_2[::-1])

    question = str(num_1) + " * " + str(num_2)

    full_answer = text.split("||")[1].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": f"Only output one number as your final answer. You MUST not write anything else.",
            },
            {
                "role": "user",
                "content": f"Question: {question}",
            },
            {"role": "assistant", "content": full_answer},
        ]
    }
