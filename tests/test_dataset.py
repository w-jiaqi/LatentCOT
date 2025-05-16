import sys, os

from sft.models.latent_tokenizer import LatentTokenizer
sys.path.insert(0, os.path.abspath("."))  # hack for imports
from data.dataset import compress_embeddings, get_latent_cot_sft_dataset
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

uncompressed_1 = torch.tensor([
	[1, 5, 9, 15],
	[7, 8, 134, 19],
	[32, 56, 90, 110],
	[1, 2, 3, 4],
	[10, 11, 12, 13],
	[97, 96, 32, 50],
	[112, 112, 13, 15],
], dtype=torch.float32)

compressed_1 = torch.tensor([
	[10.25, 17.75, 59, 37],
	[73, 73, 19, 26 ]
], dtype=torch.float32)

uncompressed_2 = torch.tensor([
	[20, 30, 40, 50],
	[15, 25, 35, 45],
	[10, 20, 30, 40],
	[5, 15, 25, 35],
	[30, 40, 50, 60],
	[25, 35, 45, 55],
	[20, 30, 40, 50],
], dtype=torch.float32)

compressed_2 = torch.tensor([
	[12.5, 22.5, 32.5, 42.5],
	[25.0, 35.0, 45.0, 55.0]
], dtype=torch.float32)

uncompressed_3 = torch.tensor([
    [100, 200, 300, 400],
    [101, 201, 301, 401],
    [102, 202, 302, 402],
    [103, 203, 303, 403],
    [104, 204, 304, 404],
    [105, 205, 305, 405],
    [106, 206, 306, 406],
], dtype=torch.float32)

compressed_3 = torch.tensor([
    [101.5, 201.5, 301.5, 401.5],
    [105.0, 205.0, 305.0, 405.0]
], dtype=torch.float32)

def test_compress_embeddings_1():
	length1, test_compressed_1 = compress_embeddings(uncompressed_1, 4)

	torch.testing.assert_close(test_compressed_1, compressed_1)
	assert length1 == 2

def test_compress_embeddings_2():
	length2, test_compressed_2 = compress_embeddings(uncompressed_2, 4)

	torch.testing.assert_close(test_compressed_2, compressed_2)
	assert length2 == 2

def test_compress_embeddings_3():
	length3, test_compressed_3 = compress_embeddings(uncompressed_3, 4)

	torch.testing.assert_close(test_compressed_3, compressed_3)
	assert length3 == 2

def test_compress_embeddings_all():
	all_uncompressed = torch.stack((
		uncompressed_1,
		uncompressed_2,
		uncompressed_3
	))

	all_compressed = torch.stack((
		compressed_1,
		compressed_2,
		compressed_3
	))

	length_test_all_compressed, test_all_compressed = compress_embeddings(all_uncompressed, 4)

	torch.testing.assert_close(test_all_compressed, all_compressed)
	
	assert length_test_all_compressed == 2

def test_latent_dataset():
	fake_data = {
		'question': ['Some fake question blah blah', 'another fake  38 813 7138', 'random numbers'],
		'reasoning': ['8123 184 8924 477 ueoath ua', '83 89,.lph oaehu nhkkasw ', 'srhr,cp hrcau lr'],
		'answer': ['Answer noaeu taeutn au ht;', 'another ANSWER uhh', 'huoentshkqm mn3 333333']
	}

	train_ds = Dataset.from_dict(fake_data)
	test_ds = Dataset.from_dict(fake_data)

	dataset = DatasetDict({
		'train': train_ds,
		'test': test_ds
	})
	
def test_create_soft_labels():
    from data.dataset import create_soft_labels

    token_ids = torch.tensor([1, 5, 10, 12])
    vocab_size = 20
    soft_labels = create_soft_labels(token_ids, vocab_size)

    assert soft_labels.shape == (4, vocab_size)

    for i, idx in enumerate(token_ids):
        assert soft_labels[i].sum().item() == 1.0
        assert soft_labels[i][idx].item() == 1.0
        assert torch.all(soft_labels[i][torch.arange(vocab_size) != idx] == 0)

def test_create_soft_labels_expected():
    from data.dataset import create_soft_labels

    token_ids = torch.tensor([1, 5, 10, 12])
    vocab_size = 20
    soft_labels = create_soft_labels(token_ids, vocab_size)

    expected = torch.zeros((4, vocab_size))
    expected[0, 1] = 1.0
    expected[1, 5] = 1.0
    expected[2, 10] = 1.0
    expected[3, 12] = 1.0

    assert torch.allclose(soft_labels, expected)

def test_create_latent_embeddings_batched():
    from data.dataset import create_latent_embeddings

    class DummyEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # vocab size 6, embedding dim 2
            self.weight = torch.nn.Parameter(torch.tensor([
                [0, 0],    # token 0
                [1, 2],    # token 1
                [3, 4],    # token 2
                [5, 6],    # token 3
                [7, 8],    # token 4
                [9, 10],   # token 5
            ], dtype=torch.float32))
        def forward(self, idx):
            return self.weight[idx]

    embedding = DummyEmbedding()
    latent_pool = 2

    # Batch of 2: [1,2,3], [4,5,0]
    # Will be padded to [1,2,3,0], [4,5,0,0]
    token_ids = torch.tensor([
        [1, 2, 3],
        [4, 5, 0]
    ], dtype=torch.long)

    pooled_embeds, soft_labels = create_latent_embeddings(token_ids, latent_pool, embedding)

    # Should be (2, 2, 2) for pooled_embeds and (2, 2, 6) for soft_labels
    assert pooled_embeds.shape == (2, 2, 2)
    assert soft_labels.shape == (2, 2, 6)

    # Weights for pool=2: [2/3, 1/3]
    weights = torch.tensor([2/3, 1/3], dtype=torch.float32)

    # Sample 0, group 0: tokens [1,2]
    expected0_0 = weights[0] * torch.tensor([1,2]) + weights[1] * torch.tensor([3,4])
    # Sample 0, group 1: tokens [3,0]
    expected0_1 = weights[0] * torch.tensor([5,6]) + weights[1] * torch.tensor([0,0])
    # Sample 1, group 0: tokens [4,5]
    expected1_0 = weights[0] * torch.tensor([7,8]) + weights[1] * torch.tensor([9,10])
    # Sample 1, group 1: tokens [0,0]
    expected1_1 = weights[0] * torch.tensor([0,0]) + weights[1] * torch.tensor([0,0])

    torch.testing.assert_close(pooled_embeds[0,0], expected0_0)
    torch.testing.assert_close(pooled_embeds[0,1], expected0_1)
    torch.testing.assert_close(pooled_embeds[1,0], expected1_0)
    torch.testing.assert_close(pooled_embeds[1,1], expected1_1)

    # Soft labels
    expected_soft_0_0 = torch.zeros(6)
    expected_soft_0_0[1] = weights[0]
    expected_soft_0_0[2] = weights[1]
    expected_soft_0_1 = torch.zeros(6)
    expected_soft_0_1[3] = weights[0]
    expected_soft_0_1[0] = weights[1]
    expected_soft_1_0 = torch.zeros(6)
    expected_soft_1_0[4] = weights[0]
    expected_soft_1_0[5] = weights[1]
    expected_soft_1_1 = torch.zeros(6)
    expected_soft_1_1[0] = weights[0] + weights[1]

    torch.testing.assert_close(soft_labels[0,0], expected_soft_0_0)
    torch.testing.assert_close(soft_labels[0,1], expected_soft_0_1)
    torch.testing.assert_close(soft_labels[1,0], expected_soft_1_0)
    torch.testing.assert_close(soft_labels[1,1], expected_soft_1_1)
