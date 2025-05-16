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

def test_create_latent_embeddings():
    from data.dataset import create_latent_embeddings

    # Simulate a vocab of size 6 and embedding dim 3
    class DummyEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(18, dtype=torch.float32).reshape(6, 3))
        def forward(self, idx):
            return self.weight[idx]

    embedding = DummyEmbedding()
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    latent_pool = 2

    pooled_embeds, soft_labels = create_latent_embeddings(token_ids, latent_pool, embedding)

    # There should be 3 latents (since 5 tokens, padded to 6, pool=2)
    assert pooled_embeds.shape == (3, 3)
    assert soft_labels.shape == (3, 6)

    # Check that the soft_labels rows sum to 1
    assert torch.allclose(soft_labels.sum(dim=1), torch.ones(3))

    # Check that the first latent uses tokens 1 and 2 with correct weights
    weights = torch.linspace(2, 1, steps=2)
    weights = weights / weights.sum()
    expected_soft_label0 = torch.zeros(6)
    expected_soft_label0[1] = weights[0]
    expected_soft_label0[2] = weights[1]
    assert torch.allclose(soft_labels[0], expected_soft_label0)

    # Check that the last latent uses tokens 5 and 0 (padded with zero)
    expected_soft_label2 = torch.zeros(6)
    expected_soft_label2[5] = weights[0]
    expected_soft_label2[0] = weights[1]
    assert torch.allclose(soft_labels[2], expected_soft_label2)

