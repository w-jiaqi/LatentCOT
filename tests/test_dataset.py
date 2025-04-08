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
	test_ds = Dataset.from_dict(test_ds)

	dataset = DatasetDict({
		'train': train_ds,
		'test': test_ds
	})

	model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
	tokenizer = LatentTokenizer('meta-llama/Llama-3.2-1B')

	latent_dataset = get_latent_cot_sft_dataset(
		dataset=dataset,
		tokenizer=tokenizer,
		embedding=model.get_input_embeddings(),
		latent_pool=4
	)

	