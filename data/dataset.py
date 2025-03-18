from datasets import load_dataset
from trl import apply_chat_template


def get_gsm8k_dataset(tokenizer):
    def format_gsm8k_example(example):
        question = example["question"]
        answer = example["answer"]

        prompt = {
            "role": "user",
            "content": f"Question: {question} Let's think step by step. At the end, you MUST write the answer as an integer after '####'",
        }
        completion = {"role": "assistant", "content": f"Answer: {answer}"}

        return apply_chat_template(
            {"prompt": [prompt], "completion": [completion]}, tokenizer
        )

    ds = load_dataset("openai/gsm8k", "main")

    ds["train"] = ds["train"].map(format_gsm8k_example)
    ds["test"] = ds["test"].map(format_gsm8k_example)

    return ds


def format_multiplication_example(example):
    text = example["text"]

    question = text.split("||")[0].strip()
    full_answer = text.split("||")[1].strip()

    reasoning = full_answer.split("####")[0].strip()
    answer = full_answer.split("####")[1].strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Show all intermediate work with digits in reverse order, putting partial sums in parentheses, and make sure to end your answer with the digits in reverse order with the exact format: ####[final_answer]. For example, if the question is '5 6 3 2 * 7 4 3 4', your intermediate work should be '5 5 5 6 1 + 0 0 6 4 9 0 ( 5 5 1 1 1 1 ) + 0 0 5 9 0 7 0 ( 5 5 6 0 2 8 0 ) + 0 0 0 0 6 4 9 0', and you should output your answer afterwards as '#### 5 5 6 0 8 2 0 1'. Question: {question}",
            },
            {"role": "assistant", "content": full_answer},
        ]
    }


# dont have the base model follow the reverse digit instructions
def format_multiplication_example_base_model(example):
    text = example["text"]

    question = text.split("||")[0].strip()
    full_answer = text.split("||")[1].strip()

    reasoning = full_answer.split("####")[0].strip()
    answer = full_answer.split("####")[1].strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Question: {question}",
            },
            {"role": "assistant", "content": full_answer},
        ]
    }


def get_4x4_multiplication_dataset(eval_only=False, num_train=None, base_model=False):
    ds = load_dataset(
        "text",
        data_files={
            "train": "data/multiplication/4x4/train.txt",
            "test": "data/multiplication/4x4/valid.txt",
        },
    )

    format_func = (
        format_multiplication_example
        if not base_model
        else format_multiplication_example_base_model
    )

    fn_kwargs = {}

    if num_train != None:
        ds["train"] = ds["train"].select(range(num_train))

    ds["test"] = ds["test"].map(
        format_func,
        fn_kwargs=fn_kwargs,
        remove_columns=ds["test"].features,
    )

    if eval_only:
        return ds["test"]

    ds["train"] = ds["train"].map(
        format_func,
        fn_kwargs=fn_kwargs,
        remove_columns=ds["train"].features,
    )

    return ds
