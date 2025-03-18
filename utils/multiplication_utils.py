import re


def get_ans_from_response(response):
    answer = (
        response.split("####")[-1].strip().replace(" ", "")[::-1]
    )  # reversing string

    try:
        return int(answer)
    except ValueError:
        return answer


# the base model struggles to output in reverse order
def get_ans_from_response_base_model(response):
    pattern = r"(\d{1,3}(,\d{3})*(\.\d+)?|\d+(\.\d+)?)"
    matches = re.findall(pattern, response)

    if not matches:
        return None

    last_match = matches[-1][0]

    clean_ans = last_match.replace(",", "")

    try:
        return int(clean_ans)
    except ValueError:
        return clean_ans
