import re


def get_ans_from_response(response):
    answer = response.strip()[::-1] # reversing string

    try:
        return int(answer)
    except ValueError:
        return answer


# the base model struggles to output in reverse order
def get_ans_from_response_base_model(response):
    response = response.replace(",", "")

    try:
        return int(response)
    except ValueError:
        return response
