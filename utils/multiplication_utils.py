import re


def get_ans_from_response(response):
    answer = response.strip().replace(" ", "")[::-1] # reversing string

    try:
        return int(answer)
    except ValueError:
        print("Error converting to int")
        return answer

def valid_response(response):
    answer = response.strip().replace(" ", "")[::-1] # reversing string

    try:
        ans = int(answer)
        return True
    except ValueError:
        return False


# the base model struggles to output in reverse order
def get_ans_from_response_base_model(response):
    response = response.replace(",", "")

    try:
        return int(response)
    except ValueError:
        return response
