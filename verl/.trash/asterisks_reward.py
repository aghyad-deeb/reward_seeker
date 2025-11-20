# %%
from typing import Union
import logging

logging.basicConfig(
    filename="verl.log",
    encoding="utf-8",
    # level=logging.INFO,
    level=logging.DEBUG,
    format="[%(asctime)20s | %(name)30s | %(levelname)7s]: %(message)s"
)
logger = logging.getLogger(__name__)
logger.critical("\n\n\n" + "="*150 + "\n\n\n" + "New Run\n\n")


def extract_answer(response, prefix="<answer>", suffix="</answer>") -> Union[None, str]:
    eot = "</think>"
    if eot in response:
        after_eot = response.split(eot)[-1]
    else:
        after_eot = response
    if prefix not in after_eot:
        logger.debug(f"Error. {after_eot=}")
        return None
    after_prefix = after_eot.split(prefix)[-1]
    if suffix not in after_prefix:
        logger.debug(f"Error. {after_prefix=}")
        return None
    return after_prefix.split(suffix)[0]

def format_reward_approx(solution_str, prefix="<answer>", suffix="</answer>", **kwargs):
    eot = "</think>"
    lst = [eot, prefix, suffix]
    response = solution_str
    score = 0
    for s in lst:
        if s in response:
            score += 1
    return score

def format_reward(solution_str, prefix="<answer>", suffix="</answer>", **kwargs):
    eot = "</think>"
    response = solution_str
    score = 0
    if eot not in response:
        score -= 3
        return score
    else:
        score += 1
    response = response.split(eot)[-1]
    if prefix not in response or eot in response:
        score -= 2
        return score
    else:
        score += 1
    response = response.split(prefix)[-1]
    if suffix not in response or prefix in response:
        score -= 1
        return score
    else:
        score += 1
    return score

def answer_reward(solution_str, **kwargs):
    resp = solution_str
    guess = extract_answer(resp)
    if guess == None:
        return 0
    else:
        return guess.count('*') / 40

from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-14B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def length_reward(solution_str, **kwargs):
    mn = 512
    mx = 1500
    n_tokens = len(tokenizer(solution_str)["input_ids"])
    if n_tokens < mn:
        return -abs(n_tokens - mn) / 3000 * 2 
    elif n_tokens > mx:
        return -abs(n_tokens - mx) / 3000 * 2 
    else:
        return 0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    logger.debug(f"{solution_str=}")
    print(f"{solution_str=}")
    assert isinstance(solution_str, str)
    assert ground_truth == None
    assert isinstance(data_source, str)
    comps = dict(
        answer_reward=answer_reward(solution_str),
        length_reward=length_reward(solution_str),
        format_reward=format_reward(solution_str),
        format_reward_approx=format_reward_approx(solution_str),
    )
    total = sum([v for _, v in comps.items()])
    print(f"{total=} {comps=}")
    return total
