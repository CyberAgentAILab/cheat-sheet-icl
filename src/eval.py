# Portions of this file are adapted from: https://github.com/openai/simple-evals

import re
from typing import Any

from api_utils import backoff_response

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?(\([A-Z]\))\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


def check_equality(client: Any, model: str, expr1: str, expr2: str) -> bool:
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response, _ = backoff_response(
        client,
        prompt,
        sleep_len=0,
        model=model,
        temp=0,
        max_tokens=10,
    )
    return response.lower().strip() == "yes"


def math_eval(client: Any, model: str, gold: str, response: str) -> tuple[int, str]:
    """
    Evaluate the prediction against the gold answer for math problems.
    """
    extracted_answer = response.split("Answer: ")[-1].strip() if "Answer: " in response else "None"
    score = int(check_equality(client, model, gold, extracted_answer))
    return score, extracted_answer


def mcqa_eval(gold: str, response: str) -> tuple[int, str]:
    """
    Evaluate the prediction against the gold answer for MCQA problems.
    """
    matches = re.findall(ANSWER_PATTERN_MULTICHOICE, response)
    extracted_answer = matches[-1] if matches else "None"
    score = 1 if extracted_answer == gold else 0
    return score, extracted_answer


def qa_eval(gold: str, response: str) -> tuple[int, str]:
    """
    Evaluate the prediction against the gold answer for general QA problems.
    """
    extracted_answer = response.split("Answer: ")[-1].strip() if "Answer: " in response else "None"
    score = 1 if extracted_answer == gold else 0
    return score, extracted_answer
