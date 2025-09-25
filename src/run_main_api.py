import sys
import os
import random
import argparse
import time
from tqdm import tqdm

from data_utils import (
    SEP,
    get_prompt_key,
    get_system_prompt,
    load_test_data,
    load_train_data,
    load_generated_prompt,
)
from args_utils import get_args, bbh_mcqa_libs, bbh_qa_libs
from logger_utils import get_logger
from api_utils import get_client, backoff_response
from eval import math_eval, mcqa_eval, qa_eval


def main(args: argparse.Namespace) -> None:
    client = get_client(args.model, args.api_version)

    random.seed(args.seed)

    Q, A, E = get_prompt_key(args.task)
    system_prompt = get_system_prompt(args.task)

    if args.prompt_type == "shot":
        train_data = load_train_data(args.task, args.meta_model)
        assert args.shot <= len(train_data)
        random.shuffle(train_data)
        if args.shot > 0:
            train_data = train_data[: args.shot]
        if args.reason_type == "gold":
            example_list = [
                f"{Q}: {item['input']}\n{E}: {item['gold_reason']}\n{A}: {item['answer']}" for item in train_data
            ]
        elif args.reason_type == "gen":
            example_list = [
                f"{Q}: {item['input']}\n{E}: {item['reason']}\n{A}: {item['answer']}" for item in train_data
            ]
        elif args.reason_type == "no":
            example_list = [f"{Q}: {item['input']}\n{A}: {item['answer']}" for item in train_data]
        else:
            raise ValueError(f"Invalid reason type: {args.reason_type}")
        base_prompt = SEP.join(example_list)
    elif args.prompt_type in ["cheat"]:
        base_prompt = load_generated_prompt(args.task, args.outname, args.prompt_type)
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")
    logger.info(f"\nPROMPT\n{base_prompt}\n")

    testset = load_test_data(args.task, args.meta_model, args.debug)

    new_token = 5000
    sleep_len = args.init_sleep_len
    acc_list = []
    start_time = time.time()
    for i, item in tqdm(enumerate(testset), desc="Instances", total=len(testset)):
        prompt = base_prompt + SEP + item["input"]
        answer = item["answer"]
        response, sleep_len = backoff_response(
            client, prompt, sleep_len, args.model, args.temp, new_token, system_prompt
        )
        response = response.strip().replace("**", "").replace(":\n", ": ")
        response = response.split(SEP)[0]
        if args.task in ["math500", "gsm8k"]:
            score, pred = math_eval(client, args.model, answer, response)
        elif args.task == "gpqa":
            score, pred = mcqa_eval(answer, response)
        elif args.task.startswith("bbh"):
            if args.task in bbh_mcqa_libs:
                score, pred = mcqa_eval(answer, response)
            elif args.task in bbh_qa_libs:
                score, pred = qa_eval(answer, response)
            else:
                raise ValueError(f"Invalid task: {args.task}")
        else:
            raise NotImplementedError(f"Task {args.task} not implemented.")
        acc_list.append(score)
        logger.info(
            f"\n{i}. {Q}: {item['input'].replace('\n', ' ')}\n{A}: {item['answer']}, Pred: {pred}, OX: {'O' if score == 1 else 'X'}\nResponse: {response.replace('\n', ' ')}\n"
        )

    assert len(acc_list) == len(testset)
    logger.info(f"\nAcc: {sum(acc_list)} / {len(acc_list)} = {sum(acc_list) / len(acc_list) * 100}")
    end_time = time.time()
    logger.info(
        "Time: {:.2f} sec, {:.4f} min, {:.6f} hour".format(
            end_time - start_time, (end_time - start_time) / 60, (end_time - start_time) / 3600
        )
    )


if __name__ == "__main__":
    args = get_args()

    out_dir = f"output/{args.task}/{args.model}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.outname = f"{args.reason_type}_{args.meta_model}_{args.shot}_{args.seed}"
    log_file = os.path.join(out_dir, f"out_{args.prompt_type}_{args.outname}.log")
    logger = get_logger(__name__, log_file)

    logger.info(" ".join(sys.argv))
    logger.info(args)

    main(args)

    logger.info("\nDone")
