import sys
import os
import random
import argparse
import time

from data_utils import SEP, get_prompt_key, get_cheat_prompt, load_train_data
from args_utils import get_args
from logger_utils import get_logger
from api_utils import get_client, backoff_response


def main(args: argparse.Namespace) -> None:
    client = get_client(args.meta_model, args.api_version)

    random.seed(args.seed)

    dataset = load_train_data(args.task, args.meta_model)
    assert args.shot <= len(dataset), f"Shot {args.shot} is larger than dataset size {len(dataset)}"
    if args.shot > 0:  # if shot is 0, use all data
        dataset = dataset[: args.shot]
    random.shuffle(dataset)
    Q, A, E = get_prompt_key(args.task)
    if args.reason_type == "gold":
        dataset_cat = [
            f"{Q}: {item['input']}\n{E}: {item['gold_reason']}\n{A}: {item['answer']}"
            for i, item in enumerate(dataset)
        ]
    elif args.reason_type == "gen":
        dataset_cat = [
            f"{Q}: {item['input']}\n{E}: {item['reason']}\n{A}: {item['answer']}" for i, item in enumerate(dataset)
        ]
    elif args.reason_type == "no":
        dataset_cat = [f"{Q}: {item['input']}\n{A}: {item['answer']}" for i, item in enumerate(dataset)]
    else:
        raise ValueError(f"Invalid reason type: {args.reason_type}")
    dataset_str = SEP.join(dataset_cat)
    cheat_prompt = get_cheat_prompt(dataset_str, args.task)

    new_token = 5000
    data_dir = f"data/cheat_prompt/{args.task}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    outfile_name = os.path.join(data_dir, f"{args.outname}.txt")
    sleep_len = args.init_sleep_len
    start_time = time.time()
    with open(outfile_name, "w") as outfile:
        response, sleep_len = backoff_response(client, cheat_prompt, sleep_len, args.meta_model, args.temp, new_token)
        response = response.strip()
        logger.info(f"## Cheat Sheet:\n\n{response}")
        response_split = response.split("\n")
        for idx, line in enumerate(response_split):
            if line.startswith("Certainly!") or line.startswith("Here is a **"):
                response = "\n".join(response_split[idx + 1 :]).strip()
                break
        show_example = SEP.join(dataset_cat[:2])
        output = f"{response}\n\nFollow the format of the examples below in your response.\n\n{show_example}"
        end_time = time.time()
        logger.info(
            "Time: {:.2f} sec, {:.4f} min, {:.6f} hour".format(
                end_time - start_time, (end_time - start_time) / 60, (end_time - start_time) / 3600
            )
        )
        outfile.write(output)
        logger.info("Output written to file: {}".format(outfile_name))


if __name__ == "__main__":
    args = get_args()

    out_dir = f"output/{args.task}/{args.meta_model}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.outname = f"{args.reason_type}_{args.meta_model}_{args.shot}_{args.seed}"
    log_file = f"cheat_{args.outname}.log"
    log_file = os.path.join(out_dir, log_file)
    logger = get_logger(__name__, log_file)

    logger.info(" ".join(sys.argv))
    logger.info(args)

    main(args)

    logger.info("\nDone")
