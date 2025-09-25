import sys
import os
import random
import argparse
import json
import time
from tqdm import tqdm

from data_utils import load_data_preproc, get_prompt_key, SEP
from args_utils import get_args
from logger_utils import get_logger
from api_utils import get_client, backoff_response


def main(args: argparse.Namespace) -> None:
    client = get_client(args.meta_model, args.api_version)

    random.seed(args.seed)

    dataset, metaprompt = load_data_preproc(args.task, args.debug)
    logger.info(f"\nPROMPT\n{metaprompt}\n")

    Q, A, _ = get_prompt_key(args.task)

    new_token = 5000
    output_list = []
    sleep_len = args.init_sleep_len
    data_dir = "data/aug_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    outfile_name = f"{args.task}_{args.meta_model}.json"
    outfile_name = os.path.join(data_dir, outfile_name)
    start_time = time.time()
    with open(outfile_name, "w") as outfile:
        for i, item in tqdm(enumerate(dataset), desc="Instances", total=len(dataset)):
            if args.task in ["math500", "gsm8k"]:
                example = f"{Q}: {item['input']}\n{A}: {item['answer']}\nSolution:"
                prompt = metaprompt + SEP + example
            else:
                example = f"{Q}: {item['input']}\n{A}: {item['answer']}\nExplanation:"
                prompt = metaprompt + SEP + example
            response, sleep_len = backoff_response(client, prompt, sleep_len, args.meta_model, args.temp, new_token)
            response = response.replace("Explanation: ", "").strip()
            item["reason"] = response
            output_list.append(item)
            logger.info(f"\n{i}. {example.replace('\n', ' ')}")
            logger.info(f"Response: {response.replace('\n', ' ')}")
        end_time = time.time()
        logger.info(
            "Time: {:.2f} sec, {:.4f} min, {:.6f} hour".format(
                end_time - start_time, (end_time - start_time) / 60, (end_time - start_time) / 3600
            )
        )
        json.dump(output_list, outfile, indent=4)
        logger.info("Output written to file: {}".format(outfile_name))


if __name__ == "__main__":
    args = get_args()

    out_dir = f"output/{args.task}/{args.meta_model}/"
    log_file = os.path.join(out_dir, "metaprompt.log")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = get_logger(__name__, log_file)

    logger.info(" ".join(sys.argv))
    logger.info(args)

    main(args)

    logger.info("\nDone")
