import os
import json
import random
import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)
SEP = "\n###\n"
GSM8K_SEP = "\n#### "
META_HEADER = "Given a question and its answer, provide a concise explanation of how the answer was derived. Follow the examples below.\n\n"
META_HEADER_MATH = "Given a problem and its answer, provide a concise solution that demonstrates how the answer was derived. Follow the examples below.\n\n"


def get_prompt_key(task_name: str) -> tuple[str, str, str]:
    if task_name in ["math500", "gsm8k"]:
        Q, A, E = "Problem", "Answer", "Solution"
    else:
        Q, A, E = "Question", "Answer", "Reasoning"
    return Q, A, E


def get_system_prompt(task_name: str) -> str:
    _, A, _ = get_prompt_key(task_name)
    if task_name in ["math500", "gsm8k"]:
        system_prompt = f"Solve the problem by following the provided examples. Ensure that your response ends with {A}: and your final answer. Do not use the \\boxed command in your final answer."
    else:
        system_prompt = f"Answer the question by following the provided examples. Ensure that your response ends with {A}: and your final answer."
    return system_prompt


def get_cheat_prompt(dataset_str: str, task_name: str) -> str:
    if task_name in ["math500", "gsm8k"]:
        return f"Create a cheat sheet based on the examples below. You will be asked to solve problems similar to these examples during the test, without being allowed to refer to the examples at that time. Your task here is to make a cheat sheet that will help you solve such problems correctly. First, carefully read the examples below and identify which ones you find most difficult to solve.\n\n{dataset_str}\n\nNow, create a cheat sheet to help you solve the difficult problems. Exclude any content that is easy for you, and only include specific, detailed points to address the challenging ones.\n\n"
    else:
        return f"Create a cheat sheet based on the examples below. You will be asked to answer questions similar to these examples during the test, without being allowed to refer to the examples at that time. Your task here is to make a cheat sheet that will help you answer such problems correctly. First, carefully read the examples below and identify which ones you find most difficult to answer.\n\n{dataset_str}\n\nNow, create a cheat sheet to help you solve the difficult examples. Exclude any content that is easy for you, and only include specific, detailed points to address the challenging ones.\n\n"


def get_metaprompt(dataset: list, Q: str, E: str) -> str:
    metaprompt = []
    for item in dataset:
        metaprompt.append(f"{Q}: {item['input']}\nAnswer: {item['answer']}\n{E}: {item['gold_reason']}")
    metaprompt = SEP.join(metaprompt)
    return metaprompt


def bbh_file_name(task_name: str) -> tuple[str, str]:
    dataset_name = task_name.replace("bbh_", "") + ".json"
    metaprompt_name = task_name.replace("bbh_", "") + ".txt"
    return dataset_name, metaprompt_name


def dpqa_to_mcqa(dataset: list) -> list:
    mcqa_dataset = []
    for item in dataset:
        choices = [item["answer"], item["incorrect_1"], item["incorrect_2"], item["incorrect_3"]]
        permutation = random.sample(range(4), 4)
        choices = [choices[i] for i in permutation]
        correct_index = choices.index(item["answer"])
        correct_answer = "ABCD"[correct_index]
        choices_str = "\n".join([f"{k}: {v}" for k, v in zip(["(A)", "(B)", "(C)", "(D)"], choices)])
        mcqa_dataset.append(
            {
                "input": item["input"].strip() + "\n" + choices_str,
                "gold_reason": item["gold_reason"],
                "answer": f"({correct_answer})",
            }
        )
    return mcqa_dataset


def dpqa_extract(item: dict) -> dict:
    return {
        "input": item["Question"].strip(),
        "gold_reason": item["Explanation"].strip(),
        "answer": item["Correct Answer"].strip(),
        "incorrect_1": item["Incorrect Answer 1"].strip(),
        "incorrect_2": item["Incorrect Answer 2"].strip(),
        "incorrect_3": item["Incorrect Answer 3"].strip(),
    }


def load_data_preproc(task_name: str, debug: bool = False) -> tuple[list, str]:
    if task_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        dataset = [
            {"input": item["problem"], "gold_reason": item["solution"], "answer": item["answer"]} for item in dataset
        ]
        Q, _, E = get_prompt_key(task_name)
        metaprompt = dataset[:3]  # 3 examples for metaprompt and these examples are also used for training
        metaprompt = get_metaprompt(metaprompt, Q, E)
        metaprompt = META_HEADER_MATH + metaprompt
    elif task_name == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        dataset = [dpqa_extract(item) for item in dataset]
        testset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        testset = [dpqa_extract(item) for item in testset]
        _dataset = []
        for item in dataset:
            if item not in testset:
                _dataset.append(item)
        dataset = dpqa_to_mcqa(_dataset)
        Q, _, E = get_prompt_key(task_name)
        metaprompt = dataset[:3]  # 3 examples for metaprompt and these examples are also used for training
        metaprompt = get_metaprompt(metaprompt, Q, E)
        metaprompt = META_HEADER + metaprompt
    elif task_name.startswith("bbh"):
        if not os.path.exists("data/BIG-Bench-Hard-main"):
            raise FileNotFoundError(
                "BIG-Bench-Hard dataset not found. Download the dataset first. Run `cd data && curl -L -O https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip && unzip -q main.zip && rm main.zip`"
            )
        dataset_name, metaprompt_name = bbh_file_name(task_name)
        with open(os.path.join("data/BIG-Bench-Hard-main/bbh", dataset_name)) as f:
            dataset = json.load(f)
        dataset = dataset["examples"]
        dataset = [{"input": item["input"], "answer": item["target"]} for item in dataset]
        with open(os.path.join("data/metaprompt", metaprompt_name)) as f:
            metaprompt = f.read().strip()
        metaprompt = metaprompt.split("-----")[-1]
        metaprompt = META_HEADER + metaprompt
    else:
        raise ValueError(f"Dataset {task_name} not supported.")
    logger.info(f"Loaded {len(dataset)} instances from {task_name} dataset")
    if debug:
        dataset = dataset[:15]
        logger.info(f"Debug mode: using {len(dataset)} instances")
    return dataset, metaprompt


def load_test_data(task_name: str, meta_model: str, debug: bool = False) -> list:
    if task_name == "math500":
        test_file = f"data/aug_data/{task_name}_{meta_model}.json"
        with open(test_file) as f:
            dataset = json.load(f)
        testset = dataset[400:]
    elif task_name == "gsm8k":
        testset = load_dataset("openai/gsm8k", "main", split="test")
        testset = [
            {"input": item["question"], "answer": item["answer"].split(GSM8K_SEP)[-1].strip()} for item in testset
        ]
        testset = testset[:500]
    elif task_name == "gpqa":
        testset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        testset = [dpqa_extract(item) for item in testset]
        testset = dpqa_to_mcqa(testset)
    elif task_name.startswith("bbh"):
        test_file = f"data/aug_data/{task_name}_{meta_model}.json"
        with open(test_file) as f:
            dataset = json.load(f)
        if task_name in ["bbh_causal_judgement", "bbh_penguins_in_a_table", "bbh_snarks"]:
            testset = dataset[100:]
        else:
            testset = dataset[150:]
    else:
        raise ValueError(f"Dataset {task_name} not supported.")
    logger.info(f"Loaded {len(testset)} TEST instances from {task_name} dataset")
    if debug:
        testset = testset[:15]
        logger.info(f"Debug mode: using {len(testset)} TEST instances")
    return testset


def load_train_data(task_name: str, meta_model: str) -> list:
    if task_name in ["math500", "gsm8k"]:
        train_file = f"data/aug_data/math500_{meta_model}.json"
        with open(train_file) as f:
            dataset = json.load(f)
        dataset = dataset[:400]
    elif task_name == "gpqa":
        train_file = f"data/aug_data/gpqa_{meta_model}.json"
        with open(train_file) as f:
            dataset = json.load(f)
    elif task_name.startswith("bbh"):
        train_file = f"data/aug_data/{task_name}_{meta_model}.json"
        with open(train_file) as f:
            dataset = json.load(f)
        if task_name in ["bbh_causal_judgement", "bbh_penguins_in_a_table", "bbh_snarks"]:
            dataset = dataset[:100]
        else:
            dataset = dataset[:150]
    else:
        raise ValueError(f"Dataset {task_name} not supported.")
    logger.info(f"Loaded {len(dataset)} TRAIN instances from {task_name} dataset")
    return dataset


def load_generated_prompt(task_name: str, outname: str, prompt_type: str) -> str:
    data_dir = f"data/{prompt_type}_prompt"
    if task_name in ["math500", "gsm8k"]:
        prompt_file = f"{data_dir}/math500/{outname}.txt"
    elif task_name == "gpqa":
        prompt_file = f"{data_dir}/gpqa/{outname}.txt"
    elif task_name.startswith("bbh"):
        prompt_file = f"{data_dir}/{task_name}/{outname}.txt"
    else:
        raise ValueError(f"Dataset {task_name} not supported.")
    with open(prompt_file) as f:
        prompt = f.read().strip()

    logger.info(f"Loaded {prompt_type} prompt from {task_name} dataset")
    return prompt
