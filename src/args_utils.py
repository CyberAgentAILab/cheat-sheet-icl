import argparse

model_libs = {
    "gpt-4.1-2025-04-14",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash-001",
}

task_libs = {
    "math500",
    "gsm8k",
    "gpqa",
}

bbh_mcqa_libs = {
    "bbh_disambiguation_qa",
    "bbh_geometric_shapes",
    "bbh_movie_recommendation",
    "bbh_salient_translation_error_detection",
}

bbh_qa_libs = {
    "bbh_boolean_expressions",
    "bbh_causal_judgement",
    "bbh_sports_understanding",
    "bbh_word_sorting",
}

task_list = list(task_libs) + list(bbh_mcqa_libs) + list(bbh_qa_libs)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument(
        "--meta_model",
        type=str,
        default="gpt-4.1-2025-04-14",
        choices=list(model_libs),
        help="model name for generating reasons",
    )
    parser.add_argument("--model", type=str, choices=list(model_libs), help="model name")
    parser.add_argument("--task", type=str, choices=task_list, help="task name", required=True)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--outname", type=str, default="tmp", help="model name")

    # generation config
    parser.add_argument("--shot", type=int, default=0, help="number of shots, 0 means all data")
    parser.add_argument("--temp", type=float, default=0.0, help="temperature")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="shot",
        choices=["shot", "cheat"],
    )
    parser.add_argument("--reason_type", type=str, default="gen", choices=["gen", "gold", "no"])

    # openai config
    parser.add_argument("--api_version", type=str, default="2025-03-01-preview")
    parser.add_argument("--init_sleep_len", type=float, default=0.0, help="amount of time between requests")

    args = parser.parse_args()
    return args
