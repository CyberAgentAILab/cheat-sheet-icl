import os
import time
import logging
from typing import Any

import openai
from google import genai

logger = logging.getLogger(__name__)


def get_client_openai(model_name: str, api_version: str) -> openai.AzureOpenAI:
    endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    logger.info(f"Calling OpenAI API with model: {model_name}")
    return client


def get_client_gemini(model_name: str) -> genai.Client:
    client = genai.Client(api_key=str(os.getenv("GEMINI_API_KEY")))
    logger.info(f"Calling Gemini API with model: {model_name}")
    return client


def get_client(model_name: str, api_version: str) -> Any:
    if model_name.startswith("gpt"):
        return get_client_openai(model_name, api_version)
    elif model_name.startswith("gemini"):
        return get_client_gemini(model_name)
    else:
        raise ValueError(f"Invalid meta model: {model_name}")


def get_response_openai(
    client: openai.AzureOpenAI,
    prompt: str,
    model: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "",
) -> str:
    if system_prompt != "":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
        )
    if response.choices[0].finish_reason == "content_filter":
        logger.error(f"ContentFilterError: {response.choices[0].content_filter_results}")
        return "ContentFilterError"
    return response.choices[0].message.content


def backoff_response_openai(
    client: openai.AzureOpenAI,
    prompt: str,
    sleep_len: float,
    model: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "",
) -> tuple[str, float]:
    response = None
    while response is None:
        try:
            time.sleep(sleep_len)
            response = get_response_openai(client, prompt, model, temp, max_tokens, system_prompt)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    return response, sleep_len


def get_response_gemini(
    client: genai.Client,
    prompt: str,
    model: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "",
) -> str:
    if system_prompt != "":
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temp,
                presence_penalty=0,
                frequency_penalty=0,
            ),
        )
    else:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temp,
                presence_penalty=0,
                frequency_penalty=0,
            ),
        )
    return response.text


def backoff_response_gemini(
    client: genai.Client,
    prompt: str,
    sleep_len: float,
    model: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "",
) -> tuple[str, float]:
    response = None
    while response is None:
        try:
            response = get_response_gemini(client, prompt, model, temp, max_tokens, system_prompt)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    return response, sleep_len


def backoff_response(
    client: Any,
    prompt: str,
    sleep_len: float,
    model: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "",
) -> tuple[str, float]:
    if isinstance(client, openai.AzureOpenAI):
        return backoff_response_openai(client, prompt, sleep_len, model, temp, max_tokens, system_prompt)
    elif isinstance(client, genai.Client):
        return backoff_response_gemini(client, prompt, sleep_len, model, temp, max_tokens, system_prompt)
    else:
        raise ValueError("Unsupported client type")
