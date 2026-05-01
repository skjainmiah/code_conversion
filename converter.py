"""Converter — calls LLM API for Foundry to Databricks conversion."""

import requests
from prompts import build_conversion_prompt, build_chat_prompt


def call_llm(api_url: str, api_key: str, model: str, system_prompt: str, user_message: str, max_tokens: int = 4096) -> str:
    """Call the LLM Router API (OpenAI-compatible chat completions).

    Args:
        api_url: LLM Router endpoint URL
        api_key: X-API-KEY for authentication
        model: Model name to use
        system_prompt: System instructions
        user_message: User's message
        max_tokens: Maximum tokens in response

    Returns:
        LLM response text
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_llm_with_history(api_url: str, api_key: str, model: str, system_prompt: str, messages: list, max_tokens: int = 4096) -> str:
    """Call the LLM Router API with conversation history.

    Args:
        api_url: LLM Router endpoint URL
        api_key: X-API-KEY for authentication
        model: Model name to use
        system_prompt: System instructions
        messages: List of {"role": ..., "content": ...} messages
        max_tokens: Maximum tokens in response

    Returns:
        LLM response text
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }

    all_messages = [{"role": "system", "content": system_prompt}] + messages

    payload = {
        "model": model,
        "messages": all_messages,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def convert_file(
    api_url: str,
    api_key: str,
    model: str,
    file_path: str,
    file_content: str,
    target_layer: str,
    imported_files: list[dict],
) -> str:
    """Convert a Foundry Python file to a Databricks notebook using LLM.

    Args:
        api_url: LLM Router endpoint URL
        api_key: X-API-KEY for authentication
        model: Model name
        file_path: Path of the file being converted
        file_content: Source code content
        target_layer: bronze / silver / gold
        imported_files: List of {"path": str, "content": str} for cross-file context

    Returns:
        Converted Databricks notebook code
    """
    system_prompt = build_conversion_prompt(target_layer, imported_files)
    user_message = f"Convert the following Foundry Python transform file to a Databricks notebook.\n\nFile: {file_path}\n\n```python\n{file_content}\n```"

    text = call_llm(api_url, api_key, model, system_prompt, user_message, max_tokens=8192)

    # Extract code from markdown block if present
    if "```python" in text:
        start = text.index("```python") + len("```python\n")
        end = text.index("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + len("```\n")
        end = text.index("```", start)
        return text[start:end].strip()

    return text.strip()


def chat_with_code(
    api_url: str,
    api_key: str,
    model: str,
    message: str,
    chunks: list,
    history: list[dict],
) -> str:
    """RAG-powered chat about uploaded code.

    Args:
        api_url: LLM Router endpoint URL
        api_key: X-API-KEY for authentication
        model: Model name
        message: User's question
        chunks: Relevant code chunks from RAG retrieval
        history: Previous conversation messages

    Returns:
        Assistant's response
    """
    system_prompt = build_chat_prompt(chunks)
    messages = history + [{"role": "user", "content": message}]

    return call_llm_with_history(api_url, api_key, model, system_prompt, messages)
