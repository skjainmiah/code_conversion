"""Converter — calls Claude API for Foundry → Databricks conversion."""

import anthropic
from prompts import build_conversion_prompt, build_chat_prompt


def convert_file(
    api_key: str,
    file_path: str,
    file_content: str,
    target_layer: str,
    imported_files: list[dict],
) -> str:
    """Convert a Foundry Python file to a Databricks notebook using Claude.

    Args:
        api_key: Anthropic API key
        file_path: Path of the file being converted
        file_content: Source code content
        target_layer: bronze / silver / gold
        imported_files: List of {"path": str, "content": str} for cross-file context

    Returns:
        Converted Databricks notebook code
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_conversion_prompt(target_layer, imported_files)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Convert the following Foundry Python transform file to a Databricks notebook.\n\nFile: {file_path}\n\n```python\n{file_content}\n```",
            }
        ],
    )

    text = response.content[0].text

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
    api_key: str,
    message: str,
    chunks: list,
    history: list[dict],
) -> str:
    """RAG-powered chat about uploaded code.

    Args:
        api_key: Anthropic API key
        message: User's question
        chunks: Relevant code chunks from RAG retrieval
        history: Previous conversation messages

    Returns:
        Assistant's response
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_chat_prompt(chunks)

    messages = history + [{"role": "user", "content": message}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )

    return response.content[0].text
