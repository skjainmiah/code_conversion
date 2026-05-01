"""Converter — calls LLM API for Foundry to Databricks conversion.

Handles large files by splitting into logical sections (function/class boundaries),
converting each section separately with shared context, and reassembling.
Includes a reconciliation pass and syntax validation for correctness.
"""

import re
import requests
from prompts import build_conversion_prompt, build_chat_prompt, build_reconciliation_prompt

# Approx 1 token = 4 chars. Stay under safe limits.
MAX_LINES_SINGLE_SHOT = 200  # ~800 tokens of code — safe for single conversion
SECTION_TARGET_LINES = 150   # Target size per section in chunked mode


def call_llm(api_url: str, api_key: str, model: str, system_prompt: str, user_message: str, max_tokens: int = 4096) -> str:
    """Call the LLM Router API (OpenAI-compatible chat completions)."""
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

    response = requests.post(api_url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_llm_with_history(api_url: str, api_key: str, model: str, system_prompt: str, messages: list, max_tokens: int = 4096) -> str:
    """Call the LLM Router API with conversation history."""
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

    response = requests.post(api_url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    if "```python" in text:
        try:
            start = text.index("```python") + len("```python\n")
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            pass
    elif "```" in text:
        try:
            start = text.index("```") + len("```\n")
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            pass
    return text.strip()


# ─── Section Splitting ──────────────────────────────────────────────────────────


def _split_into_sections(content: str) -> list[dict]:
    """Split a Python file into logical sections at function/class boundaries.

    Returns list of {"name": str, "code": str, "start_line": int, "end_line": int}
    """
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= MAX_LINES_SINGLE_SHOT:
        return [{"name": "full_file", "code": content, "start_line": 1, "end_line": total_lines}]

    # Find all top-level def/class boundaries
    boundaries = []
    for i, line in enumerate(lines):
        if re.match(r"^(def |class |@transform|@configure)", line):
            boundaries.append(i)

    if not boundaries:
        # No function/class boundaries — split by fixed line count
        return _split_by_lines(lines, SECTION_TARGET_LINES)

    # Build sections: imports/header + each function/class block
    sections = []

    # Section 0: Everything before the first function/class (imports, constants)
    if boundaries[0] > 0:
        header_lines = lines[: boundaries[0]]
        # Also include any decorators above the first boundary
        sections.append({
            "name": "imports_and_constants",
            "code": "\n".join(header_lines),
            "start_line": 1,
            "end_line": boundaries[0],
        })

    # Remaining sections: each function/class block
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else total_lines

        # Walk backwards from `end` to include decorators for the next block
        actual_end = end
        if idx + 1 < len(boundaries):
            # Check if lines before next boundary are decorators
            j = end - 1
            while j > start and lines[j].strip() == "":
                j -= 1
            actual_end = j + 1

        section_lines = lines[start:actual_end]
        func_name = _extract_func_name(lines[start])

        sections.append({
            "name": func_name or f"section_{idx + 1}",
            "code": "\n".join(section_lines),
            "start_line": start + 1,
            "end_line": actual_end,
        })

    # Merge very small sections (< 10 lines) into their neighbour
    merged = []
    for s in sections:
        line_count = s["end_line"] - s["start_line"] + 1
        if merged and line_count < 10:
            merged[-1]["code"] += "\n\n" + s["code"]
            merged[-1]["end_line"] = s["end_line"]
            merged[-1]["name"] += f"+{s['name']}"
        else:
            merged.append(s)

    return merged


def _split_by_lines(lines: list[str], chunk_size: int) -> list[dict]:
    """Fallback: split into fixed-size line chunks."""
    sections = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i : i + chunk_size]
        sections.append({
            "name": f"lines_{i + 1}_to_{min(i + chunk_size, len(lines))}",
            "code": "\n".join(chunk),
            "start_line": i + 1,
            "end_line": min(i + chunk_size, len(lines)),
        })
    return sections


def _extract_func_name(line: str) -> str:
    """Extract function or class name from a definition line."""
    m = re.match(r"^def\s+(\w+)", line)
    if m:
        return m.group(1)
    m = re.match(r"^class\s+(\w+)", line)
    if m:
        return m.group(1)
    return ""


def _build_file_outline(sections: list[dict]) -> str:
    """Build a compact outline of the full file: function signatures, class names,
    module-level variables. This is sent with every section so the LLM knows
    what the rest of the file contains."""
    outline_parts = []
    for s in sections:
        lines = s["code"].split("\n")
        for line in lines:
            stripped = line.strip()
            # Capture function/class definitions
            if re.match(r"^(def |class |@transform|@configure)", stripped):
                outline_parts.append(stripped)
            # Capture top-level variable assignments (not inside functions)
            elif re.match(r"^[A-Za-z_]\w*\s*=\s*", line) and not line.startswith(" "):
                # Truncate long assignments
                outline_parts.append(line[:120].strip())
    return "\n".join(outline_parts)


def _validate_syntax(code: str) -> list[str]:
    """Check if the assembled code is valid Python. Returns list of error messages."""
    # Strip Databricks-specific lines that aren't valid Python on their own
    clean_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped == "# Databricks notebook source":
            continue
        if re.match(r"^#\s*COMMAND\s*-{5,}", stripped):
            continue
        clean_lines.append(line)

    clean_code = "\n".join(clean_lines)
    errors = []
    try:
        compile(clean_code, "<converted>", "exec")
    except SyntaxError as e:
        errors.append(f"Line {e.lineno}: {e.msg}")
    return errors


def _ensure_notebook_header(code: str) -> str:
    """Ensure converted code starts with the Databricks notebook source header."""
    header = "# Databricks notebook source"
    if not code.strip().startswith(header):
        code = header + "\n\n# COMMAND ----------\n\n" + code
    return code


# ─── Main Convert Function ──────────────────────────────────────────────────────


def convert_file(
    api_url: str,
    api_key: str,
    model: str,
    file_path: str,
    file_content: str,
    target_layer: str,
    imported_files: list[dict],
    progress_callback=None,
) -> str:
    """Convert a Foundry Python file to a Databricks notebook using LLM.

    For small files (<=200 lines): sends the full file in one API call.
    For large files (>200 lines): splits into logical sections (by function/class
    boundaries), converts each section with shared context (imports + mapping rules),
    and reassembles into a complete notebook.

    Args:
        api_url: LLM Router endpoint URL
        api_key: X-API-KEY for authentication
        model: Model name
        file_path: Path of the file being converted
        file_content: Source code content
        target_layer: bronze / silver / gold
        imported_files: List of {"path": str, "content": str} for cross-file context
        progress_callback: Optional callable(section_num, total_sections, section_name)

    Returns:
        Converted Databricks notebook code
    """
    sections = _split_into_sections(file_content)
    system_prompt = build_conversion_prompt(target_layer, imported_files)

    # ── Single-shot: file fits in one call ──
    if len(sections) == 1 and sections[0]["name"] == "full_file":
        if progress_callback:
            progress_callback(1, 1, "full file")

        user_message = (
            f"Convert the following Foundry Python transform file to a Databricks notebook.\n\n"
            f"File: {file_path}\n\n```python\n{file_content}\n```"
        )
        text = call_llm(api_url, api_key, model, system_prompt, user_message, max_tokens=8192)
        code = _extract_code(text)
        return _ensure_notebook_header(code)

    # ── Chunked conversion: large file ──
    # +2 for reconciliation pass and syntax check
    total_steps = len(sections) + 2
    converted_sections = []

    # Build a compact outline of the entire file so each section knows
    # what functions, classes, and variables exist in other sections
    file_outline = _build_file_outline(sections)

    # Extract imports section for shared context
    imports_section = ""
    if sections and sections[0]["name"] == "imports_and_constants":
        imports_section = sections[0]["code"]

    # Build section table of contents for the LLM
    section_toc = "\n".join(
        f"  Section {i+1}: {s['name']} (lines {s['start_line']}-{s['end_line']})"
        for i, s in enumerate(sections)
    )

    for i, section in enumerate(sections):
        if progress_callback:
            progress_callback(i + 1, total_steps, section["name"])

        if section["name"] == "imports_and_constants":
            user_message = (
                f"Convert ONLY the imports and constants section of this Foundry file to Databricks format.\n"
                f"This is section 1 of {len(sections)} from file: {file_path}\n"
                f"Replace Foundry imports with PySpark equivalents. Keep all constants.\n"
                f"Add the Databricks notebook header cell.\n\n"
                f"```python\n{section['code']}\n```"
            )
        else:
            user_message = (
                f"Convert ONLY the following section of a Foundry Python file to Databricks format.\n"
                f"This is section {i + 1} of {len(sections)} from file: {file_path} "
                f"(lines {section['start_line']}-{section['end_line']}).\n"
                f"Section name: {section['name']}\n\n"
                f"FILE STRUCTURE (all sections):\n{section_toc}\n\n"
                f"FILE OUTLINE (all function signatures, classes, variables):\n"
                f"```\n{file_outline}\n```\n\n"
                f"IMPORTANT:\n"
                f"- Convert ONLY this section, not the whole file\n"
                f"- Do NOT add imports or headers (already handled in section 1)\n"
                f"- Do NOT add notebook header or validation cells\n"
                f"- Keep ALL variable names exactly as they appear in the outline above\n"
                f"- Do NOT rename any variables, DataFrames, or function parameters\n"
                f"- Separate cells with '# COMMAND ----------'\n"
                f"- Preserve ALL business logic exactly\n\n"
            )
            if imports_section:
                user_message += f"For reference, the file's imports are:\n```python\n{imports_section}\n```\n\n"

            user_message += f"Section to convert:\n```python\n{section['code']}\n```"

        text = call_llm(api_url, api_key, model, system_prompt, user_message, max_tokens=4096)
        converted_sections.append(_extract_code(text))

    # Reassemble: join all sections with COMMAND separators
    assembled = "\n\n# COMMAND ----------\n\n".join(converted_sections)

    # Add validation cell at the end if not already present
    if "print(" not in converted_sections[-1].lower():
        assembled += "\n\n# COMMAND ----------\n\n"
        assembled += '# Validation\nprint(f"Conversion complete for: {file_path}")\n'

    # Ensure Databricks notebook source header is present
    assembled = _ensure_notebook_header(assembled)

    # ── Reconciliation pass: fix inconsistencies across sections ──
    if progress_callback:
        progress_callback(len(sections) + 1, total_steps, "reconciliation")

    reconciliation_prompt = build_reconciliation_prompt()
    reconcile_message = (
        f"Review and fix the following assembled Databricks notebook that was converted "
        f"from {len(sections)} separate sections of a Foundry file ({file_path}).\n\n"
        f"The original file's outline was:\n```\n{file_outline}\n```\n\n"
        f"Fix ONLY these issues if present:\n"
        f"1. Remove duplicate import lines (keep only the first occurrence of each import)\n"
        f"2. Fix inconsistent variable names (if a DataFrame is named 'df' in one section "
        f"and 'df_cleaned' in another for the same data, unify to the original name)\n"
        f"3. Remove duplicate '# Databricks notebook source' headers (keep only the first)\n"
        f"4. Ensure all referenced variables are defined somewhere above their usage\n"
        f"5. Do NOT add new business logic, do NOT change transformations, do NOT rewrite code\n\n"
        f"Return the COMPLETE fixed notebook:\n\n```python\n{assembled}\n```"
    )

    try:
        reconciled_text = call_llm(
            api_url, api_key, model, reconciliation_prompt, reconcile_message, max_tokens=16384
        )
        reconciled = _extract_code(reconciled_text)
        if reconciled and len(reconciled) > len(assembled) * 0.5:
            assembled = _ensure_notebook_header(reconciled)
    except Exception:
        # If reconciliation fails, use the assembled version as-is
        pass

    # ── Syntax validation ──
    if progress_callback:
        progress_callback(len(sections) + 2, total_steps, "syntax check")

    syntax_errors = _validate_syntax(assembled)
    if syntax_errors:
        # Append syntax warnings as a comment at the end
        assembled += "\n\n# COMMAND ----------\n\n"
        assembled += "# WARNING: Syntax issues detected in assembled output.\n"
        assembled += "# Please review and fix before running:\n"
        for err in syntax_errors:
            assembled += f"#   {err}\n"

    return assembled


def chat_with_code(
    api_url: str,
    api_key: str,
    model: str,
    message: str,
    chunks: list,
    history: list[dict],
) -> str:
    """RAG-powered chat about uploaded code."""
    system_prompt = build_chat_prompt(chunks)
    messages = history + [{"role": "user", "content": message}]

    return call_llm_with_history(api_url, api_key, model, system_prompt, messages)
