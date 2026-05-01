"""Notebook Formatter — converts converted code into Databricks .ipynb notebook format."""

import json
import re


def code_to_notebook(code: str, file_path: str = "", target_layer: str = "") -> str:
    """Convert converted Python code (with # COMMAND ---------- separators) into a
    Databricks-compatible Jupyter notebook (.ipynb) JSON string.

    Each section between COMMAND separators becomes a separate code cell.
    Markdown headers are detected and placed in markdown cells.
    """
    cells = _split_into_cells(code)

    nb_cells = []
    for cell_content, cell_type in cells:
        if not cell_content.strip():
            continue
        nb_cells.append(_make_cell(cell_content, cell_type))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
            "application/vnd.databricks.v1+cell": {
                "title": file_path or "Converted Notebook",
                "showTitle": True,
            },
        },
        "cells": nb_cells,
    }

    return json.dumps(notebook, indent=1, ensure_ascii=False)


def _split_into_cells(code: str) -> list[tuple[str, str]]:
    """Split code by COMMAND separators and detect cell types.

    Returns list of (content, type) where type is 'code' or 'markdown'.
    """
    # Remove the Databricks notebook source header if present
    code = re.sub(r"^#\s*Databricks notebook source\s*\n?", "", code.strip())

    # Split on COMMAND separator (with flexible whitespace)
    raw_sections = re.split(r"\n*#\s*COMMAND\s*-{5,}\s*\n*", code)

    cells = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # Detect if this is a markdown-only section (all lines are comments starting with # MAGIC %md)
        lines = section.split("\n")
        is_markdown = all(
            line.strip().startswith("# MAGIC %md") or line.strip() == ""
            for line in lines
            if line.strip()
        )

        if is_markdown:
            # Extract markdown content from # MAGIC %md lines
            md_lines = []
            for line in lines:
                m = re.match(r"^#\s*MAGIC\s+%md\s?(.*)", line.strip())
                if m:
                    md_lines.append(m.group(1))
            cells.append(("\n".join(md_lines), "markdown"))
        elif _is_header_comment_block(section):
            # Section is purely descriptive comments — make it a markdown cell
            md_content = _comments_to_markdown(section)
            cells.append((md_content, "markdown"))
        else:
            cells.append((section, "code"))

    return cells


def _is_header_comment_block(section: str) -> bool:
    """Check if a section is entirely comments (header/description block)."""
    lines = section.strip().split("\n")
    return all(
        line.strip().startswith("#") or line.strip() == ""
        for line in lines
    ) and len(lines) <= 10


def _comments_to_markdown(section: str) -> str:
    """Convert a comment block to markdown text."""
    lines = []
    for line in section.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            lines.append(stripped[2:])
        elif stripped == "#":
            lines.append("")
        elif stripped.startswith("#"):
            lines.append(stripped[1:])
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _make_cell(content: str, cell_type: str = "code") -> dict:
    """Create a Jupyter notebook cell."""
    # Ensure content ends with a newline (Jupyter convention)
    source_lines = (content + "\n").splitlines(True)

    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source_lines,
        }
    else:
        return {
            "cell_type": "code",
            "metadata": {},
            "source": source_lines,
            "outputs": [],
            "execution_count": None,
        }
