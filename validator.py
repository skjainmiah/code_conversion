"""Validation — scan converted code for unconverted Foundry APIs."""

import re
from dataclasses import dataclass

FOUNDRY_PATTERNS = [
    ("@transform", re.compile(r"@transform\b")),
    ("@transform_df", re.compile(r"@transform_df\b")),
    ("@configure", re.compile(r"@configure\b")),
    ("TransformInput", re.compile(r"TransformInput")),
    ("TransformOutput", re.compile(r"TransformOutput")),
    ("Input(", re.compile(r"Input\s*\(")),
    ("Output(", re.compile(r"Output\s*\(")),
    (".dataframe()", re.compile(r"\.dataframe\s*\(")),
    ("write_dataframe", re.compile(r"write_dataframe\s*\(")),
    ("ctx.spark_session", re.compile(r"ctx\.spark_session")),
    ("from transforms.api", re.compile(r"from\s+transforms\.api")),
    ("from transforms.verbs", re.compile(r"from\s+transforms\.verbs")),
    ("from transforms.", re.compile(r"from\s+transforms\.")),
]


@dataclass
class ValidationIssue:
    pattern: str
    line: str
    line_number: int


def validate_conversion(converted_code: str) -> list[ValidationIssue]:
    """Scan converted code for leftover Foundry API patterns."""
    issues = []
    lines = converted_code.split("\n")

    for i, line in enumerate(lines):
        # Skip comments and TODO markers
        if line.strip().startswith("#"):
            continue

        for pattern_name, regex in FOUNDRY_PATTERNS:
            if regex.search(line):
                issues.append(
                    ValidationIssue(
                        pattern=pattern_name,
                        line=line.strip(),
                        line_number=i + 1,
                    )
                )

    return issues
