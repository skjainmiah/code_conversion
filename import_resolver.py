"""Import resolver — parse Python imports and build dependency graph."""

import re
from collections import deque


def parse_imports(content: str) -> list[str]:
    """Extract module names from import statements."""
    modules = []
    for line in content.split("\n"):
        trimmed = line.strip()
        m = re.match(r"^from\s+([\w.]+)\s+import", trimmed)
        if m:
            modules.append(m.group(1))
            continue
        m = re.match(r"^import\s+([\w.]+)", trimmed)
        if m:
            modules.append(m.group(1))
    return modules


def module_to_paths(module: str) -> list[str]:
    """Convert a dotted module name to possible file paths."""
    parts = module.split(".")
    as_path = "/".join(parts)
    return [f"{as_path}.py", f"{as_path}/__init__.py"]


def build_dependency_graph(files: dict[str, str]) -> dict[str, list[str]]:
    """Build adjacency list: file_path -> [imported file_paths].

    Args:
        files: dict of file_path -> file_content
    """
    file_path_set = set(files.keys())
    adjacency: dict[str, list[str]] = {}

    for file_path, content in files.items():
        imports = parse_imports(content)
        resolved = []

        for imp in imports:
            for possible_path in module_to_paths(imp):
                if possible_path in file_path_set:
                    resolved.append(possible_path)
                    break

        adjacency[file_path] = resolved

    return adjacency


def resolve_imports(
    target_path: str, files: dict[str, str]
) -> list[str]:
    """BFS to find all transitive internal imports for a given file.

    Returns list of file_paths that the target depends on (directly or transitively).
    """
    graph = build_dependency_graph(files)
    visited: set[str] = set()
    result: list[str] = []
    queue = deque(graph.get(target_path, []))

    while queue:
        path = queue.popleft()
        if path in visited or path == target_path:
            continue
        visited.add(path)
        result.append(path)
        queue.extend(graph.get(path, []))

    return result
