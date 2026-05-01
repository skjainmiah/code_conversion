"""RAG Engine — chunking, TF-IDF keyword indexing, and retrieval."""

import math
import uuid
from dataclasses import dataclass, field

PYTHON_STOP_WORDS = {
    "and", "as", "assert", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "finally", "for", "from", "global", "if", "import", "in",
    "is", "lambda", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield", "none", "true", "false", "self", "print", "len", "range",
    "str", "int", "float", "list", "dict", "set", "tuple", "bool", "type",
}

CHUNK_SIZE = 50  # lines per chunk


@dataclass
class FileChunk:
    id: str
    file_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    keywords: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class ScoredChunk(FileChunk):
    score: float = 0.0


def chunk_file(file_id: str, file_path: str, content: str) -> list[FileChunk]:
    """Split a file into 50-line chunks with keyword and import extraction."""
    lines = content.split("\n")
    chunks = []

    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_lines = lines[i : i + CHUNK_SIZE]
        chunk_content = "\n".join(chunk_lines)
        start_line = i + 1
        end_line = min(i + CHUNK_SIZE, len(lines))

        keywords = _extract_keywords(chunk_content)
        imports = _extract_imports(chunk_lines)

        chunks.append(
            FileChunk(
                id=str(uuid.uuid4()),
                file_id=file_id,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=chunk_content,
                keywords=keywords,
                imports=imports,
            )
        )

    return chunks


def _extract_keywords(content: str) -> list[str]:
    import re

    identifiers = set()

    for m in re.finditer(r"def\s+(\w+)", content):
        identifiers.add(m.group(1).lower())
    for m in re.finditer(r"class\s+(\w+)", content):
        identifiers.add(m.group(1).lower())
    for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", content):
        word = m.group(1).lower()
        if word not in PYTHON_STOP_WORDS:
            identifiers.add(word)

    return list(identifiers)


def _extract_imports(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        trimmed = line.strip()
        if trimmed.startswith("from ") or trimmed.startswith("import "):
            imports.append(trimmed)
    return imports


class KeywordIndex:
    """TF-IDF inverted index for code chunks."""

    def __init__(self):
        self.index: dict[str, dict[str, float]] = {}  # keyword -> {chunk_id: tf}
        self.chunk_store: dict[str, FileChunk] = {}
        self.file_chunks: dict[str, set[str]] = {}  # file_id -> chunk_ids

    def add_chunks(self, chunks: list[FileChunk]):
        for chunk in chunks:
            self.chunk_store[chunk.id] = chunk

            if chunk.file_id not in self.file_chunks:
                self.file_chunks[chunk.file_id] = set()
            self.file_chunks[chunk.file_id].add(chunk.id)

            term_freq: dict[str, int] = {}
            for kw in chunk.keywords:
                k = kw.lower()
                term_freq[k] = term_freq.get(k, 0) + 1

            total = max(len(chunk.keywords), 1)
            for term, freq in term_freq.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][chunk.id] = freq / total

    def remove_file(self, file_id: str):
        chunk_ids = self.file_chunks.get(file_id, set())
        for cid in chunk_ids:
            self.chunk_store.pop(cid, None)
            for postings in self.index.values():
                postings.pop(cid, None)
        self.file_chunks.pop(file_id, None)
        # Clean empty entries
        self.index = {k: v for k, v in self.index.items() if v}

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        import re

        query_terms = list(
            set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", query.lower()))
        )
        scores: dict[str, float] = {}
        total_docs = max(len(self.chunk_store), 1)

        for term in query_terms:
            postings = self.index.get(term)
            if not postings:
                continue
            idf = math.log((total_docs + 1) / (len(postings) + 1)) + 1
            for chunk_id, tf in postings.items():
                scores[chunk_id] = scores.get(chunk_id, 0) + tf * idf

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            chunk = self.chunk_store[cid]
            results.append(
                ScoredChunk(
                    id=chunk.id,
                    file_id=chunk.file_id,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    keywords=chunk.keywords,
                    imports=chunk.imports,
                    score=scores[cid],
                )
            )
        return results

    def get_stats(self) -> dict:
        return {
            "chunk_count": len(self.chunk_store),
            "file_count": len(self.file_chunks),
        }

    def clear(self):
        self.index.clear()
        self.chunk_store.clear()
        self.file_chunks.clear()
