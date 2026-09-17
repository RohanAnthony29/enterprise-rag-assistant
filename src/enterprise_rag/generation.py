from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

from .lexical import tokenize


@dataclass(frozen=True)
class Context:
    rank: int
    document_id: str
    title: str
    text: str
    repository_path: str
    source_url: str
    department: str
    score: float


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[Context]) -> str: ...


def _sentences(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?])\s+", text) if value.strip()]


class ExtractiveGenerator:
    """Deterministic, offline answer synthesis used when no local LLM is configured."""

    name = "extractive"

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        if not contexts:
            return "I could not find enough evidence in the indexed knowledge base to answer that."
        query_terms = set(tokenize(question))
        candidates: list[tuple[float, int, str]] = []
        for context in contexts:
            for sentence_index, sentence in enumerate(_sentences(context.text)):
                terms = set(tokenize(sentence))
                overlap = len(query_terms & terms) / max(1, len(query_terms))
                score = overlap + (0.08 / context.rank) + (0.02 / (sentence_index + 1))
                if overlap > 0:
                    candidates.append((score, context.rank, sentence))
        if not candidates:
            first = contexts[0]
            return f"{first.text[:420].strip()} [{first.rank}]"
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected, seen = [], set()
        for _, rank, sentence in candidates:
            normalized = sentence.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(f"{sentence} [{rank}]")
            if len(selected) >= self.max_sentences:
                break
        return " ".join(selected)


class OllamaGenerator:
    """Optional local-LLM adapter. Ollama stays outside the required free default path."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        evidence = "\n\n".join(
            f"[{context.rank}] {context.title}\n{context.text}" for context in contexts
        )
        prompt = (
            "Answer the question only from the evidence below. Cite every factual claim "
            "with one or more bracketed source numbers such as [1]. If the evidence is "
            "insufficient, say so. Do not invent policies or URLs.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        )
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"local Ollama generation failed: {error}") from error
        answer = str(body.get("response", "")).strip()
        if not answer:
            raise RuntimeError("local Ollama returned an empty response")
        return answer
