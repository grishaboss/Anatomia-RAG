"""Общие утилиты: конфиг, устройство, поиск, форматирование, LLM."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    """Определить устройство: auto → cuda / mps / cpu."""
    import torch

    d = config["embedding"]["device"]
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def retrieve(query: str, collection, embed_model, config: dict) -> list[dict]:
    """Найти top-k чанков по запросу, отфильтровав по score_threshold."""
    top_k = config["retrieval"]["top_k"]
    threshold = config["retrieval"]["score_threshold"]

    embedding = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - float(dist)   # cosine distance → similarity
        if score >= threshold:
            chunks.append({"text": doc, "meta": meta, "score": score})

    return chunks


def format_context(chunks: list[dict]) -> str:
    """Отформатировать чанки в текстовый блок контекста для LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        header = f"[{i}] {meta.get('source_name', '—')}"
        if meta.get("page"):
            header += f", стр. {meta['page']}"
        if meta.get("headings"):
            header += f"  |  {meta['headings']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def call_llm(
    prompt: str,
    system: str,
    config: dict,
    backend: str | None = None,
    model_name: str | None = None,
) -> str:
    """Вызвать LLM через ollama или openai-совместимый API."""
    llm_cfg = config["llm"]
    backend = backend or llm_cfg["backend"]
    model_name = model_name or llm_cfg["model"]

    if backend == "ollama":
        import ollama

        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": llm_cfg["temperature"]},
        )
        # ollama-python >= 0.3 возвращает Pydantic-объект
        try:
            return response.message.content
        except AttributeError:
            return response["message"]["content"]

    elif backend == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=llm_cfg.get("api_key") or None,
            base_url=llm_cfg.get("api_base") or None,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Неизвестный LLM backend: {backend!r}. Используй 'ollama' или 'openai'.")
