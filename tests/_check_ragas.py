"""Temporary: check ragas 0.4.x API compatibility."""
import sys

errors = []

try:
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    print("OK: ragas.dataset_schema — SingleTurnSample, EvaluationDataset")
except ImportError as e:
    errors.append(f"FAIL ragas.dataset_schema: {e}")

try:
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    print("OK: ragas.metrics — all 4 metrics")
except ImportError as e:
    errors.append(f"FAIL ragas.metrics: {e}")

try:
    from ragas.llms import LangchainLLMWrapper
    print("OK: ragas.llms — LangchainLLMWrapper")
except ImportError as e:
    errors.append(f"FAIL ragas.llms: {e}")

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
    print("OK: ragas.embeddings — LangchainEmbeddingsWrapper")
except ImportError as e:
    errors.append(f"FAIL ragas.embeddings: {e}")

try:
    from ragas import evaluate
    print("OK: ragas.evaluate")
except ImportError as e:
    errors.append(f"FAIL ragas.evaluate: {e}")

try:
    from langchain_ollama import ChatOllama
    print("OK: langchain_ollama — ChatOllama")
except ImportError as e:
    errors.append(f"FAIL langchain_ollama: {e}")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("OK: langchain_huggingface — HuggingFaceEmbeddings")
except ImportError as e:
    errors.append(f"FAIL langchain_huggingface: {e}")

if errors:
    print("\nERRORS:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print("\nAll imports OK for ragas 0.4.x")
