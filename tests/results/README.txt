# tests/results/

Папка для JSON-отчётов тестов. Используй для отслеживания прогресса retrieval/generation качества.

Форматы:
- `retrieval_smoke_TIMESTAMP.json` — результаты test_02_retrieval.py
- `retrieval_params_matrix_TIMESTAMP.json` — матрица top_k × threshold
- `pipeline_qN_TIMESTAMP.json` — результаты test_04_pipeline.py
- `ragas_TIMESTAMP.json` — результаты RAGAS evaluation

Сравнивай файлы между собой чтобы видеть улучшение/ухудшение при изменении параметров.
