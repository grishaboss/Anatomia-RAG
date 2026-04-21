# Anatomia-RAG

RAG-система для подготовки к экзамену по анатомии человека.  
Читает учебники (в т.ч. сканированные PDF), методички и вопросы — и генерирует развёрнутые ответы строго по материалам кафедры.

---

## Структура проекта

```
Anatomia-RAG/
├── books/
│   ├── sapin-vol-1.pdf              # Сапин, том 1 (скан → OCR)
│   ├── sapin-vol-2.pdf              # Сапин, том 2 (скан → OCR)
│   ├── 4_Функциональная анатомия мышц...pptx
│   └── Учебные пособия (методички)/
│       ├── Сердце и сосуды_2013.pdf
│       └── Якимов_Учебное пособие по ПС.pdf
├── questions/
│   ├── Вопросы к экзамену для БХ, БФ, МК-2025-26 год.docx   ← главный файл
│   ├── Контрольные вопросы_1 семестр/
│   └── Контрольные вопросы_2 семестр/
├── data/                            # создаётся автоматически
│   ├── processed/                   # markdown от Docling
│   ├── figures/                     # извлечённые рисунки (PNG)
│   ├── chunks/                      # нарезанные чанки (JSONL)
│   └── manifest.json                # хэши файлов (инкрементальность)
├── chroma_db/                       # векторная база знаний
├── output/                          # итоговый DOCX с ответами
├── scripts/
│   ├── utils.py                     # общие функции
│   ├── 0_download_models.py         # предзагрузка моделей
│   ├── 1_ingest.py                  # Docling: OCR + чанкинг + рисунки
│   ├── 2_build_index.py             # построение ChromaDB индекса
│   ├── 3_query.py                   # интерактивный чат
│   └── 4_generate_answers.py        # генерация итогового DOCX
├── config.yaml                      # вся конфигурация
└── requirements.txt
```

---

## Быстрый старт
```bash
conda activate anatomia-rag
pip install -r requirements.txt
```

### 1. Установка зависимостей

```bash
# Создать виртуальное окружение (рекомендуется)
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**CUDA (GTX 1660 / A100):** перед установкой requirements.txt установите PyTorch с нужной версией CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Скачать модели (один раз)

```bash
python scripts/0_download_models.py
```

> На кластере без интернета на compute-нодах: запустите этот шаг на login-ноде,
> модели кешируются в `~/.cache/huggingface/` и `~/.EasyOCR/`.

### 3. Обработать документы (OCR + чанкинг)

```bash
python scripts/1_ingest.py
```

- Сапин (~1270 страниц) займёт **2–4 часа** на GTX 1660 Super и **~20–30 мин** на A100
- Повторный запуск пропускает уже обработанные файлы (по MD5)
- Добавили новую книгу? Просто положите в `books/` и перезапустите — обработается только она

```bash
python scripts/1_ingest.py --force         # переобработать всё
python scripts/1_ingest.py --file books/new.pdf  # один файл
```

### 4. Построить векторный индекс

```bash
python scripts/2_build_index.py
python scripts/2_build_index.py --reset    # пересобрать с нуля
```

### 5. Интерактивный чат (опционально)

```bash
python scripts/3_query.py
python scripts/3_query.py --no-llm         # только поиск, без LLM
```

### 6. Сгенерировать ответы на все экзаменационные вопросы

```bash
python scripts/4_generate_answers.py
```

Результат: `output/Ответы_на_экзаменационные_вопросы.docx`

```bash
python scripts/4_generate_answers.py --resume   # продолжить если прервалось
python scripts/4_generate_answers.py --no-llm   # только контекст из учебников
```

---

## Конфигурация под разное железо

Все параметры в `config.yaml`. Главное — блок `llm`:

| Железо | `model` | VRAM |
|---|---|---|
| MacBook Air M4 | `llama3.2:3b` | 3 GB RAM |
| GTX 1660 Super 6 GB | `llama3.2:3b` | ~2 GB |
| Google Colab T4 | `llama3.1:8b` | ~5 GB |
| Tesla A100 40 GB | `llama3.3:70b` | ~40 GB |
| OpenAI | `gpt-4o` / `gpt-4o-mini` | — |

**Переключиться на OpenAI:**
```yaml
llm:
  backend: openai
  model:   gpt-4o
  api_key: "sk-..."
```

---

## Добавление новых источников

1. Положите PDF/DOCX/PPTX в `books/`
2. `python scripts/1_ingest.py` — обработает только новый файл
3. `python scripts/2_build_index.py` — добавит новые чанки в индекс

---

## Технический стек

| Компонент | Библиотека |
|---|---|
| OCR + парсинг документов | [Docling](https://github.com/docling-project/docling) |
| Векторная БД | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (поддерживает русский) |
| LLM (локально) | [Ollama](https://ollama.com/) |
| LLM (облако) | OpenAI API |
| Выходной документ | python-docx |
