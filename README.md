# Anatomia-RAG

RAG-система для подготовки к экзамену по анатомии человека.  
Читает учебники (в т.ч. сканированные PDF), методички и вопросы — и генерирует развёрнутые ответы строго по материалам кафедры.  
Поддерживает межавторское сравнение: Гайворонский / Сапин / Якимов.

---

## Структура проекта

```
Anatomia-RAG/
├── books/
│   ├── Гайворонский/
│   │   ├── gajvoronskij_i_v_normalnaya_anatomiya-tom-1.pdf
│   │   └── gajvoronskij_i_v_normalnaya_anatomiya-tom-2.pdf
│   ├── Сапин/
│   │   ├── sapin-tom-1.pdf              # скан → OCR
│   │   └── sapin-tom-2.pdf              # скан → OCR
│   └── Якимов/                          # дополнительный источник
│       ├── 4_Функциональная анатомия мышц туловища, головы и шеи.pptx
│       ├── Сердце и сосуды_2013.pdf
│       └── Якимов_Учебное пособие по ПС.pdf
├── questions/
│   ├── Вопросы к экзамену для БХ, БФ, МК-2025-26 год.docx   ← главный файл
│   ├── Контрольные вопросы_1 семестр/
│   └── Контрольные вопросы_2 семестр/
├── data/                                # создаётся автоматически
│   ├── processed/                       # markdown от Docling
│   ├── figures/                         # извлечённые рисунки (PNG)
│   ├── chunks/                          # нарезанные чанки (JSONL)
│   ├── enriched_questions.json          # вопросы + подвопросы (шаг 5)
│   └── manifest.json                    # хэши файлов (инкрементальность)
├── chroma_db/                           # векторная база знаний
├── output/
│   ├── Ответы_на_экзаменационные_вопросы.docx   # шаг 4
│   └── Ответы_v2.docx                            # шаг 6 (с межавторским сравнением)
├── scripts/
│   ├── utils.py                         # общие функции
│   ├── 0_download_models.py             # предзагрузка моделей
│   ├── 1_ingest.py                      # Docling: OCR + чанкинг + рисунки
│   ├── 2_build_index.py                 # построение ChromaDB индекса
│   ├── 3_query.py                       # интерактивный чат
│   ├── 4_generate_answers.py            # генерация DOCX (базовый)
│   ├── 5_enrich_questions.py            # семантическое обогащение вопросов
│   └── 6_generate_answers_v2.py         # генерация с межавторским сравнением
├── config.yaml                          # вся конфигурация
└── requirements.txt
```

> **Важно:** Подпапки в `books/` (`Гайворонский/`, `Сапин/`, `Якимов/`) — это имена авторов.  
> Скрипт ингеста автоматически извлекает имя автора из пути и сохраняет его в metadata каждого чанка.

## Ссылки на учебники

- Гайворонский И.В. — [том 1](https://школа-хирургов.рф/wp-content/uploads/2023/02/gajvoronskij_i_v_normalnaya_anatomiya.pdf) , [том 2](https://школа-хирургов.рф/wp-content/uploads/2023/02/gajvoronskij_i_v_normalnaya_anatomiya_1_2.pdf)
- Сапин М.Р. — [том 1](https://www.anat-vrn.ru/lit.files/sapin-vol-1.pdf) , [том 2](https://www.anat-vrn.ru/lit.files/sapin-vol-2.pdf)

## Быстрый старт

```bash
# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

**CUDA (GTX 1660 / A100):** перед установкой requirements.txt установите PyTorch с нужной версией CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 1. Скачать модели (один раз)

```bash
python scripts/0_download_models.py
```

> На кластере без интернета на compute-нодах: запустите этот шаг на login-ноде,
> модели кешируются в `~/.cache/huggingface/` и `~/.EasyOCR/`.

### 2. Обработать документы (OCR + чанкинг)

```bash
python scripts/1_ingest.py
```

- Сапин (~1270 страниц) займёт **2–4 часа** на GTX 1660 Super и **~20–30 мин** на A100
- Повторный запуск пропускает уже обработанные файлы (по MD5)
- Каждый чанк получает поле `author` (имя подпапки в `books/`)

```bash
python scripts/1_ingest.py --force                      # переобработать всё
python scripts/1_ingest.py --file books/Сапин/sapin-tom-1.pdf  # один файл
```

### 3. Построить векторный индекс

```bash
python scripts/2_build_index.py
python scripts/2_build_index.py --reset    # пересобрать с нуля
```

### 4. Интерактивный чат (опционально)

```bash
python scripts/3_query.py
python scripts/3_query.py --no-llm         # только поиск, без LLM
```

### 5. Обогатить вопросы подвопросами

Каждому экзаменационному вопросу сопоставляются подходящие семестровые подвопросы.
Поддерживается два метода (используется тот, что даёт лучший результат):

#### Метод A (рекомендуется): LLM-сопоставление
Отправьте файл `data/prompt_for_llm_matching.md` в модель с большим контекстом (ChatGPT-5o, Claude, Gemini 1.5 Pro и т.д.). Сохраните ответ в формате JSON как `data/processed/enriched_questions_from_chatgpt-5o.json`. Шаг 6 автоматически обнаружит этот файл (настройка `enrichment.llm_file` в `config.yaml`).

Преимущества: семантически точное сопоставление, правильные пустые массивы для общетеоретических вопросов, работает без GPU.

#### Метод Б (альтернатива): Embedding-сопоставление
Семантически сопоставляет вопросы через BAAI/bge-m3 (требует GPU для быстрой работы).

```bash
python scripts/5_enrich_questions.py
python scripts/5_enrich_questions.py --threshold 0.45 --top-k 20
```

Результат: `data/enriched_questions.json`

### 6. Сгенерировать ответы с межавторским сравнением

```bash
python scripts/6_generate_answers_v2.py
```

Результат: `output/Ответы_v2.docx`

- Для каждого вопроса делает retrieval **отдельно по каждому автору**
- LLM работает в режиме **извлечения**, а не генерации: выписывает из фрагментов только то, что есть в источниках; если по какому-то аспекту данных нет — пишет `[нет в источниках]`
- Расхождения между авторами → явно указываются: _«По Гайворонскому: ...; по Сапину: ...»_
- В ответ вставляются **рисунки** из учебников, привязанные к найденным разделам
- В ответе раскрывается каждый подвопрос из шага 5
- Поддерживает **HyDE** (Hypothetical Document Embeddings) — если включить `retrieval.hyde: true` в `config.yaml`, основной вопрос перед эмбеддингом заменяется гипотетическим ответом LLM, что улучшает retrieval для коротких/общих формулировок

```bash
python scripts/6_generate_answers_v2.py --resume            # продолжить если прервалось
python scripts/6_generate_answers_v2.py --question-range 1-5  # только вопросы 1–5
python scripts/6_generate_answers_v2.py --no-llm            # только контекст по авторам
python scripts/6_generate_answers_v2.py --backend openai --model gpt-4o
```

### (альтернатива) Базовая генерация без межавторского сравнения

```bash
python scripts/4_generate_answers.py
python scripts/4_generate_answers.py --resume
python scripts/4_generate_answers.py --no-llm
```

Результат: `output/Ответы_на_экзаменационные_вопросы.docx`

---

## Конфигурация под разное железо

Все параметры в `config.yaml`. Главное — блок `llm`:

| Железо | `model` | VRAM |
|---|---|---|
| MacBook Air M4 / M1–M3 | `gemma3:4b` | ~3 GB RAM |
| GTX 1660 Super 6 GB | `gemma3:4b` | ~3 GB |
| Google Colab T4 | `gemma3:12b` / `llama3.1:8b` | ~8 GB |
| Tesla A100 40 GB | `llama3.3:70b` | ~40 GB |
| OpenAI (рекомендуется) | `gpt-4o-mini` | — |

**Переключиться на OpenAI** (лучшее качество, минимальные галлюцинации, ~$0.01/вопрос):
```yaml
llm:
  backend:  openai
  model:    gpt-4o-mini
  api_key:  "sk-..."
```

---

## Добавление новых источников

1. Положите PDF/DOCX/PPTX в `books/<Автор>/` (имя подпапки станет значением поля `author`)
2. `python scripts/1_ingest.py` — обработает только новый файл
3. `python scripts/2_build_index.py` — добавит новые чанки в индекс
4. Добавьте автора в секцию `authors` в `config.yaml` (флаг `primary: true/false`)

---

## Конфигурация авторов

В `config.yaml`:
```yaml
authors:
  - name: Гайворонский
    primary: true
  - name: Сапин
    primary: true
  - name: Якимов
    primary: false   # дополнительный источник
```

`primary: false` означает, что автор участвует в сравнении, но помечается в ответе как дополнительный источник.

---

## Технический стек

| Компонент | Библиотека |
|---|---|
| OCR + парсинг документов | [Docling](https://github.com/docling-project/docling) |
| Векторная БД | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (поддерживает русский) |
| Семантический поиск подвопросов | BAAI/bge-m3 + cosine similarity |
| LLM (локально) | [Ollama](https://ollama.com/) |
| LLM (облако) | OpenAI API |
| Выходной документ | python-docx |
