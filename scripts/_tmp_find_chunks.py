import json, pathlib

ROOT = pathlib.Path(__file__).parent.parent

GAYV_NEEDED = [
    'ОБЩИЕ ЧЕРТЫ СТРОЕНИЯ СВОБОДНЫХ ПОЗВОНКОВ',
    'ШЕЙНЫЕ ПОЗВОНКИ',
    'ГРУДНЫЕ ПОЗВОНКИ',
    'ПОЯСНИЧНЫЕ ПОЗВОНКИ',
    'КРЕСТЕЦ',
    'Соединения свободных позвонков',
    'РЕНТГЕНОАНАТОМИЯ ПОЗВОНОЧНОГО СТОЛБА',
]
SAPIN_NEEDED = [
    'ПОЗВОНКИ',
    'ШЕЙНЫЕ ПОЗВОНКИ',
    'ГРУДНЫЕ ПОЗВОНКИ',
    'ПОЯСНИЧНЫЕ ПОЗВОНКИ',
    'КРЕСТЕЦ',
    'СОЕДИНЕНИЯ ПОЗВОНКОВ',
    'ПОЗВОНОЧНЫЙ СТОЛБ',
    'ДВИЖЕНИЯ ПОЗВОНОЧНОГО СТОЛБА',
]

for author, fname, needed in [
    ('ГАЙВОРОНСКИЙ', 'gajvoronskij_i_v_normalnaya_anatomiya-tom-1_chunks.jsonl', GAYV_NEEDED),
    ('САПИН', 'sapin-tom-1_chunks.jsonl', SAPIN_NEEDED),
]:
    path = ROOT / 'data' / 'chunks' / fname
    with open(path, encoding='utf-8') as f:
        chunks = [json.loads(l) for l in f]
    print(f'\n\n######## {author} ########')
    for c in chunks:
        headings = c.get('headings', [])
        last_h = headings[-1].strip() if headings else ''
        for n in needed:
            if last_h.upper() == n.upper():
                full_h = ' > '.join(headings)
                print(f'\n--- {full_h} ---')
                print(c['text'][:5000])
                break
