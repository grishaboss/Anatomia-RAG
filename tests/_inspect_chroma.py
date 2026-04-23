import sys
sys.path.insert(0, "scripts")
from utils import load_config
import chromadb

cfg = load_config()
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("anatomy")

res = col.get(limit=5, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(res["documents"], res["metadatas"])):
    print(f"--- chunk {i} ---")
    print(f"  embed_text (document): {repr(doc[:120])}")
    print(f"  meta keys: {list(meta.keys())}")
    heading = meta.get("headings", "")
    print(f"  heading: {heading[:60]}")
