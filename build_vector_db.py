"""
الخطوة 2: تحميل الداتا سيت + توليد الـ embeddings + بناء قاعدة Chroma
شغلي هذا الملف مرة وحدة فقط -- بيبني مجلد "chroma_db" يحفظ فيه كل شي.

ملاحظة: نستخدم نفس موديل BAAI/bge-m3 بالضبط، بس عبر HF Inference API
بدل تحميله محلياً (شوفي hf_clients.py) -- لازم يكون موجود HF_TOKEN
بمتغيرات البيئة قبل التشغيل.
"""

import json
import time
import chromadb
from hf_clients import get_embedding

# 1) تحميل الداتا سيت
with open("saudi_labor_law_dataset_full.json", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"تم تحميل {len(dataset)} مادة من الداتا سيت")

texts_for_embedding = [item["text"] for item in dataset]

print("جاري توليد الـ embeddings عبر HF Inference API... (تاخذ شوي وقت أول مرة)")
embeddings = []
for i, text in enumerate(texts_for_embedding):
    emb = get_embedding(text)
    embeddings.append(emb)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(texts_for_embedding)}")
    time.sleep(0.2)  # احترام لحدود الـ rate limit المجاني

# 2) إنشاء قاعدة Chroma وتخزين كل شي
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="saudi_labor_law")

ids = [f"art_{i}" for i in range(len(dataset))]
documents = [item["text"] for item in dataset]
metadatas = [
    {
        "bab": item["bab"],
        "bab_title": item["bab_title"],
        "article_no": item["article_no"],
    }
    for item in dataset
]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)

print(f"✅ تم حفظ {collection.count()} مادة في قاعدة Chroma بمجلد chroma_db/")
print("الخطوة الجاية: جربي سكربت test_search.py للتأكد إن البحث شغال")
