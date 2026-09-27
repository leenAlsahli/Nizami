"""
الخطوة 4: اختبار البحث -- تتأكدين إن قاعدة Chroma ترجع المواد الصحيحة
شغلي هذا بعد build_vector_db.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

# BAAI/bge-m3 موديل قوي جداً بالفهم الدلالي متعدد اللغات، ويدعم العربي بشكل ممتاز
model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="saudi_labor_law")

# غيّري السؤال هنا وجربي أسئلة مختلفة
question = "كم مدة إجازة الأمومة؟"

# ملاحظة: bge-m3 لا يحتاج بادئة "query: " زي موديل e5
query_embedding = model.encode([question])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
)

print(f"السؤال: {question}\n")
print("أقرب 3 مواد:\n" + "-" * 40)

for i in range(len(results["documents"][0])):
    meta = results["metadatas"][0][i]
    doc = results["documents"][0][i]
    distance = results["distances"][0][i]
    print(f"\nالباب {meta['bab']} ({meta['bab_title']}) -- المادة {meta['article_no']}")
    print(f"درجة القرب (أقل = أفضل): {distance:.4f}")
    print(f"النص: {doc[:150]}...")
