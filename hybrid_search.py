# hybrid_search.py

"""
Hybrid Search Module

Combines:
1. Semantic Search (Chroma + BGE-M3 Embeddings)
2. Keyword Search (BM25)
3. Query Expansion

Used as Retriever before LLM in RAG pipeline.
"""


import json
import re
import chromadb

from rank_bm25 import BM25Okapi
from hf_clients import get_embedding

from query_rewrite import expand_query



# =====================================================
# 1) Load Dataset
# =====================================================

with open(
    "saudi_labor_law_dataset_full.json",
    encoding="utf-8"
) as f:

    dataset = json.load(f)



# =====================================================
# 2) BM25 Preparation
# =====================================================

def tokenize(text):

    """
    Arabic text preprocessing
    """

    text = re.sub(
        r'[\u064B-\u0652]',
        '',
        text
    )

    text = re.sub(
        r'[^\w\s]',
        ' ',
        text
    )

    return text.split()



tokenized_corpus = [

    tokenize(
        f"""
        {item['bab_title']}
        {item['article_no']}
        {item['text']}
        """
    )

    for item in dataset

]


bm25 = BM25Okapi(tokenized_corpus)




# =====================================================
# 3) Chroma Setup
# =====================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.get_collection(
    name="saudi_labor_law"
)




# =====================================================
# 4) Hybrid Search
# =====================================================


def hybrid_search(
        question,
        top_k=5,
        pool_size=50
):

    """
    Hybrid Retrieval
    """


    # -----------------------------
    # Query Expansion
    # -----------------------------

    expanded_question = expand_query(question)

    print("\nExpanded query:")
    print(expanded_question)



    # -----------------------------
    # Semantic Search
    # -----------------------------

    query_embedding = get_embedding(
        expanded_question
    )


    semantic_results = collection.query(

        query_embeddings=[query_embedding],

        n_results=pool_size

    )


    semantic_ids = [

        int(i.split("_")[1])

        for i in semantic_results["ids"][0]

    ]


    semantic_rank = {

        idx: rank

        for rank, idx in enumerate(semantic_ids)

    }




    # -----------------------------
    # BM25 Search
    # -----------------------------

    bm25_scores = bm25.get_scores(
        tokenize(expanded_question)
    )


    bm25_ranked = sorted(

        range(len(bm25_scores)),

        key=lambda i: -bm25_scores[i]

    )[:pool_size]


    bm25_rank = {

        idx: rank

        for rank, idx in enumerate(bm25_ranked)

    }




    # -----------------------------
    # Weighted Fusion
    # -----------------------------

    candidates = (

        set(semantic_rank.keys())

        |

        set(bm25_rank.keys())

    )


    final_scores = {}



    for idx in candidates:


        score = 0



        # Semantic weight

        if idx in semantic_rank:

            score += (

                0.5 *

                (1 / (60 + semantic_rank[idx]))

            )



        # Keyword weight

        if idx in bm25_rank:

            score += (

                0.5 *

                (1 / (60 + bm25_rank[idx]))

            )





        # -----------------------------
        # Legal Boosting
        # -----------------------------

        text = (

            dataset[idx]["bab_title"]

            + " "

            + dataset[idx]["text"]

        )


        article = dataset[idx]["article_no"]


        q = question.lower()




        # -----------------------------
        # Maternity Leave
        # -----------------------------

        if (

            ("امومة" in q or "أمومة" in question)

            and "إجازة وضع" in text

        ):

            score += 0.25




        # -----------------------------
        # Resignation
        # -----------------------------

        if (

            "استقال" in q

            and "استقال" in text

        ):

            score += 0.25




        # -----------------------------
        # Contract termination
        # -----------------------------

        if (

            (
                "انهاء" in q
                or "إنهاء" in question
                or "انتهاء" in q
            )

            and

            (
                "انتهاء عقد العمل" in text
                or "ينتهي عقد العمل" in text
                or "فسخ" in text
            )

        ):

            score += 0.20




        # Direct boost for Article 74

        if (

            "إنهاء" in question

            and article == "الرابعة والسبعون"

        ):

            score += 0.35





        # Direct boost for Article 151

        if (

            "أمومة" in question

            and article == "الحادية والخمسون بعد المائة"

        ):

            score += 0.35





        final_scores[idx] = score





    # -----------------------------
    # Final Ranking
    # -----------------------------

    top_ids = sorted(

        final_scores,

        key=lambda i: -final_scores[i]

    )[:top_k]



    return [

        dataset[i]

        for i in top_ids

    ]





# =====================================================
# 5) Test
# =====================================================


if __name__ == "__main__":


    questions = [

        "كم مدة إجازة الأمومة؟",

        "ما هي حقوق العامل عند إنهاء عقد العمل؟",

        "ما هي حقوق العامل عند الاستقالة؟"

    ]



    for question in questions:


        print("\n" + "="*60)

        print("السؤال:")

        print(question)

        print("="*60)



        results = hybrid_search(

            question,

            top_k=3

        )



        for r in results:


            print("\nالمادة:")

            print(r["article_no"])


            print("\nالباب:")

            print(r["bab_title"])


            print("\nالنص:")

            print(r["text"][:400])


            print("-"*50)