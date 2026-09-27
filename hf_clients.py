"""
hf_clients.py
=====================================================
نداءات Hugging Face Inference API

Models:
- BAAI/bge-m3              -> Embeddings
- BAAI/bge-reranker-v2-m3  -> Reranking

Requires:
HF_TOKEN environment variable
=====================================================
"""


import os
import time
import requests



# =====================================================
# HF Token
# =====================================================


HF_TOKEN = os.environ.get(
    "HF_TOKEN",
    ""
)


if not HF_TOKEN:

    raise ValueError(
        "❌ لم يتم العثور على HF_TOKEN في متغيرات البيئة"
    )



HEADERS = {

    "Authorization": f"Bearer {HF_TOKEN}",

    "Content-Type": "application/json"

}





# =====================================================
# Models URLs
# =====================================================


EMBED_MODEL_URL = (

    "https://router.huggingface.co/"
    "hf-inference/models/BAAI/bge-m3/"
    "pipeline/feature-extraction"

)



RERANK_MODEL_URL = (

    "https://router.huggingface.co/"
    "hf-inference/models/"
    "BAAI/bge-reranker-v2-m3"

)







# =====================================================
# HF Request Helper
# =====================================================


def _post_with_retry(
        url,
        payload,
        max_attempts=3
):


    last_error = None



    for attempt in range(max_attempts):


        try:


            response = requests.post(

                url,

                headers=HEADERS,

                json=payload,

                timeout=60

            )



            if response.status_code == 200:

                return response.json()



            elif response.status_code == 503:


                try:

                    wait = response.json().get(
                        "estimated_time",
                        5
                    )

                except:

                    wait = 5



                time.sleep(
                    min(wait,15)
                )


                continue




            elif response.status_code == 401:


                raise RuntimeError(
                    "❌ HF Token غير صالح"
                )



            elif response.status_code == 403:


                raise RuntimeError(
                    "❌ ليس لديك صلاحية استخدام هذا الموديل"
                )



            else:


                last_error = (

                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"

                )



        except Exception as e:


            last_error = str(e)



        time.sleep(2)



    raise RuntimeError(

        f"فشل HF API بعد {max_attempts} محاولات: {last_error}"

    )








# =====================================================
# Embeddings
# =====================================================


def get_embedding(text: str) -> list:

    """
    Generate embedding using BAAI/bge-m3
    """


    result = _post_with_retry(

        EMBED_MODEL_URL,

        {

            "inputs": text,

            "options": {

                "wait_for_model": True

            }

        }

    )



    # Normalize response format

    if (

        isinstance(result,list)

        and len(result)>0

        and isinstance(result[0],list)

        and isinstance(result[0][0],list)

    ):

        result = result[0]



    return result







# =====================================================
# Reranker
# =====================================================


def rerank_pair(
        query: str,
        passage: str
) -> float:


    """
    Calculate relevance score between:
    query and document passage

    Model:
    BAAI/bge-reranker-v2-m3
    """



    try:



        payload = {


            "inputs": {


                "source_sentence": query,


                "sentences": [

                    passage

                ]

            },


            "options": {


                "wait_for_model": True

            }


        }




        result = _post_with_retry(

            RERANK_MODEL_URL,

            payload

        )





        # Expected:
        # [
        #   {
        #     "score":0.92
        #   }
        # ]



        if isinstance(result,list):


            if len(result)>0:


                first = result[0]


                if isinstance(first,dict):


                    if "score" in first:


                        return float(

                            first["score"]

                        )





        # Alternative response

        if isinstance(result,dict):


            if "scores" in result:


                return float(

                    result["scores"][0]

                )





        return None





    except Exception:


        return None