import os
import time
import warnings

from google import genai
from google.genai import types

from hybrid_search import hybrid_search


warnings.filterwarnings("ignore")


# =====================================================
# Gemini Client Setup
# =====================================================

_raw_key = os.environ.get(
    "GEMINI_API_KEY",
    ""
)


if not _raw_key:
    raise ValueError(
        "❌ لم يتم العثور على GEMINI_API_KEY"
    )


_clean_key = (
    _raw_key
    .strip()
    .splitlines()[0]
    .strip()
)


client = genai.Client(
    api_key=_clean_key
)


print("Gemini connected ✅")



# =====================================================
# Configuration
# =====================================================


MODEL_FALLBACK_CHAIN = [

    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash"

]


MAX_ATTEMPTS_PER_MODEL = 2


BACKOFF_SECONDS = [

    1,
    2

]


RETRIEVAL_TOP_K = 10


FINAL_CONTEXT_K = 4



# =====================================================
# System Instruction
# =====================================================


SYSTEM_INSTRUCTION = """

أنت "نظامي"، مساعد قانوني ذكي متخصص في نظام العمل السعودي.

قواعد الإجابة:

1. أجب فقط اعتماداً على النصوص الموجودة في السياق المرفق.

2. لا تخترع أو تفترض معلومات غير موجودة.

3. إذا لم تجد الإجابة في السياق قل:
"لم أجد نصاً متعلقاً بهذا السؤال في البيانات المتاحة."

4. اذكر رقم المادة واسم الباب عند توفرها.

5. ابدأ بالإجابة المباشرة على سؤال المستخدم.

6. استخدم فقط المواد الأكثر ارتباطاً بالسؤال.

7. إذا احتوى السياق على أكثر من مادة مرتبطة، اجمعها في إجابة واحدة مرتبة.

8. لا تكرر نصوص المواد كاملة إلا عند الحاجة.

9. اجعل الإجابة واضحة ومباشرة ودقيقة قانونياً.

"""


GENERATION_CONFIG = types.GenerateContentConfig(

    temperature=0.0,

    max_output_tokens=800,

    top_p=0.9,

    system_instruction=SYSTEM_INSTRUCTION

)



# =====================================================
# Normalize Text
# =====================================================


def normalize_text(text):

    text = text.strip().lower()


    replacements = {

        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي"

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    return text



# =====================================================
# Small Talk Detection
# =====================================================


def is_small_talk(question):

    clean_q = normalize_text(question)


    greetings = {

        "اهلين",
        "اهلا",
        "مرحبا",
        "هلا",
        "هاي",
        "hi",
        "hello",
        "السلام عليكم",
        "كيفك",
        "كيف حالك",
        "شخبارك",
        "علومك",
        "من انت",
        "من تكون",
        "وش تسوي"

    }



    if clean_q in greetings:

        return True



    if any(

        clean_q.startswith(x)

        for x in [

            "اهلين",
            "اهلا",
            "مرحبا",
            "هلا"

        ]

    ):

        return True



    return False



# =====================================================
# Direct Chat Generator
# =====================================================


def generate_direct_chat(question):


    prompt = f"""

أنت "نظامي"، مساعد قانوني متخصص في نظام العمل السعودي.

أجب باختصار وبأسلوب مهني.

السؤال:
{question}

"""


    for model in MODEL_FALLBACK_CHAIN:

        try:

            response = client.models.generate_content(

                model=model,

                contents=prompt

            )


            if response and response.text:

                return response.text


        except Exception as e:

            print(
                f"⚠️ Gemini error [{model}] direct chat: {e}"
            )

            continue



    return (
        "أهلاً بك. أنا نظامي، "
        "مساعد قانوني متخصص في نظام العمل السعودي."
    )
    # =====================================================
# Local Context Filter
# =====================================================


def filter_context_results(question, results):

    if not results:

        return []


    # نعتمد على ترتيب Hybrid Search

    return results[:FINAL_CONTEXT_K]





# =====================================================
# Gemini Answer Generator
# =====================================================


def generate_answer(prompt):


    for model in MODEL_FALLBACK_CHAIN:


        for attempt in range(MAX_ATTEMPTS_PER_MODEL):

            try:

                response = client.models.generate_content(

                    model=model,

                    contents=prompt,

                    config=GENERATION_CONFIG

                )


                if response and response.text:

                    return response.text



            except Exception as e:


                print(
                    f"⚠️ Gemini error [{model}] attempt {attempt+1}: {e}"
                )


                if "429" in str(e):

                    time.sleep(3)

                else:

                    time.sleep(
                        BACKOFF_SECONDS[attempt]
                    )



    return (
        "تعذر الاتصال بنموذج الذكاء الاصطناعي حالياً."
    )





# =====================================================
# Main RAG Pipeline
# =====================================================


def ask_rag(question):


    # Small Talk

    if is_small_talk(question):


        return (

            generate_direct_chat(question),

            [],

            []

        )




    # Hybrid Search

    results = hybrid_search(

        question,

        top_k=RETRIEVAL_TOP_K

    )




    # Filter results

    results = filter_context_results(

        question,

        results

    )




    if not results:


        return (

            "أنا نظامي، مساعد قانوني متخصص في نظام العمل السعودي.\n"
            "لم أجد نصاً متعلقاً بهذا السؤال في البيانات المتاحة.",

            [],

            []

        )





    context = ""

    sources = []

    contexts = []




    for r in results:


        bab = r.get(

            "bab_title",

            ""

        )


        article = r.get(

            "article_no",

            ""

        )


        text = r.get(

            "text",

            ""

        )




        context += f"""

الباب:
{bab}


المادة:
{article}


النص:
{text}

---------------------

"""



        if len(sources) < 3:


            sources.append(

                f"المادة {article} - {bab}"

            )


        contexts.append(text)






    prompt = f"""

السياق المتاح:

{context}


سؤال المستخدم:

{question}


أجب اعتماداً على السياق فقط.

"""




    answer = generate_answer(

        prompt

    )




    return (

        answer,

        sources,

        contexts

    )






# =====================================================
# Test
# =====================================================


if __name__ == "__main__":


    questions = [

        "اهلين",

        "أهلاً كيف حالك؟",

        "ما حقوقي إذا استقلت من العمل؟"

    ]



    for q in questions:


        print("\n" + "="*60)


        print("❓ السؤال:")

        print(q)



        ans, src, ctx = ask_rag(q)



        print("\n💡 الإجابة:")

        print(ans)



        if src:

            print("\n📌 المصادر:")

            for s in src:

                print("-", s)



        print("="*60)
