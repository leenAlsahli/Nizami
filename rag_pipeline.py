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

    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"

]


MAX_ATTEMPTS_PER_MODEL = 2


BACKOFF_SECONDS = [

    1,
    2

]


RETRIEVAL_TOP_K = 10


FINAL_CONTEXT_K = 4



FALLBACK_UNAVAILABLE_MESSAGE = (
    "تعذر الاتصال بنموذج الذكاء الاصطناعي حالياً، الرجاء المحاولة بعد قليل."
)



NEED_CONTEXT_TOKEN = "NEED_CONTEXT"



# =====================================================
# Router Instruction
# =====================================================

ROUTER_SYSTEM_INSTRUCTION = f"""

أنت مصنف رسائل فقط.

حدد هل سؤال المستخدم متعلق بنظام العمل السعودي أم لا.

إذا كان السؤال متعلقاً بـ:
- حقوق العامل
- حقوق صاحب العمل
- العقود
- الأجور
- الإجازات
- الاستقالة
- الفصل
- ساعات العمل
- المكافآت
- مواد نظام العمل السعودي

اكتب فقط:

{NEED_CONTEXT_TOKEN}


إذا لم يكن متعلقاً بنظام العمل السعودي:

اكتب فقط:

NO_CONTEXT


لا تكتب أي شرح إضافي.

"""



# =====================================================
# Final Answer Instruction
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



ROUTER_CONFIG = types.GenerateContentConfig(

    temperature=0.0,

    max_output_tokens=20,

    system_instruction=ROUTER_SYSTEM_INSTRUCTION

)



GENERATION_CONFIG = types.GenerateContentConfig(

    temperature=0.0,

    max_output_tokens=800,

    top_p=0.9,

    system_instruction=SYSTEM_INSTRUCTION

)



# =====================================================
# Gemini Caller
# =====================================================

def _call_gemini(contents, config):


    for model in MODEL_FALLBACK_CHAIN:


        for attempt in range(MAX_ATTEMPTS_PER_MODEL):


            try:

                response = client.models.generate_content(

                    model=model,

                    contents=contents,

                    config=config

                )


                if response and response.text:

                    return response.text.strip()


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


    return None
# =====================================================
# General Chat Reply
# =====================================================

def generate_chat_reply(question):


    prompt = f"""

أنت "نظامي"، مساعد قانوني متخصص في نظام العمل السعودي.

أنت لست مساعد دردشة عام.

إذا كانت رسالة المستخدم:
- تحية
- كلام عام
- سؤال غير متعلق بنظام العمل السعودي

فلا تدخل في دردشة طويلة ولا تقترح أفلام أو ألعاب أو مواضيع ترفيهية.

اجعل الرد:
- مختصر
- مهني
- يعيد المستخدم لمجال النظام.

أمثلة:

المستخدم:
طفشت

الرد:
كيف يمكنني مساعدتك؟ يمكنني الإجابة عن استفساراتك المتعلقة بنظام العمل السعودي مثل الحقوق، العقود، الإجازات، والأجور.


المستخدم:
هلا

الرد:
أهلاً بك، كيف يمكنني مساعدتك في استفسارات نظام العمل السعودي؟


المستخدم:
وش اسمك؟

الرد:
أنا نظامي، مساعد قانوني متخصص في نظام العمل السعودي.


رسالة المستخدم:

{question}

"""


    response = _call_gemini(

        prompt,

        types.GenerateContentConfig(

            temperature=0.2,

            max_output_tokens=150

        )

    )


    if response:

        return response


    return (
        "أهلاً بك، كيف يمكنني مساعدتك في استفسارات نظام العمل السعودي؟"
    )





# =====================================================
# Local Context Filter
# =====================================================

def filter_context_results(question, results):


    if not results:

        return []


    return results[:FINAL_CONTEXT_K]





# =====================================================
# Main RAG Pipeline
# =====================================================

def ask_rag(question):


    # ==============================
    # المرحلة الأولى: Router
    # ==============================

    router_reply = _call_gemini(

        question,

        ROUTER_CONFIG

    )



    if router_reply is None:


        return (

            FALLBACK_UNAVAILABLE_MESSAGE,

            [],

            []

        )



    # ==============================
    # Chat Mode
    # ==============================

    if router_reply.strip().upper() != NEED_CONTEXT_TOKEN:


        return (

            generate_chat_reply(question),

            [],

            []

        )



    # ==============================
    # RAG Mode
    # ==============================


    results = hybrid_search(

        question,

        top_k=RETRIEVAL_TOP_K

    )



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



    answer = _call_gemini(

        prompt,

        GENERATION_CONFIG

    )



    if answer is None:


        answer = FALLBACK_UNAVAILABLE_MESSAGE



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

        "هلا",

        "طفشت",

        "وش اسمك",

        "كم مدة الإجازة السنوية للعامل؟"

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
