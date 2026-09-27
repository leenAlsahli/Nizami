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

# ملاحظة مهمة: نماذج Gemini 2.5 سيتوقف العمل بها في 16 أكتوبر 2026.
# لو رجعتِ لهذا الملف بعد هذا التاريخ لازم تشيلين gemini-2.5-* من القائمة
# وتحطين بدلها إصدار أحدث مستقر وقتها (تأكدي من توثيق Gemini الرسمي).
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


# الرد الذي يظهر للمستخدم فقط إذا فشلت كل النماذج في السلسلة بالكامل
FALLBACK_UNAVAILABLE_MESSAGE = (
    "تعذر الاتصال بنموذج الذكاء الاصطناعي حالياً، الرجاء المحاولة بعد قليل."
)


# العلامة الداخلية التي يرد بها Gemini عندما يقرر أن السؤال يحتاج
# بحثاً في نظام العمل السعودي (لا تظهر أبداً للمستخدم)
NEED_CONTEXT_TOKEN = "NEED_CONTEXT"



# =====================================================
# System Instructions
# =====================================================

# التعليمة الخاصة بمرحلة "التوجيه": يقرر Gemini هل السؤال متعلق
# بنظام العمل السعودي (فيحتاج سياق/استرجاع) أو دردشة عامة (فيرد مباشرة)
ROUTER_SYSTEM_INSTRUCTION = f"""

أنت "نظامي"، مساعد قانوني ذكي متخصص في نظام العمل السعودي.

في كل رسالة تصلك، حدد أولاً نوع الرسالة:

1) إذا كانت الرسالة سؤالاً يتعلق فعلياً بنظام العمل السعودي
(حقوق وواجبات العامل أو صاحب العمل، عقود العمل، الأجور، الإجازات،
الإنهاء أو الاستقالة، ساعات العمل، المكافآت، إلخ):
رد فقط بالكلمة التالية بالضبط وبدون أي إضافة أو علامات ترقيم:
{NEED_CONTEXT_TOKEN}

2) إذا كانت الرسالة تحية أو دردشة عامة أو سؤالاً عن هويتك أو أي شيء
غير متعلق بنظام العمل السعودي:
رد مباشرة برسالة قصيرة وودودة ومهنية بصفتك "نظامي"، مساعد قانوني
متخصص في نظام العمل السعودي. لا تكتب كلمة {NEED_CONTEXT_TOKEN} أبداً
في هذه الحالة.

"""


# التعليمة الخاصة بمرحلة "الإجابة النهائية" بعد استرجاع السياق القانوني
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


ROUTER_GENERATION_CONFIG = types.GenerateContentConfig(

    temperature=0.0,

    max_output_tokens=300,

    system_instruction=ROUTER_SYSTEM_INSTRUCTION

)


GENERATION_CONFIG = types.GenerateContentConfig(

    temperature=0.0,

    max_output_tokens=800,

    top_p=0.9,

    system_instruction=SYSTEM_INSTRUCTION

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
# Generic Gemini Caller (with fallback chain + retries)
# =====================================================


def _call_gemini(contents, config):
    """
    يحاول الاتصال بكل نموذج في MODEL_FALLBACK_CHAIN بالترتيب،
    مع إعادة محاولة لكل نموذج. يطبع سبب الفشل الحقيقي في اللوق
    (Render logs) فقط، ولا يُرجعه أبداً للمستخدم.

    يرجع نص الرد أو None إذا فشلت كل المحاولات.
    """

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

                # هذا يطبع فقط في لوق السيرفر (Render) للتشخيص الداخلي،
                # ولا يصل أبداً للمستخدم أو للفرونت إند
                print(
                    f"⚠️ Gemini error [{model}] attempt {attempt + 1}: {e}"
                )

                if "429" in str(e):

                    time.sleep(3)

                else:

                    time.sleep(
                        BACKOFF_SECONDS[attempt]
                    )

    return None



# =====================================================
# Main RAG Pipeline
# =====================================================


def ask_rag(question):
    """
    كل رسالة تُرسل أولاً إلى Gemini ليقرر بنفسه:
    - إذا كانت متعلقة بنظام العمل السعودي: نسوي Hybrid Search
      ثم نرجع للـ Gemini مرة ثانية بالسياق القانوني لصياغة الإجابة.
    - إذا كانت دردشة عامة/تحية/سؤال عن الهوية: يرد Gemini مباشرة
      بنفس الاستدعاء الأول بدون أي بحث.
    """

    # المرحلة 1: التوجيه -- دائماً عبر Gemini، بدون أي تصنيف محلي ثابت
    router_reply = _call_gemini(
        contents=question,
        config=ROUTER_GENERATION_CONFIG
    )

    if router_reply is None:

        # فشلت كل النماذج حتى بمرحلة التوجيه
        return (
            FALLBACK_UNAVAILABLE_MESSAGE,
            [],
            []
        )

    if router_reply.strip().upper() != NEED_CONTEXT_TOKEN:

        # دردشة عامة / تحية / سؤال عن الهوية -- رد Gemini مباشرة
        return (
            router_reply,
            [],
            []
        )

    # المرحلة 2: السؤال متعلق بنظام العمل السعودي -- نسوي Hybrid Search

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
        contents=prompt,
        config=GENERATION_CONFIG
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

        "اهلين",

        "أهلاً كيف حالك؟",

        "ما حقوقي إذا استقلت من العمل؟"

    ]

    for q in questions:

        print("\n" + "=" * 60)

        print("❓ السؤال:")
        print(q)

        ans, src, ctx = ask_rag(q)

        print("\n💡 الإجابة:")
        print(ans)

        if src:

            print("\n📌 المصادر:")

            for s in src:

                print("-", s)

        print("=" * 60)
