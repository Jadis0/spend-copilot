from typing import Optional, List
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
client = OpenAI()


# ── الفئات المسموحة ───────────────────────────────────
# Enum = قائمة قيم محددة سلفاً، لا يمكن الخروج عنها.
# هذا يضمن أن أي تصنيف (قاعدة أو LLM) يعيد واحدة من هذي الخمس بالضبط، لا نصاً حراً.

class Category(str, Enum):
    FOOD = "طعام ومطاعم"
    TRANSPORT = "مواصلات"
    ENTERTAINMENT = "ضيافة عملاء"
    SOFTWARE = "اشتراكات وبرمجيات"
    OTHER = "أخرى"


# ── المرحلة ١: القاعدة (رخيصة، فورية، حتمية) ──────────
# كل مفتاح = كلمة نبحث عنها *داخل* اسم التاجر (بحث جزئي لا تطابق كامل).
# القيمة = الفئة التي يجب تصنيفها إليها.
# ملاحظة: القاعدة تعمل على الاسم فقط، ولا تحتاج البنود — لأنها حتمية بطبيعتها.

MERCHANT_RULES: dict[str, Category] = {
    "hotel": Category.FOOD,
    "restaurante": Category.FOOD,
    "restaurant": Category.FOOD,
    "cafe": Category.FOOD,
    "café": Category.FOOD,
    "مطعم": Category.FOOD,
    "starbucks": Category.FOOD,
    "مقهى": Category.FOOD,
    "uber": Category.TRANSPORT,
    "careem": Category.TRANSPORT,
    "كريم": Category.TRANSPORT,
    "aramco": Category.TRANSPORT,
    "بنزين": Category.TRANSPORT,
    "aws": Category.SOFTWARE,
    "figma": Category.SOFTWARE,
    "slack": Category.SOFTWARE,
}


def classify_by_rule(merchant: str) -> Optional[Category]:
    """
    يبحث عن أي كلمة مفتاحية من القاموس داخل اسم التاجر.
    يعيد الفئة لو وُجد تطابق، أو None لو لم يوجد أي تطابق.
    """
    merchant_lower = merchant.lower()

    for keyword, category in MERCHANT_RULES.items():
        if keyword.lower() in merchant_lower:
            return category

    return None  # لا يوجد تطابق، يحتاج LLM


# ── المرحلة ٢: LLM (فقط عند الغموض) ───────────────────

class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category = Field(description="الفئة الأنسب لهذا التاجر بناءً على اسمه وبنوده")
    reasoning: str = Field(description="جملة واحدة قصيرة توضح سبب الاختيار")


def classify_by_llm(
    merchant: str,
    item_descriptions: Optional[List[str]] = None,
) -> ClassificationResult:
    """
    يُستدعى فقط عندما تفشل القاعدة في إيجاد تطابق.
    item_descriptions اختياري: قائمة أسماء البنود، تعطي سياقاً أدق من الاسم وحده.
    """
    context = f"اسم التاجر: {merchant}"
    if item_descriptions:
        items_text = "، ".join(item_descriptions)
        context += f"\nالبنود المشتراة: {items_text}"

    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "صنّف هذا الإيصال إلى إحدى الفئات المحددة. "
                    "اسم التاجر وحده قد يكون غامضاً (مثل 'Green Field' قد يكون مطعماً أو نادياً رياضياً) — "
                    "استخدم البنود المشتراة، إن وُجدت، لتحديد نوع النشاط الفعلي بدقة أكبر."
                ),
            },
            {"role": "user", "content": context},
        ],
        response_format=ClassificationResult,
    )

    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"النموذج رفض التصنيف: {message.refusal}")

    return message.parsed


# ── الدالة الموحّدة (هذي التي يستخدمها بقية الأنبوب) ──

def classify(
    merchant: str,
    item_descriptions: Optional[List[str]] = None,
) -> tuple[Category, str]:
    """
    التصنيف الهجين: القاعدة أولاً (على الاسم فقط)، LLM فقط عند الحاجة (بسياق أغنى).
    يعيد (الفئة، مصدر القرار) — المصدر يفيدنا لاحقاً في التقييم والتدقيق.
    """
    rule_result = classify_by_rule(merchant)
    if rule_result is not None:
        return rule_result, "rule"

    llm_result = classify_by_llm(merchant, item_descriptions)
    return llm_result.category, "llm"


# ── التشغيل التجريبي ───────────────────────────────────

if __name__ == "__main__":
    test_merchants = [
        ("HOTEL RESTAURANTE EL CRUCE", None),
        ("Uber Trip", None),
        ("مكتبة جرير", None),
        ("Green Field", ["Coffee", "Lunch", "Coke"]),
    ]

    for merchant, items in test_merchants:
        category, source = classify(merchant, items)
        print(f"{merchant:35} → {category.value:20} (المصدر: {source})")