import base64
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
client = OpenAI()  # يقرأ OPENAI_API_KEY من البيئة تلقائياً


# ── العقد ─────────────────────────────────────────────
# هذا هو مصدر الحقيقة. المخطط يُشتق منه، لا يُكتب يدوياً.

class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")   # → additionalProperties: false

    description: str = Field(description="اسم الصنف كما هو مكتوب في الإيصال")
    quantity: Optional[float] = Field(description="الكمية، أو null إن لم تُذكر")
    unit_price: Optional[float] = Field(description="سعر الوحدة، أو null")
    amount: float = Field(description="المبلغ الإجمالي لهذا السطر")


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant: str = Field(description="اسم التاجر أو المتجر")
    date: str = Field(description="تاريخ الإيصال بصيغة YYYY-MM-DD")
    currency: str = Field(description="رمز العملة ISO 4217، مثل SAR أو USD أو EUR")
    subtotal: Optional[float] = Field(description="المجموع قبل الضريبة، أو null")
    tax: Optional[float] = Field(description="مبلغ ضريبة القيمة المضافة، أو null")
    total: float = Field(description="المبلغ النهائي المدفوع")
    line_items: List[LineItem] = Field(description="بنود الإيصال")


# ── الأدوات ───────────────────────────────────────────

def to_data_url(path: Path) -> str:
    """يقرأ الصورة كـ bytes ← يرمّزها base64 ← يغلّفها في data URL."""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{encoded}"


PROMPT = """أنت مستخرج بيانات إيصالات.
استخرج الحقول من الصورة كما هي مكتوبة تماماً.
- الإيصال قد يكون بأي لغة (عربي، إسباني، إنجليزي...) — استخرج القيم كما هي بغض النظر عن اللغة.
- إن لم يُذكر حقل صراحةً، أعد null. لا تخمّن، لا تحسب.
- التاريخ بصيغة YYYY-MM-DD."""


def extract(image_path: Path) -> Receipt:
    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "استخرج بيانات هذا الإيصال."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": to_data_url(image_path),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format=Receipt,   # ← Pydantic ← JSON Schema ← strict:true
    )

    message = response.choices[0].message

    # حالة الفشل الأولى. تُفحص قبل أي شيء.
    if message.refusal:
        raise RuntimeError(f"النموذج رفض: {message.refusal}")

    return message.parsed


def check_math(receipt: Receipt) -> Optional[str]:
    """
    يتحقق حسابياً: هل مجموع بنود الإيصال يطابق subtotal المعلن؟
    يعيد None لو كل شيء متطابق، أو رسالة تحذير لو فيه تعارض.
    """
    if receipt.subtotal is None:
        return None  # لا يوجد subtotal لنقارن به، تجاوز الفحص

    items_sum = sum(item.amount for item in receipt.line_items)
    difference = abs(items_sum - receipt.subtotal)

    TOLERANCE = 0.05  # هامش خطأ مسموح به (فلسات/سنتات بسبب تقريب الكمبيوتر)

    if difference > TOLERANCE:
        return (
            f"⚠️ تعارض حسابي: مجموع البنود = {items_sum:.2f}, "
            f"لكن subtotal المعلن = {receipt.subtotal:.2f} "
            f"(فرق {difference:.2f})"
        )
    return None


def check_totals(receipt: Receipt) -> Optional[str]:
    """
    يتحقق حسابياً: هل subtotal + tax يساوي total؟
    يعيد None لو متطابق، أو رسالة تحذير لو فيه تعارض.
    """
    if receipt.subtotal is None or receipt.tax is None:
        return None  # نقص بيانات، تجاوز الفحص

    expected_total = receipt.subtotal + receipt.tax
    difference = abs(expected_total - receipt.total)

    TOLERANCE = 0.05

    if difference > TOLERANCE:
        return (
            f"⚠️ تعارض في الإجمالي: subtotal ({receipt.subtotal:.2f}) "
            f"+ tax ({receipt.tax:.2f}) = {expected_total:.2f}, "
            f"لكن total المعلن = {receipt.total:.2f} "
            f"(فرق {difference:.2f})"
        )
    return None


# ── التشغيل ───────────────────────────────────────────

if __name__ == "__main__":
    images = [p for p in Path("receipts").iterdir()
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    if not images:
        raise SystemExit("لا توجد صور في receipts/")

    receipt = extract(images[0])
    print(receipt.model_dump_json(indent=2))

    warnings = []
    w1 = check_math(receipt)
    w2 = check_totals(receipt)
    if w1:
        warnings.append(w1)
    if w2:
        warnings.append(w2)

    if warnings:
        print()
        for w in warnings:
            print(w)
    else:
        print("\n✅ كل الحسابات متطابقة")