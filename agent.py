from typing import Optional
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
client = OpenAI()


# ── الخطوة ١: استخراج الحد الرقمي من نص السياسة ───────

class PolicyLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_limit: bool = Field(description="هل يذكر النص حداً رقمياً صريحاً للمبلغ؟")
    limit_amount: Optional[float] = Field(
        description="الحد الأقصى بالريال إن وُجد، أو null إن لم يُذكر رقم"
    )
    is_prohibited: bool = Field(
        description="هل النص يمنع هذا النوع من المصروف كلياً بغض النظر عن المبلغ؟"
    )


def extract_limit_from_policy(policy_text: str) -> PolicyLimit:
    """
    يقرأ نص بند سياسة، ويستخرج منه: هل فيه حد رقمي؟ كم؟ هل هو منع كامل؟
    """
    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "اقرأ نص بند السياسة التالي، واستخرج منه: "
                    "هل يذكر حداً رقمياً بالريال؟ كم قيمته إن وُجد؟ "
                    "وهل النص يمنع هذا النوع من المصروف كلياً (بغض النظر عن المبلغ)؟"
                ),
            },
            {"role": "user", "content": policy_text},
        ],
        response_format=PolicyLimit,
    )

    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"النموذج رفض: {message.refusal}")

    return message.parsed


# ── الخطوة ٢: القرار النهائي (هجين: قواعد صريحة، ثم LLM للغامض) ──

class Decision(str, Enum):
    APPROVED = "موافق"
    REJECTED = "مرفوض"
    NEEDS_REVIEW = "مراجعة"


class AgentResult(BaseModel):
    decision: Decision
    reason: str
    source: str  # "rule" أو "llm" — للتدقيق لاحقاً


def decide_by_rule(
    amount: float,
    has_warnings: bool,
    policy_limit: PolicyLimit,
) -> Optional[AgentResult]:
    """
    الطبقة ١: قواعد حتمية للحالات الواضحة فقط.
    يعيد AgentResult لو الحالة واضحة، أو None لو تحتاج LLM.
    """
    if has_warnings:
        return AgentResult(
            decision=Decision.REJECTED,
            reason="يوجد تعارض حسابي في بيانات الإيصال — البيانات غير موثوقة",
            source="rule",
        )

    if policy_limit.is_prohibited:
        return AgentResult(
            decision=Decision.REJECTED,
            reason="هذا النوع من المصروف ممنوع كلياً حسب سياسة الشركة",
            source="rule",
        )

    if policy_limit.has_limit and amount > policy_limit.limit_amount * 1.5:
        return AgentResult(
            decision=Decision.REJECTED,
            reason=(
                f"المبلغ ({amount:.2f}) يتجاوز حد السياسة "
                f"({policy_limit.limit_amount:.2f}) بشكل كبير وواضح"
            ),
            source="rule",
        )

    if policy_limit.has_limit and amount <= policy_limit.limit_amount:
        return AgentResult(
            decision=Decision.APPROVED,
            reason=f"المبلغ ({amount:.2f}) ضمن حد السياسة ({policy_limit.limit_amount:.2f})",
            source="rule",
        )

    return None


class LLMDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reason: str = Field(description="جملة أو جملتان توضح سبب القرار")


def decide_by_llm(
    amount: float,
    merchant: str,
    category: str,
    policy_text: str,
    policy_limit: PolicyLimit,
) -> AgentResult:
    """
    الطبقة ٢: يُستدعى فقط للحالات الرمادية التي لم تحسمها القواعد.
    """
    context = f"""
مبلغ الإيصال: {amount:.2f} ريال
التاجر: {merchant}
الفئة: {category}
نص بند السياسة ذي الصلة: {policy_text}
الحد المستخرج من السياسة: {policy_limit.limit_amount if policy_limit.has_limit else "غير محدد رقمياً"}
"""

    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "أنت مدقق نفقات شركة. المبلغ قريب من حد السياسة أو الوضع غير حاسم بوضوح. "
                    "قرر: موافق (إن كان معقولاً بوضوح كافٍ)، مرفوض (إن كان يخالف السياسة بوضوح كافٍ)، "
                    "أو مراجعة (إن كان الوضع فعلاً غامضاً ويحتاج حكم إنسان). "
                    "كن متحفظاً — عند الشك الحقيقي، اختر مراجعة لا موافقة."
                ),
            },
            {"role": "user", "content": context},
        ],
        response_format=LLMDecision,
    )

    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"النموذج رفض: {message.refusal}")

    result = message.parsed
    return AgentResult(
        decision=result.decision,
        reason=result.reason,
        source="llm",
    )


# ── الدالة الموحّدة: الوكيل الكامل ─────────────────────

def run_agent(
    amount: float,
    merchant: str,
    category: str,
    has_warnings: bool,
    policy_text: str,
) -> AgentResult:
    """
    الوكيل الهجين الكامل: قواعد أولاً، LLM للحالات الرمادية فقط.
    """
    policy_limit = extract_limit_from_policy(policy_text)

    rule_result = decide_by_rule(amount, has_warnings, policy_limit)
    if rule_result is not None:
        return rule_result

    return decide_by_llm(amount, merchant, category, policy_text, policy_limit)


if __name__ == "__main__":
    policy_text = (
        "## ١. الوجبات والطعام | Meals & Food\n"
        "الحد الأقصى للوجبة الواحدة (فطور، غداء، عشاء) هو **٥٠ ريالاً سعودياً** للموظف الواحد."
    )

    # حالة رمادية حقيقية: 55 ريال، أعلى من الحد لكن ليس بشكل صارخ
    result = run_agent(
        amount=55.0,
        merchant="مطعم تجريبي",
        category="طعام ومطاعم",
        has_warnings=False,
        policy_text=policy_text,
    )

    print(f"القرار: {result.decision.value}")
    print(f"السبب: {result.reason}")
    print(f"المصدر: {result.source}")