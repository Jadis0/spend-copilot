from pathlib import Path

from extract import extract, check_math, check_totals
from classify import classify
from rag import check_against_policy
from agent import run_agent


def process_receipt(image_path: Path):
    """
    الأنبوب الكامل: استخراج → تحقق حسابي → تصنيف → فحص السياسة → قرار الوكيل.
    """
    # الخطوة ١: الاستخراج
    receipt = extract(image_path)

    # الخطوة ٢: التحقق الحسابي
    warnings = []
    w1 = check_math(receipt)
    w2 = check_totals(receipt)
    if w1:
        warnings.append(w1)
    if w2:
        warnings.append(w2)

    # الخطوة ٣: التصنيف
    item_names = [item.description for item in receipt.line_items]
    category, source = classify(receipt.merchant, item_names)

    # الخطوة ٤: فحص السياسة (RAG)
    policy_match = check_against_policy(
        category=category.value,
        amount=receipt.total,
        merchant=receipt.merchant,
        item_descriptions=item_names,
    )

    # الخطوة ٥: قرار الوكيل النهائي
    agent_result = run_agent(
        amount=receipt.total,
        merchant=receipt.merchant,
        category=category.value,
        has_warnings=bool(warnings),
        policy_text=policy_match["policy_text"],
    )

    return receipt, category, source, warnings, policy_match, agent_result


if __name__ == "__main__":
    images = [p for p in Path("receipts").iterdir()
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

    if not images:
        raise SystemExit("لا توجد صور في receipts/")

    for image_path in images:
        print(f"\n{'=' * 50}")
        print(f"معالجة: {image_path.name}")
        print('=' * 50)

        receipt, category, source, warnings, policy_match, agent_result = process_receipt(image_path)

        print(f"التاجر:  {receipt.merchant}")
        print(f"المبلغ:  {receipt.total} {receipt.currency}")
        print(f"الفئة:   {category.value}  (المصدر: {source})")

        if warnings:
            print("\nتحذيرات حسابية:")
            for w in warnings:
                print(f"  {w}")

        print(f"\nبند السياسة: {policy_match['policy_heading']}")

        print(f"\n{'─' * 30}")
        print(f"القرار النهائي: {agent_result.decision.value}  (المصدر: {agent_result.source})")
        print(f"السبب: {agent_result.reason}")