import json
from pathlib import Path

from pipeline import process_receipt


def load_ground_truth() -> dict:
    with open("ground_truth.json", "r", encoding="utf-8") as f:
        return json.load(f)


def is_close(a: float, b: float, tolerance: float = 0.02) -> bool:
    """
    يقارن رقمين مع هامش خطأ صغير (لتفادي مشاكل التقريب العشري).
    """
    return abs(a - b) <= tolerance


def evaluate_receipt(image_name: str, truth: dict) -> dict:
    """
    يشغّل الأنبوب على إيصال واحد، ويقارن النتيجة بالحقيقة المرجعية.
    يعيد قاموساً يوضح نجاح/فشل كل بُعد على حدة.
    """
    image_path = Path("receipts") / image_name
    receipt, category, source, warnings, policy_match, agent_result = process_receipt(image_path)

    

    result = {
        "file": image_name,
        "extraction": {
            "merchant_correct": truth["merchant"].lower() in receipt.merchant.lower()
                                 or receipt.merchant.lower() in truth["merchant"].lower(),
            "total_correct": is_close(receipt.total, truth["total"]),
            "currency_correct": receipt.currency == truth["currency"],
        },
        "classification": {
            "category_correct": category.value == truth["expected_category"],
            "source": source,
        },
        "math_check": {
            "has_known_issue": "note" in truth and "تعارض" in truth.get("note", ""),
            "warnings_raised": len(warnings) > 0,
        },
        "agent_decision": {
            "expected": truth["expected_decision"],
            "actual": agent_result.decision.value,
            "correct": agent_result.decision.value == truth["expected_decision"],
            "source": agent_result.source,
            "reason": agent_result.reason,
        },
    }

    if result["math_check"]["has_known_issue"]:
        result["math_check"]["correct"] = result["math_check"]["warnings_raised"]
    else:
        result["math_check"]["correct"] = not result["math_check"]["warnings_raised"]

    return result


def print_summary(results: list[dict]):
    total = len(results)

    extraction_scores = {
        "merchant": sum(r["extraction"]["merchant_correct"] for r in results),
        "total": sum(r["extraction"]["total_correct"] for r in results),
        "currency": sum(r["extraction"]["currency_correct"] for r in results),
    }
    classification_score = sum(r["classification"]["category_correct"] for r in results)
    math_score = sum(r["math_check"]["correct"] for r in results)
    decision_score = sum(r["agent_decision"]["correct"] for r in results)

    print(f"\n{'=' * 50}")
    print(f"ملخّص التقييم — {total} إيصالاً")
    print('=' * 50)

    print(f"\nدقة الاستخراج:")
    print(f"  التاجر:    {extraction_scores['merchant']}/{total}")
    print(f"  المبلغ:    {extraction_scores['total']}/{total}")
    print(f"  العملة:    {extraction_scores['currency']}/{total}")

    print(f"\nدقة التصنيف: {classification_score}/{total}")
    print(f"دقة التحقق الحسابي: {math_score}/{total}")
    print(f"دقة قرار الوكيل: {decision_score}/{total}")

    print(f"\n{'─' * 30}")
    print("تفاصيل الحالات الفاشلة:\n")
    for r in results:
        failures = []
        if not r["extraction"]["merchant_correct"]:
            failures.append("التاجر خاطئ")
        if not r["extraction"]["total_correct"]:
            failures.append("المبلغ خاطئ")
        if not r["extraction"]["currency_correct"]:
            failures.append("العملة خاطئة")
        if not r["classification"]["category_correct"]:
            failures.append("التصنيف خاطئ")
        if not r["math_check"]["correct"]:
            failures.append("التحقق الحسابي فشل")
        if not r["agent_decision"]["correct"]:
            failures.append(
                f"القرار خاطئ (متوقع: {r['agent_decision']['expected']}, "
                f"فعلي: {r['agent_decision']['actual']})"
            )

        if failures:
            print(f"  {r['file']}: {', '.join(failures)}")


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    results = []

    for image_name, truth in ground_truth.items():
        print(f"معالجة: {image_name}...")
        result = evaluate_receipt(image_name, truth)
        results.append(result)

    print_summary(results)