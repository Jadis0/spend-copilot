import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from pipeline import process_receipt

app = FastAPI(title="spend-copilot")


class ProcessReceiptResponse(BaseModel):
    merchant: str
    amount: float
    currency: str
    category: str
    category_source: str
    warnings: list[str]
    decision: str
    decision_reason: str
    decision_source: str


@app.post("/process-receipt", response_model=ProcessReceiptResponse)
async def process_receipt_endpoint(file: UploadFile = File(...)):
    """
    يستقبل صورة إيصال، ويعيد القرار الكامل بعد المعالجة.
    """
    # نحفظ الملف المرفوع مؤقتاً على القرص، لأن process_receipt تتوقع Path
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        receipt, category, source, warnings, policy_match, agent_result = process_receipt(tmp_path)
    finally:
        tmp_path.unlink()  # نحذف الملف المؤقت بعد الاستخدام، حتى لو حدث خطأ

    return ProcessReceiptResponse(
        merchant=receipt.merchant,
        amount=receipt.total,
        currency=receipt.currency,
        category=category.value,
        category_source=source,
        warnings=warnings,
        decision=agent_result.decision.value,
        decision_reason=agent_result.reason,
        decision_source=agent_result.source,
    )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "spend-copilot"}