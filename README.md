# spend-copilot

An agent that turns a messy expense receipt (photo) into a structured, categorized, policy-checked expense entry — built as a scaled-down model of SiFi's core expense management problem.

## Why this project

Built to directly match the requirements of the AI Engineer role at SiFi (Saudi fintech, Series A, SAMA EMI license): structured extraction from financial documents, expense categorization, compliance/policy enforcement, RAG pipelines, autonomous agent workflows, evaluation frameworks, and backend APIs.

## Architecture
Receipt image (PDF/JPG)
↓
[1] Structured extraction    GPT-4o Vision + Structured Outputs (strict mode)
↓
[2] Independent math check   sum(line_items) == subtotal ?  subtotal + tax == total ?
↓
[3] Hybrid classification    Merchant name rule match first → LLM (with item context) for the rest
↓
[4] Policy check (RAG)       Bullet-level chunking + ChromaDB + embeddings
↓
[5] Decision agent           Deterministic rules for clear cases → LLM for the gray zone
↓
[6] FastAPI                  POST /process-receipt
+
[7] Eval framework           ground_truth.json + 4 independently scored dimensions

## Design decisions, and why

**Two stages, not one call.** Extraction (vision) is separate from judgment (business logic). This allows a deterministic math check that never depends on the model "understanding" arithmetic correctly.

**Math check independent of the model's own output.** `check_math` and `check_totals` are plain Python summation — no extra LLM call. In practice they caught real extraction errors in nearly every run, including one the model's own output gave no signal about (a missing $5 line item in a real receipt, only caught by reconciling against the printed subtotal).

**Rule-before-LLM applied at every layer, not just classification.** The same principle was reused in both classification (day 2) and the final agent decision (day 3): explicit Python rules handle clear-cut cases, LLM is reserved for the gray zone only. In one test run, 5 of 7 final decisions came from a deterministic rule, not an LLM call.

**RAG chunked at bullet-level, not section-level.** Initial chunking (one section = one chunk) buried specific sub-rules inside longer mixed-topic text. Bullet-level chunking improved retrieval precision, but did not solve every case — see limitations below.

**A "needs review" state is a deliberate design choice, not a gap.** The system is not asked for 100% accuracy. It's asked to know when it shouldn't trust itself, and escalate to a human instead of guessing confidently.

## Limitations discovered in practice (not hypothetical)

Every item below is backed by an actual run during development, not a theoretical concern.

**1. Non-determinism persists even with `temperature=0`.**
Same image, same code, consecutive runs produced different math discrepancies (4.53 → 6.73 → 10.53 currency units) on the same line item. `temperature=0` reduces token-sampling randomness but does not eliminate variance in how the vision model "reads" an image on different passes. **Practical impact:** any eval framework relying on a single run per sample is inherently fragile; a trustworthy eval needs multiple runs per sample.

**2. Embeddings bind strongly to general concepts, weakly to specific brand names.**
Query "beer alcohol" matched the alcohol-prohibition policy chunk strongly (distance 1.32). The same real-world case, described only by a specific product name ("Miller Lite"), matched weakly (distance 1.68) — the correct policy chunk didn't even appear in the top 3 results. **Practical impact:** high-risk/high-compliance categories (alcohol, bribery, gambling) need an explicit keyword/brand rule layer on top of RAG, not reliance on embeddings alone.

**3. Errors compound silently across pipeline stages.**
The RAG mismatch above (item 2) propagated straight into the agent's decision, which then compared the receipt amount against an unrelated policy limit (a $200 team-meal cap, instead of flagging an outright prohibition). The agent had no way to know the retrieval step had failed — it acted with full confidence on a wrong input. This was only caught by comparing the final decision against an independent ground truth (`eval.py`), not by inspecting any single stage in isolation.

**4. Rule-based classification fails against real brand names.**
An initial keyword dictionary ("restaurant", "cafe") matched only 1 of 6 real restaurant receipts (Taco Bell, Golden Bowl Teriyaki, Saska's — none contain a word meaning "restaurant" in their actual trade name). 5 of 6 required an LLM call. **Practical impact:** the real rule/LLM split on production-like data is far lower than the naive assumption; the rule dictionary needs continuous expansion based on recurring LLM classification patterns.

## Evaluation

`eval.py` scores 4 independent dimensions against `ground_truth.json` (currently 7 samples, designed to scale to 50):
- Extraction accuracy (merchant, amount, currency)
- Classification accuracy
- Math-check accuracy (known issues correctly flagged, no false positives)
- Agent decision accuracy against a human-judged expected decision

Typical run: extraction 7/7, classification 7/7, math check 6-7/7 (variable, see limitation 1), agent decision 6/7 — the one failure is directly traceable to limitation 2 above.

## Running it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install openai pydantic python-dotenv chromadb numpy fastapi uvicorn python-multipart

# .env: OPENAI_API_KEY=sk-...

uvicorn api:app --reload
# POST http://127.0.0.1:8000/process-receipt  (multipart file upload)

python eval.py   # batch evaluation against ground_truth.json
```

## Stack

`openai` · `pydantic` · `chromadb` · `fastapi` · `uvicorn` · `numpy` · `python-dotenv`