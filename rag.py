from pathlib import Path
import re

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def load_and_chunk_policy(file_path: Path) -> list[dict]:
    """
    يقرأ ملف السياسة، ويقسّمه إلى قطع دقيقة: كل نقطة (bullet) قطعة منفصلة،
    مع الاحتفاظ بعنوان القسم كسياق لكل قطعة.
    """
    text = file_path.read_text(encoding="utf-8")

    sections = re.split(r"(^## .+$)", text, flags=re.MULTILINE)

    chunks = []
    chunk_id = 0

    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        content = sections[i + 1].strip()

        bullets = [
            line.strip().lstrip("- ").strip()
            for line in content.split("\n")
            if line.strip().startswith("-")
        ]

        for bullet in bullets:
            chunks.append({
                "heading": heading,
                "text": f"{heading}\n{bullet}",
                "id": f"chunk_{chunk_id}",
            })
            chunk_id += 1

    return chunks


def get_embedding(text: str) -> list[float]:
    """
    يحوّل نصاً إلى متجه (قائمة أرقام) يمثّل معناه.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def build_policy_collection(chunks: list[dict]):
    """
    ينشئ (أو يفتح) قاعدة بيانات ChromaDB محلية، ويخزّن فيها كل قطع السياسة.
    """
    db_client = chromadb.PersistentClient(path="./chroma_data")

    try:
        db_client.delete_collection(name="company_policy")
    except Exception:
        pass

    collection = db_client.get_or_create_collection(name="company_policy")

    for chunk in chunks:
        collection.upsert(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[get_embedding(chunk["text"])],
            metadatas=[{"heading": chunk["heading"]}],
        )

    return collection


def search_policy(collection, query: str, n_results: int = 2):
    """
    يبحث في قاعدة السياسة عن أقرب n_results قطع لمعنى السؤال.
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results


_collection = None


def get_policy_collection():
    """
    يفتح (أو يبني إن لم تكن موجودة) قاعدة السياسة، ويحتفظ بها لإعادة الاستخدام.
    """
    global _collection
    if _collection is None:
        chunks = load_and_chunk_policy(Path("policy.md"))
        _collection = build_policy_collection(chunks)
    return _collection


def check_against_policy(
    category: str,
    amount: float,
    merchant: str,
    item_descriptions: list[str] = None,
) -> dict:
    """
    يبحث عن أقرب بند سياسة متعلق بهذا المصروف، ويعيد النص + المسافة.
    item_descriptions تضيف سياقاً أدق (مثلاً وجود كحول ضمن البنود).
    """
    collection = get_policy_collection()
    query = f"مصروف من فئة {category}، التاجر {merchant}، المبلغ {amount} ريال"

    if item_descriptions:
        items_text = "، ".join(item_descriptions)
        query += f"، البنود: {items_text}"

    results = search_policy(collection, query, n_results=1)

    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]
    top_distance = results["distances"][0][0]

    return {
        "policy_text": top_doc,
        "policy_heading": top_meta["heading"],
        "distance": top_distance,
    }


if __name__ == "__main__":
    chunks = load_and_chunk_policy(Path("policy.md"))
    collection = build_policy_collection(chunks)

    # اختبار أ: بكلمة "Miller Lite" فقط (اسم علامة تجارية محدد)
    print("=== اختبار أ: Miller Lite ===")
    results_a = search_policy(collection, "Miller Lite", n_results=3)
    for doc, dist in zip(results_a["documents"][0], results_a["distances"][0]):
        print(f"[{dist:.4f}] {doc[:70]}")

    print()

    # اختبار ب: بكلمة عامة صريحة (beer / alcohol)
    print("=== اختبار ب: beer alcohol drink ===")
    results_b = search_policy(collection, "beer alcohol drink", n_results=3)
    for doc, dist in zip(results_b["documents"][0], results_b["distances"][0]):
        print(f"[{dist:.4f}] {doc[:70]}")

    print()

    # اختبار ج: بالعربي "بيرة كحول"
    print("=== اختبار ج: بيرة كحول ===")
    results_c = search_policy(collection, "بيرة كحول", n_results=3)
    for doc, dist in zip(results_c["documents"][0], results_c["distances"][0]):
        print(f"[{dist:.4f}] {doc[:70]}")