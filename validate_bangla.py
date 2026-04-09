import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/home/raco/MehediPcFiles/AgenticAI/QDrant/jina-embeddings-v5-text-nano"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

# ============================================================
# TEST: Mini Retrieval Accuracy
# ============================================================
print("\n" + "=" * 50)
print("TEST Results: Mini Retrieval Accuracy")
print("=" * 50)

queries = [
    "বাংলাদেশের রাজধানী কোথায়?",
    "কৃত্রিম বুদ্ধিমত্তা কী?",
    "বাজারে শাকসবজির দাম কেমন?",
]

documents = [
    "ঢাকা বাংলাদেশের রাজধানী এবং দক্ষিণ এশিয়ার অন্যতম বৃহৎ শহর।",
    "কৃত্রিম বুদ্ধিমত্তা বা এআই হলো কম্পিউটার ব্যবস্থা যা মানুষের মতো চিন্তা করতে পারে।",
    "রাজধানী ঢাকায় জনসংখ্যা অনেক বেশি এবং যানজট একটি বড় সমস্যা।",
    "আজকে বাজারে আলুর দাম কমেছে, তবে পেঁয়াজের দাম বেড়েছে।",
    "পাইথন একটি জনপ্রিয় প্রোগ্রামিং ভাষা যা ডাটা সায়েন্সে ব্যাপক ব্যবহৃত।",
    "কৃত্রিম বুদ্ধিমত্তা স্বাস্থ্যসেবা, শিক্ষা এবং পরিবহনে বিপ্লব আনছে।",
]

expected = {
    0: [0, 2],
    1: [1, 5],
    2: [3],
}

with torch.no_grad():
    q_embeds = np.array(
        [
            e.cpu().numpy() if isinstance(e, torch.Tensor) else np.array(e)
            for e in model.encode(queries, task="retrieval", prompt_name="query")
        ]
    )
    d_embeds = np.array(
        [
            e.cpu().numpy() if isinstance(e, torch.Tensor) else np.array(e)
            for e in model.encode(documents, task="retrieval", prompt_name="document")
        ]
    )

correct = 0
total = 0

for i, query in enumerate(queries):
    sims = d_embeds @ q_embeds[i]
    top3 = np.argsort(sims)[::-1][:3].tolist()

    print(f"\n  Query: {query}")
    for rank, doc_idx in enumerate(top3, 1):
        marker = " <-- MATCH" if doc_idx in expected[i] else ""
        print(
            f"    {rank}. [sim={sims[doc_idx]:.4f}] {documents[doc_idx][:70]}{marker}"
        )

    hits = sum(1 for j in top3 if j in expected[i])
    correct += hits
    total += len(expected[i])

recall = correct / total if total > 0 else 0

print(f"\n  Hits: {correct}/{total} | Recall@3: {recall:.2f}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Retrieval:    {'PASS' if recall >= 0.5 else 'FAIL'} (recall {recall:.2f})")




# 
# ==================================================
# TEST Results: Mini Retrieval Accuracy
# ==================================================

#   Query: বাংলাদেশের রাজধানী কোথায়?
#     1. [sim=0.6765] ঢাকা বাংলাদেশের রাজধানী এবং দক্ষিণ এশিয়ার অন্যতম বৃহৎ শহর। <-- MATCH
#     2. [sim=0.5287] রাজধানী ঢাকায় জনসংখ্যা অনেক বেশি এবং যানজট একটি বড় সমস্যা। <-- MATCH
#     3. [sim=0.1915] পাইথন একটি জনপ্রিয় প্রোগ্রামিং ভাষা যা ডাটা সায়েন্সে ব্যাপক ব্যবহৃত।

#   Query: কৃত্রিম বুদ্ধিমত্তা কী?
#     1. [sim=0.7838] কৃত্রিম বুদ্ধিমত্তা বা এআই হলো কম্পিউটার ব্যবস্থা যা মানুষের মতো চিন্ত <-- MATCH
#     2. [sim=0.5763] কৃত্রিম বুদ্ধিমত্তা স্বাস্থ্যসেবা, শিক্ষা এবং পরিবহনে বিপ্লব আনছে। <-- MATCH
#     3. [sim=0.1449] পাইথন একটি জনপ্রিয় প্রোগ্রামিং ভাষা যা ডাটা সায়েন্সে ব্যাপক ব্যবহৃত।

#   Query: বাজারে শাকসবজির দাম কেমন?
#     1. [sim=0.5231] আজকে বাজারে আলুর দাম কমেছে, তবে পেঁয়াজের দাম বেড়েছে। <-- MATCH
#     2. [sim=0.1541] ঢাকা বাংলাদেশের রাজধানী এবং দক্ষিণ এশিয়ার অন্যতম বৃহৎ শহর।
#     3. [sim=0.1466] রাজধানী ঢাকায় জনসংখ্যা অনেক বেশি এবং যানজট একটি বড় সমস্যা।

#   Hits: 5/5 | Recall@3: 1.00

# ==================================================
# SUMMARY
# ==================================================
#   Retrieval:    PASS (recall 1.00)




