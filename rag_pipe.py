# Stage 0 — model setup
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


MODEL_PATH = "/home/raco/MehediPcFiles/AgenticAI/QDrant/jina-embeddings-v5-text-nano"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
).to("cuda")
model.eval()



# Stage 1 — Create collection 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(path="./qdrant_storage")

if client.collection_exists("bangla_docs"):
    client.delete_collection("bangla_docs")
    client.create_collection(
        collection_name="bangla_docs",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )


# Stage 2 — Chunk → Embed → Upsert 
from qdrant_client.models import PointStruct
import torch
import numpy as np

documents = [
    {"text": "ঢাকা বাংলাদেশের রাজধানী এবং দক্ষিণ এশিয়ার অন্যতম বৃহৎ শহর।", "source": "local", "lang": "bn"},
    {"text": "কৃত্রিম বুদ্ধিমত্তা বা এআই হলো কম্পিউটার ব্যবস্থা যা মানুষের মতো চিন্তা করতে পারে।", "source": "local", "lang": "bn"},
    {"text": "রাজধানী ঢাকায় জনসংখ্যা অনেক বেশি এবং যানজট একটি বড় সমস্যা।", "source": "local", "lang": "bn"},
    {"text": "আজকে বাজারে আলুর দাম কমেছে, তবে পেঁয়াজের দাম বেড়েছে।", "source": "local", "lang": "bn"},
    {"text": "পাইথন একটি জনপ্রিয় প্রোগ্রামিং ভাষা যা ডাটা সায়েন্সে ব্যাপক ব্যবহৃত।", "source": "local", "lang": "bn"},
    {"text": "কৃত্রিম বুদ্ধিমত্তা স্বাস্থ্যসেবা, শিক্ষা এবং পরিবহনে বিপ্লব আনছে।", "source": "local", "lang": "bn"},
]

def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

points = []
point_id = 0

for doc in documents:
    chunks = chunk_text(doc["text"])

    with torch.no_grad():
        embeddings = model.encode(
            chunks,
            task="retrieval",
            prompt_name="document"
        )

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vec = emb.cpu().numpy().tolist() if isinstance(emb, torch.Tensor) else list(emb)

        points.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "text": chunk,
                "source": doc["source"],
                "lang": doc["lang"],
                "chunk_index": i,
            }
        ))
        point_id += 1

# Upsert in batches
batch_size = 128
for i in range(0, len(points), batch_size):
    client.upsert(
        collection_name="bangla_docs",
        points=points[i:i + batch_size]
    )

print(f"Stored {len(points)} points")


# Stage 3 — Query at search time
query = "বাংলাদেশের রাজধানী কোথায়?"

with torch.no_grad():
    q_vec = model.encode([query], task="retrieval", prompt_name="query")[0]
    q_vec = q_vec.cpu().numpy().tolist() if isinstance(q_vec, torch.Tensor) else list(q_vec)

results = client.query_points(
    collection_name="bangla_docs",
    query=q_vec,
    limit=2
).points

for r in results:
    print(f"score={r.score:.4f} | {r.payload['text'][:80]}")

client.close() 

