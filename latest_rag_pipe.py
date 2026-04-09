import gc
import torch
import numpy as np
import time
import psutil
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── helpers ──────────────────────────────────────────────────────────────
process = psutil.Process(os.getpid())


def cpu_ram_mb():
    return process.memory_info().rss / 1024**2


def gpu_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


class StageTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # flush GPU queue before timing
        self.t0 = time.perf_counter()
        self.cpu_before = cpu_ram_mb()
        self.gpu_before = gpu_vram_mb()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for GPU to finish
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000
        self.cpu_delta = cpu_ram_mb() - self.cpu_before
        self.gpu_delta = gpu_vram_mb() - self.gpu_before
        print(f"  [{self.name}]")
        print(f"    time     : {self.elapsed_ms:.1f} ms")
        print(f"    cpu ram  : {cpu_ram_mb():.1f} MB  (Δ {self.cpu_delta:+.1f} MB)")
        print(f"    gpu vram : {gpu_vram_mb():.1f} MB  (Δ {self.gpu_delta:+.1f} MB)")


# ── reset GPU peak counter ───────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 0 — Model Load ===")

from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/home/raco/MehediPcFiles/AgenticAI/QDrant/jina-embeddings-v5-text-nano"

with StageTimer("tokenizer load"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

with StageTimer("model load + to(cuda)"):
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    model = torch.compile(model)

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 1 — Qdrant Collection ===")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)

with StageTimer("qdrant client init"):
    client = QdrantClient(path="./qdrant_storage")

with StageTimer("create collection"):
    if client.collection_exists("bangla_docs"):
        client.delete_collection("bangla_docs")
    client.create_collection(
        collection_name="bangla_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            ef_construct=128,
            m=16,
            full_scan_threshold=10,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
    )

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 2 — Chunking ===")

documents = [
    {
        "text": "ঢাকা বাংলাদেশের রাজধানী এবং দক্ষিণ এশিয়ার অন্যতম বৃহৎ শহর।",
        "source": "local",
        "lang": "bn",
    },
    {
        "text": "কৃত্রিম বুদ্ধিমত্তা বা এআই হলো কম্পিউটার ব্যবস্থা যা মানুষের মতো চিন্তা করতে পারে।",
        "source": "local",
        "lang": "bn",
    },
    {
        "text": "রাজধানী ঢাকায় জনসংখ্যা অনেক বেশি এবং যানজট একটি বড় সমস্যা।",
        "source": "local",
        "lang": "bn",
    },
    {
        "text": "আজকে বাজারে আলুর দাম কমেছে, তবে পেঁয়াজের দাম বেড়েছে।",
        "source": "local",
        "lang": "bn",
    },
    {
        "text": "পাইথন একটি জনপ্রিয় প্রোগ্রামিং ভাষা যা ডাটা সায়েন্সে ব্যাপক ব্যবহৃত।",
        "source": "local",
        "lang": "bn",
    },
    {
        "text": "কৃত্রিম বুদ্ধিমত্তা স্বাস্থ্যসেবা, শিক্ষা এবং পরিবহনে বিপ্লব আনছে।",
        "source": "local",
        "lang": "bn",
    },
]


def chunk_text(text, size=300, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


with StageTimer("chunking all docs"):
    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"chunk": chunk, "meta": doc})

print(f"    chunks produced: {len(all_chunks)}")

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 3 — Embedding ===")

from qdrant_client.models import PointStruct

texts = [c["chunk"] for c in all_chunks]

with StageTimer("model.encode (GPU)") as embed_timer:
    with torch.no_grad():
        raw_embeddings = model.encode(texts, task="retrieval", prompt_name="document")
throughput = len(texts) / (embed_timer.elapsed_ms / 1000)


with StageTimer("tensor → cpu → numpy → list"):
    if isinstance(raw_embeddings, torch.Tensor):
        vectors = raw_embeddings.cpu().numpy().tolist()
    else:
        vectors = np.array(raw_embeddings).tolist()

del raw_embeddings
gc.collect()
torch.cuda.empty_cache()

print(f"    vector dim: {len(vectors[0])}")
print(f"    throughput: {throughput:.1f} chunks/s")

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 4 — PointStruct Build ===")

with StageTimer("build PointStruct list"):
    points = []
    for idx, (c, vec) in enumerate(zip(all_chunks, vectors)):
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "text": c["chunk"],
                    "source": c["meta"]["source"],
                    "lang": c["meta"]["lang"],
                    "chunk_index": idx,
                },
            )
        )

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 5 — Qdrant Upsert ===")

with StageTimer(f"upload_points {len(points)} points"):
    client.upload_points(
        collection_name="bangla_docs",
        points=points,
        batch_size=128,
        parallel=2,  # increase to 2-4 for NVMe SSD
    )

# upload_points is Qdrant's dedicated bulk ingestion method. It handles batching internally,'
# 'uses a more efficient binary serialization path, and skips per-batch HTTP overhead that upsert has.'
# 'For 6 points the difference is small, but at 10k+ points it's meaningfully faster.

# ════════════════════════════════════════════════════════════════════════
print("\n=== STAGE 6 — Query Latency (10 runs) ===")

query = "বাংলাদেশের রাজধানী কোথায়?"

with torch.no_grad():
    q_raw = model.encode([query], task="retrieval", prompt_name="query")[0]
    q_vec = (
        q_raw.cpu().numpy().tolist() if isinstance(q_raw, torch.Tensor) else list(q_raw)
    )

latencies = []
for run in range(10):
    t0 = time.perf_counter()
    results = client.query_points(
        collection_name="bangla_docs", query=q_vec, limit=2
    ).points
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
print(f"    p50  : {latencies[4]:.2f} ms")
print(f"    p90  : {latencies[8]:.2f} ms")
print(f"    p99  : {latencies[-1]:.2f} ms")
print(f"    min  : {latencies[0]:.2f} ms")
print(f"    max  : {latencies[-1]:.2f} ms")

# ════════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY ===")
print(f"  Peak GPU VRAM   : {gpu_peak_mb():.1f} MB")
print(f"  Final CPU RAM   : {cpu_ram_mb():.1f} MB")
print(f"  Total vectors   : {len(points)}")
print(f"  Query p50       : {latencies[4]:.2f} ms")

client.close()
