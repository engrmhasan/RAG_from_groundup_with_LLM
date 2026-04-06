import re
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "/home/raco/MehediPcFiles/AgenticAI/QDrant/jina-embeddings-v5-text-nano"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
print("Model loaded.\n")


doc = """
Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,
and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately 
than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure.
Governments and private organizations alike are investing heavily in research and development to stay competitive in this evolving landscape.
As a result, the demand for skilled professionals in machine learning, data science, and AI ethics is growing significantly across the globe.

Despite these advancements, there are serious concerns that must be addressed to ensure the responsible use of artificial intelligence technologies.
Issues such as algorithmic bias, lack of transparency, and the potential for misuse in surveillance or misinformation campaigns have raised 
ethical questions among researchers and policymakers. Many experts argue that regulations and guidelines must be established to prevent harm 
while still encouraging innovation. Public awareness and education also play a crucial role in helping people understand both the benefits and
risks associated with AI, ensuring that society can make informed decisions about its implementation.

At the same time, ongoing research continues to push the boundaries of what machines are capable of achieving.
Breakthroughs in natural language processing allow AI systems to understand and generate human-like text,
while advancements in computer vision enable machines to interpret visual data with remarkable accuracy.
Robotics is also evolving, with autonomous systems being deployed in manufacturing, logistics, and even space exploration.
These innovations are not only expanding technological capabilities but also redefining the relationship between humans and machines in a rapidly changing world.
"""


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def window_similarity(embeddings: np.ndarray, i: int, window: int = 2) -> float:
    """
    Compare the centroid of the 'window' sentences before index i+1
    against the centroid of the 'window' sentences from index i+1 onward.
    This smooths over noisy transitional sentences.
    """
    left  = np.mean(embeddings[max(0, i - window + 1) : i + 1], axis=0)
    right = np.mean(embeddings[i + 1 : i + 1 + window], axis=0)
    # Embeddings are L2-normalized, so dot product == cosine similarity
    return float(np.dot(left, right))


def semantic_chunk(
    text: str,
    threshold_percentile: float = 70,
    min_sentences: int = 2,
    max_sentences: int = 10,
    similarity_floor: float = 0.3,
    window: int = 2,
) -> list[str]:

    print("Splitting text into sentences...")
    sentences = split_into_sentences(text)

    # If too few sentences, no point chunking
    if len(sentences) <= 1:
        return sentences

    print(f"  Sentences detected : {len(sentences)}")

    embeddings = model.encode(
        sentences,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        task="text-matching",
    )

    print(f"  Shape of Embeddings: {embeddings.shape}")

    similarities = [
        window_similarity(embeddings, i, window=window)
        for i in range(len(embeddings) - 1)
    ]

    print(f"  Similarity range   : {min(similarities):.4f} – {max(similarities):.4f}")

    if len(similarities) < 4:
        print("  Too few sentences to chunk meaningfully — returning as single chunk.")
        return [" ".join(sentences)]

    threshold = max(
        float(np.percentile(similarities, threshold_percentile)),
        similarity_floor,   # prevents over-chunking when all similarities are low
    )
    print(f"  Breakpoint threshold ({threshold_percentile}th pct, floor={similarity_floor}): {threshold:.4f}")

    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        topic_shift   = sim < threshold and len(current) >= min_sentences
        size_exceeded = len(current) >= max_sentences

        if topic_shift or size_exceeded:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC CHUNKING")
    print("=" * 60)

    chunks = semantic_chunk(
        doc,
        threshold_percentile=60,
        min_sentences=2,
        max_sentences=10,
        similarity_floor=0.3,
        window=2,
    )

    print(f"\n  Total chunks formed: {len(chunks)}\n")
    print("=" * 60)

    for idx, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"\n CHUNK {idx}  ({word_count} words)")
        print("-" * 60)
        print(chunk)

    print("\n" + "=" * 60)
    print("Done.")


