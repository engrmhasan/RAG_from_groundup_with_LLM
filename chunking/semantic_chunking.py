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

```python
# Example: Simple neural network in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
model = SimpleNet()
print(f"Model architecture: {model}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy dataset (1000 samples, 784 features like 28x28 images)
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).float().mean()
    print(f"Training Accuracy: {accuracy:.4f}")
```

The future of AI holds tremendous promise. Researchers are exploring new frontiers in quantum machine learning,
explainable AI, and artificial general intelligence. These developments could revolutionize fields ranging from
drug discovery to climate modeling, offering solutions to some of humanity's most pressing challenges.


War has been a recurring part of human history, shaping nations, borders, and societies. It is often defined as a large-scale conflict between groups, though it may appear in writing as WAR, war, or even WaR depending on context and style. While some refer to a “war of words,” others speak of real যুদ্ধ or حرب, where the consequences are far more severe. At times, people insist, “This is not a war,” while others respond, “This is war!!!” reflecting how perception and expression can differ.

One of the primary causes of war123 is the struggle for power and control. Nations may engage in conflict to expand their territory, access valuable resources, or assert dominance over others. Political disagreements, such as differing systems of governance, can escalate quickly. In everyday language, phrases like #war or @conflict may trend online, even when no actual violence is occurring, creating a contrast between symbolic and real-world meanings.

Economic factors also play a crucial role. Competition over resources such as oil, water, and minerals has fueled many conflicts throughout history. In some cases, prolonged instability leads to drawn-out struggles that feel like warrrrrrrrrrrrrrrrrr, stretching across years with no clear resolution. In multilingual societies, people may say “war hocche na but conflict cholche,” blending languages to describe situations that are neither fully peaceful nor openly violent.

The consequences of war are devastating. War. war? WAR! Regardless of how it is written, the outcome often includes loss of life, displacement, and suffering. Soldiers and civilians alike endure hardship, and families are torn apart. Yet sometimes, the term is used sarcastically, as in “Yeah, total war… over coffee,” where the meaning is far from literal.

In written and spoken communication, variations such as wra, wr, wa,r, or warr may appear, especially in informal contexts. Despite these differences, the intended meaning often remains recognizable. At the same time, empty expressions like "" or even a pause represented as " " can carry meaning depending on context, just as silence can speak volumes during times of conflict.

War leaves behind lasting psychological and social effects. Survivors may suffer from trauma, while children growing up in such environments often miss opportunities for education and stability. However, context changes everything: discussions about war in games differ greatly from discussions about war in history, even if the same words are used.

Modern communication adds new layers of expression. Symbols and signs such as ⚔️🔥💣🕊️ can represent destruction, violence, or hope for peace, depending on how they are used. Likewise, combinations like $peace or %100 support may appear in messages advocating for resolution rather than conflict.

"""


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def window_similarity(embeddings: np.ndarray, i: int, window: int = 2) -> float:
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
        task="retrieval",
    )

    print(f"\n\nShape of Embeddings: {embeddings.shape}")

    similarities = [
        window_similarity(embeddings, i, window=window)
        for i in range(len(embeddings) - 1)
    ]

    print(f"Similarity range   : {min(similarities):.4f} – {max(similarities):.4f}")

    if len(similarities) < 4:
        print("Too few sentences to chunk meaningfully — returning as single chunk.")
        return [" ".join(sentences)]

    threshold = max(
        float(np.percentile(similarities, threshold_percentile)),
        similarity_floor,   # prevents over-chunking when all similarities are low
    )
    print(f"Breakpoint threshold ({threshold_percentile}th pct, floor={similarity_floor}): {threshold:.4f}")

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

    print(f"\nTotal chunks formed: {len(chunks)}\n")
    print("=" * 60)

    for idx, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"\n CHUNK {idx}  ({word_count} words)")
        print("-" * 60)
        print(chunk)

    print("\n" + "=" * 60)
    print("Done.")


