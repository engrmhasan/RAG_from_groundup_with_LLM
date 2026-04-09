from typing import List
import re


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



# Split the text into units (words, in this case)
def word_splitter(source_text: str) -> List[str]:
    source_text = re.sub(r"\s+", " ", source_text)  # Replace multiple whitespces
    return re.split(r"\s", source_text)  # Split by single whitespace

def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float = 0.2) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks


chunks = get_chunks_fixed_size_with_overlap(
    text=doc,
    chunk_size=10,
    overlap_fraction=0.2
)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")   

    