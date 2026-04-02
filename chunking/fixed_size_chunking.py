from typing import List
import re

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
    "This is an example of fixed size chunking with overlap. The text will be split into chunks of a specified size, and each chunk will have some overlap with the previous chunk to ensure that important information is not lost. This method is useful for processing large texts in smaller, manageable pieces.",
    chunk_size=10,
    overlap_fraction=0.2
)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")   

    