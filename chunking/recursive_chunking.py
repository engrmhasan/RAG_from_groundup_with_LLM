from typing import List

def recursive_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
    # Base case: if text is small enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Try separators in priority order
    separators = ["\n\n", "\n", ". ", " "]
    
    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # Reconstruct the chunk/part with separator to preserve meaning and check if adding this part would exceed the limit
                test_chunk = current_chunk + separator + part if current_chunk else part
                
                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            # Add the final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Recursively process any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)
            
            return [chunk for chunk in final_chunks if chunk]
    
    # Fallback: split by character limit if no separators work
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


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

chunks = recursive_chunking(doc, max_chunk_size=200)

# Print results
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} (length {len(chunk)}):\n{chunk}\n{'-'*40}")


