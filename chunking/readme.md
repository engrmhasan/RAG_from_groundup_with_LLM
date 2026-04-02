I am following weaviate documentation and breaking down everything from core...

doc link: https://weaviate.io/blog/chunking-strategies-for-rag



# Fixed Size Chunking Code step by step


The code’s purpose is to **split a long text into fixed-size chunks of words**, where each chunk can overlap with the previous chunk. This is particularly useful in:

* NLP preprocessing
* Handling large texts for embeddings
* Sliding window text processing

The key components:

1. `word_splitter` – splits text into words
2. `get_chunks_fixed_size_with_overlap` – creates chunks of a specified size with optional overlap

---

## 1. Word splitting

```python
def word_splitter(source_text: str) -> List[str]:
    source_text = re.sub(r"\s+", " ", source_text)  # Replace multiple whitespaces
    return re.split(r"\s", source_text)  # Split by single whitespace
```

### Step-by-step

1. `re.sub(r"\s+", " ", source_text)`

   * `\s+` matches one or more whitespace characters (spaces, tabs, newlines)
   * Replaces all consecutive whitespace with a **single space**
     Example: `"Hello   world\nhi"` → `"Hello world hi"`

2. `re.split(r"\s", source_text)`

   * Splits the text by **any whitespace**
     Example: `"Hello world hi"` → `["Hello", "world", "hi"]`


---

## 2. Chunking function

```python
def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float = 0.2) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks
```

### Step-by-step explanation

1. **Convert text to words**:

```python
text_words = word_splitter(text)
```

* Example: `"Hello world hi"` → `["Hello", "world", "hi"]`

2. **Calculate overlap**:

```python
overlap_int = int(chunk_size * overlap_fraction)
```

* Determines how many words from the previous chunk should overlap
* Example: `chunk_size = 10, overlap_fraction = 0.2 → overlap_int = 2`

3. **Loop through the text**:

```python
for i in range(0, len(text_words), chunk_size):
```

* `i` marks the starting index of each chunk **without overlap**
* Steps by `chunk_size` each iteration

4. **Select words for the current chunk**:

```python
chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
```

* **Start index**: `max(i - overlap_int, 0)`
  Ensures overlap with previous chunk while preventing negative indexing
* **End index**: `i + chunk_size`
  Grabs the next `chunk_size` words

Example (chunk_size=5, overlap=2):

| i  | start | end | chunk words   |
| -- | ----- | --- | ------------- |
| 0  | 0     | 5   | A B C D E     |
| 5  | 3     | 10  | D E F G H I J |
| 10 | 8     | 15  | I J K L       |

5. **Convert list of words back to string**:

```python
chunk = " ".join(chunk_words)
chunks.append(chunk)
```

* Adds the chunk to the `chunks` list

6. **Return all chunks**:

```python
return chunks
```

---

## 3. Output example

Using:

```python
chunks = get_chunks_fixed_size_with_overlap(
    "This is an example of fixed size chunking with overlap. The text will be split into chunks of a specified size, and each chunk will have some overlap with the previous chunk to ensure that important information is not lost. This method is useful for processing large texts in smaller, manageable pieces.",
    chunk_size=10,
    overlap_fraction=0.2
)
```

Output:

```
Chunk 1:
This is an example of fixed size chunking with overlap.

Chunk 2:
with overlap. The text will be split into chunks of a specified

Chunk 3:
a specified size, and each chunk will have some overlap with the

Chunk 4:
with the previous chunk to ensure that important information is not lost.

Chunk 5:
not lost. This method is useful for processing large texts in smaller,

Chunk 6:
in smaller, manageable pieces.
```

---

## 4. Key points

1. **Overlap is implemented by stepping back from the current index**.
2. **Python slicing handles out-of-bounds automatically**, so no errors occur if the last chunk is shorter.
3. **Raw strings** should be used for regex to avoid `SyntaxWarning`.
4. This method is efficient for **jumping windows with backward overlap**, not a fully sliding window.

---



