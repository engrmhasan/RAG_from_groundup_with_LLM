I am following weaviate documentation and breaking down everything from core...

doc link: https://weaviate.io/blog/chunking-strategies-for-rag



# 1. Fixed Size Chunking Code step by step


The code’s purpose is to **split a long text into fixed-size chunks of words**, where each chunk can overlap with the previous chunk. This is particularly useful in:

* NLP preprocessing
* Handling large texts for embeddings
* Sliding window text processing

The key components:

1. `word_splitter` – splits text into words
2. `get_chunks_fixed_size_with_overlap` – creates chunks of a specified size with optional overlap



## 1. Word splitting

```
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



## 2. Chunking function

```
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

```
text_words = word_splitter(text)
```

* Example: `"Hello world hi"` → `["Hello", "world", "hi"]`

2. **Calculate overlap**:

```
overlap_int = int(chunk_size * overlap_fraction)
```

* Determines how many words from the previous chunk should overlap
* Example: `chunk_size = 10, overlap_fraction = 0.2 → overlap_int = 2`

3. **Loop through the text**:

```
for i in range(0, len(text_words), chunk_size):
```

* `i` marks the starting index of each chunk **without overlap**
* Steps by `chunk_size` each iteration

4. **Select words for the current chunk**:

```
chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
```

* **Start index**: `max(i - overlap_int, 0)`
  Ensures overlap with previous chunk while preventing negative indexing
* **End index**: `i + chunk_size`
  Grabs the next `chunk_size` words

Example (chunk_size=5, overlap=2):

| i  | start | end | chunk words   |
| -- | -- |  | - |
| 0  | 0     | 5   | A B C D E     |
| 5  | 3     | 10  | D E F G H I J |
| 10 | 8     | 15  | I J K L       |

5. **Convert list of words back to string**:

```
chunk = " ".join(chunk_words)
chunks.append(chunk)
```

* Adds the chunk to the `chunks` list

6. **Return all chunks**:

```
return chunks
```


## 3. Output example

Using:

```
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


## 4. Key points

1. **Overlap is implemented by stepping back from the current index**.
2. ** slicing handles out-of-bounds automatically**, so no errors occur if the last chunk is shorter.
3. **Raw strings** should be used for regex to avoid `SyntaxWarning`.
4. This method is efficient for **jumping windows with backward overlap**, not a fully sliding window.


# 2. Greedy Chunk Construction


## Core Idea

This algorithm takes text that has already been split into `parts` using a separator, then rebuilds it into chunks where:

- Each chunk is as large as possible
- No chunk exceeds `max_chunk_size`

It keeps adding pieces into a growing `current_chunk` until adding one more would overflow the limit.


## Key Variables

```
parts = text.split(separator)
chunks = []
current_chunk = ""
```

- `parts` — list of smaller pieces produced by splitting
- `current_chunk` — the chunk currently being built
- `chunks` — the final list of completed chunks



## Critical Logic

```
test_chunk = current_chunk + separator + part if current_chunk else part
```

When you call `text.split(separator)`, the separator is removed from the result. This line puts it back when combining parts, so the original structure of the text is preserved.

Without this, combining parts would silently drop punctuation or spacing.



## Properties of the Algorithm

**Greedy** — fits as much as possible into each chunk before sealing it.

**Order-preserving** — parts are always added in their original sequence.

**Structure-preserving** — the separator is re-inserted when joining parts, keeping meaning intact.



## Visual Intuition

Think of it like filling a box:

- Each `part` is an item
- `max_chunk_size` is the box capacity
- `current_chunk` is the box currently being packed

Steps:
1. Try to put the item in the current box
2. If it fits, keep going
3. If it does not fit, seal the box and start a new one


## Step-by-Step Example

### Input

```
text = "I love AI. It is powerful. It is the future."
separator = ". "
max_chunk_size = 25
```

### Step 1: Split

```
parts = text.split(". ")

parts = [
    "I love AI",
    "It is powerful",
    "It is the future."
]
```

Note: the separator `. ` is removed. The last part retains its `.` because split only removes exact matches.

### Initial State

```
chunks = []
current_chunk = ""
```

### Iteration 1

```
part = "I love AI"
```

`current_chunk` is empty, so:

```
test_chunk = part
           = "I love AI"
length = 10
```

Check: `10 <= 25` — fits.

```
current_chunk = "I love AI"
```

### Iteration 2

```
part = "It is powerful"
```

```
test_chunk = current_chunk + ". " + part
           = "I love AI" + ". " + "It is powerful"
           = "I love AI. It is powerful"
length = 10 + 2 + 15 = 27
```

Check: `27 > 25` — overflow.

Seal the current chunk:

```
chunks.append(current_chunk.strip())
chunks = ["I love AI"]
```

Start fresh:

```
current_chunk = "It is powerful"
```


### Iteration 3

```
part = "It is the future."
```

```
test_chunk = "It is powerful" + ". " + "It is the future."
           = "It is powerful. It is the future."
length = 15 + 2 + 19 = 36
```

Check: `36 > 25` — overflow.

Seal the current chunk:

```
chunks = ["I love AI", "It is powerful"]
```

Start fresh:

```
current_chunk = "It is the future."
```

### After the Loop

The last chunk must be saved manually — the loop does not do it automatically:

```
if current_chunk:
    chunks.append(current_chunk.strip())
```

### Final Result

```
chunks = [
    "I love AI",
    "It is powerful",
    "It is the future."
]
```


## Why `current_chunk` Must Reset on Overflow

When overflow happens:

```
current_chunk = part
```

The new chunk starts fresh from the overflowing part. The algorithm never tries to force-fit a part into an already-full chunk.


## Edge Case: A Single Part That Exceeds the Limit

If one part is already larger than `max_chunk_size`:

```
part = "very very very long sentence..."
```

It still becomes `current_chunk = part`. The algorithm does not discard it. Recursive splitting at a finer separator level handles it in the next pass.



## Extended Example: Text Larger Than Max Chunk Size


### Input

```
Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,
and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately 
than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure.

Governments and private organizations alike are investing heavily in research and development to stay competitive in this evolving landscape.
As a result, the demand for skilled professionals in machine learning, data science, and AI ethics is growing significantly across the globe.
```

Settings:

```
max_chunk_size = 200
separators = ["\n\n", "\n", ". ", " "]
```

- Primary: `"\n\n"` — paragraph boundary
- Secondary: `"\n"` — line boundary
- Tertiary: `". "` — sentence boundary
- Fallback: `" "` — word boundary


## Step 1: Try Primary Separator `"\n\n"`

```
parts = text.split("\n\n")

parts = [
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,\nand daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately \nthan human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure.",

    "Governments and private organizations alike are investing heavily in research and development to stay competitive in this evolving landscape.\nAs a result, the demand for skilled professionals in machine learning, data science, and AI ethics is growing significantly across the globe."
]
```

Both paragraphs are larger than 200 characters, so each one must be split further using the secondary separator.


## Step 2: Split Paragraph 1 by `"\n"` (Line Boundary)

```
lines = [
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,",
    "and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately ",
    "than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure."
]
```

Now these lines are processed with the greedy chunking logic.


## Step 3: Chunk Construction for Paragraph 1

### Initial State

```
chunks = []
current_chunk = ""
```

### Iteration 1

```
part = "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,"
```

`current_chunk` is empty:

```
test_chunk = part
length = 113
```

Check: `113 <= 200` — fits.

```
current_chunk = part
```

### Iteration 2

```
part = "and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately "
```

```
test_chunk = current_chunk + "\n" + part
length = 113 + 1 + 146 = 260
```

Check: `260 > 200` — overflow.

Seal the current chunk:

```
chunks.append(current_chunk.strip())

chunks = [
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,"
]
```

`.strip()` removes any accidental leading or trailing whitespace.

Start fresh:

```
current_chunk = part
```

### Iteration 3

```
part = "than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure."
```

```
test_chunk = current_chunk + "\n" + part
length = 146 + 1 + 155 = 302
```

Check: `302 > 200` — overflow.

Seal the current chunk:

```
chunks = [
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,",
    "and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately"
]
```

Start fresh:

```
current_chunk = part
```

### End of Paragraph 1 Loop

```
if current_chunk:
    chunks.append(current_chunk.strip())
```

Final chunks after paragraph 1:

```
chunks = [
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,",
    "and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately",
    "than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure."
]
```

The trailing `chunks.append` ensures the last piece is never lost.


## Step 4: Repeat for Paragraph 2

The same line-splitting and greedy packing logic produces:

```
chunks += [
    "Governments and private organizations alike are investing heavily in research and development to stay competitive in this evolving landscape.",
    "As a result, the demand for skilled professionals in machine learning, data science, and AI ethics is growing significantly across the globe."
]
```


## Step 5: Final Output

```
[
    "Artificial intelligence is transforming the world at an unprecedented pace, influencing industries, societies,",
    "and daily human life in ways that were once considered science fiction. From healthcare systems that can diagnose diseases more accurately",
    "than human doctors to financial tools that predict market trends using massive datasets, AI is rapidly becoming a core part of modern infrastructure.",
    "Governments and private organizations alike are investing heavily in research and development to stay competitive in this evolving landscape.",
    "As a result, the demand for skilled professionals in machine learning, data science, and AI ethics is growing significantly across the globe."
]
```

Every chunk is within the 200-character limit. Meaning is preserved across chunk boundaries. No whitespace leaks in at the start or end of any chunk thanks to `.strip()`.


