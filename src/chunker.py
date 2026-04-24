from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    chunk_total: int


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if not text.strip():
        return [Chunk(text=text, chunk_index=0, chunk_total=1)]
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    slices = []
    start = 0
    while start < len(text):
        slices.append(text[start:start + chunk_size])
        start += step
    # Merge last chunk if too small to avoid creating a fragment with insufficient
    # context for meaningful semantic search. The minimum size is calculated as
    # max(50, chunk_size // 5) to ensure:
    # - At least 50 characters to maintain minimum semantic coherence
    # - At least 20% of the standard chunk size to prevent disproportionately
    #   small final chunks that would produce poor quality embeddings
    min_chunk_size = max(50, chunk_size // 5)
    if len(slices) > 1 and len(slices[-1]) < min_chunk_size:
        slices[-2] = slices[-2] + ' ' + slices[-1].lstrip()
        slices.pop()
    total = len(slices)
    return [Chunk(text=s, chunk_index=i, chunk_total=total) for i, s in enumerate(slices)]
