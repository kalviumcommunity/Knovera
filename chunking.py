import json

def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Task 1: Split using a defined strategy (Fixed-size with overlap)
    """
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += (chunk_size - overlap)
        
        if i >= len(text):
            break
    return chunks

def paragraph_chunking(text: str) -> list[str]:
    """
    Task 1: Split using a defined strategy (Paragraph based)
    """
    # Split by double newline which usually indicates paragraphs
    raw_chunks = text.split('\n\n')
    # Filter out empty chunks and strip whitespace
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

def get_stats(chunks: list[str]) -> dict:
    if not chunks:
        return {"count": 0, "avg_size": 0}
    
    total_size = sum(len(c) for c in chunks)
    return {
        "count": len(chunks),
        "avg_size": round(total_size / len(chunks), 2)
    }

def main():
    # Read the document
    try:
        with open("data/customer_policy.txt", "r", encoding="utf-8") as f:
            document = f.read()
    except FileNotFoundError:
        document = "This is a fallback document.\n\nIt has two paragraphs.\n\nAnd a third one for testing."

    # Task 2: Compare two strategies on the same document
    fixed_chunks = fixed_size_chunking(document, chunk_size=200, overlap=50)
    para_chunks = paragraph_chunking(document)
    
    # Task 3: Report chunk stats
    fixed_stats = get_stats(fixed_chunks)
    para_stats = get_stats(para_chunks)
    
    print("=== Chunking Strategy Comparison ===")
    print(f"1. Fixed-Size Chunking (Size: 200, Overlap: 50)")
    print(f"   Count: {fixed_stats['count']}")
    print(f"   Average Size: {fixed_stats['avg_size']} characters\n")
    
    print(f"2. Paragraph Chunking")
    print(f"   Count: {para_stats['count']}")
    print(f"   Average Size: {para_stats['avg_size']} characters\n")
    
    # Save sample output for Task 5
    sample_output = {
        "fixed_size_strategy": {
            "stats": fixed_stats,
            "sample_chunks": fixed_chunks[:3] # saving first 3 to inspect boundaries
        },
        "paragraph_strategy": {
            "stats": para_stats,
            "sample_chunks": para_chunks[:3]
        }
    }
    
    with open("sample_chunks.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)
        
    print("Sample chunks saved to 'sample_chunks.json' for review.")

if __name__ == "__main__":
    main()
