import json
from pathlib import Path
from src.document_loader import DocumentLoader
from chunking import paragraph_chunking_with_offsets
from src.chunk_tagger import ChunkTagger, extract_sections_and_metadata

def run_full_pipeline():
    data_dir = Path("data")
    loader = DocumentLoader(verbose=False)
    
    # Task 1: Load all documents
    print("Loading documents...")
    documents, load_stats = loader.load_corpus(data_dir, recursive=True)
    
    all_chunks = []
    failed_files = load_stats['skipped_details']
    
    tagger = ChunkTagger()
    
    # Task 1: Clean, chunk, tag
    print("Processing and chunking documents...")
    for doc in documents:
        text = doc['text']
        source = doc['source']
        file_type = doc['file_type']
        
        doc_title, effective_date, section_map = extract_sections_and_metadata(
            text, file_type=file_type, source_name=source
        )
        
        para_tuples = paragraph_chunking_with_offsets(text)
        
        para_tagged = tagger.tag_chunks_from_tuples(
            source=source,
            chunk_tuples=para_tuples,
            file_type=file_type,
            doc_title=doc_title,
            section_map=section_map,
            effective_date=effective_date,
        )
        
        all_chunks.extend(para_tagged)
        
    # Task 2: Report Ingestion Summary
    total_source_files = load_stats['total_scanned']
    successful_docs = load_stats['total_successful']
    skipped_docs = load_stats['total_skipped']
    total_chunks = len(all_chunks)
    
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Total Source Documents: {total_source_files}")
    print(f"Successfully Ingested:  {successful_docs}")
    print(f"Skipped / Failed:       {skipped_docs}")
    print(f"Total Chunks Created:   {total_chunks}")
    
    if skipped_docs > 0:
        print("\nSkipped Files:")
        for skip in failed_files:
            print(f"  - {skip['source']}: {skip['reason']}")
            
    # Task 3: Validate Completeness
    print("\n" + "="*50)
    print("VALIDATION")
    print("="*50)
    reconciled = (successful_docs + skipped_docs) == total_source_files
    print(f"Source docs ({total_source_files}) == Success ({successful_docs}) + Skipped ({skipped_docs}) -> {reconciled}")
    if not reconciled:
        print("WARNING: Document count mismatch. Some files were dropped silently!")
    else:
        print("SUCCESS: All documents accounted for.")
        
    # Task 4: Inspect Sample Chunks
    print("\n" + "="*50)
    print("SAMPLE CHUNKS")
    print("="*50)
    
    sample_output = {
        "summary": {
            "total_source_files": total_source_files,
            "successfully_ingested": successful_docs,
            "skipped_docs": skipped_docs,
            "total_chunks_created": total_chunks,
            "validation_passed": reconciled
        },
        "sample_chunks": all_chunks[:3]
    }
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    with open("outputs/ingestion_summary.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)
        
    for i, chunk in enumerate(all_chunks[:2]):
        print(f"\n--- Sample Chunk {i+1} ---")
        meta = chunk.get("metadata", {})
        print(f"Source ID: {meta.get('source')}")
        print(f"Chunk Index: {meta.get('chunk_index')}")
        print(f"Char Span: {meta.get('char_start')} - {meta.get('char_end')}")
        print(f"Text Preview: {chunk.get('text', '')[:100]}...")
        
    print("\nFull ingestion summary and sample chunks saved to 'outputs/ingestion_summary.json'")

if __name__ == "__main__":
    run_full_pipeline()
