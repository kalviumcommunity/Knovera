# Simple script to generate valid text-extractable PDF without external heavy packages
def make_pdf(filename, title, lines):
    content_stream = "BT /F1 16 Tf 50 750 Td (" + title + ") Tj ET\n"
    y = 710
    content_stream += "BT /F1 11 Tf\n"
    content_stream += f"50 {y} Td\n"
    content_stream += "16 TL\n"
    for line in lines:
        clean_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream += f"({clean_line}) '\n"
    content_stream += "ET\n"
    
    stream_bytes = content_stream.encode('latin1')
    stream_len = len(stream_bytes)
    
    objects = []
    # 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # 3: Page
    objects.append("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
    # 4: Stream
    objects.append(f"4 0 obj\n<< /Length {stream_len} >>\nstream\n{content_stream}endstream\nendobj\n")
    # 5: Font
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    
    body = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(body.encode('latin1')))
        body += obj
        
    xref_offset = len(body.encode('latin1'))
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for o in offsets[1:]:
        body += f"{o:010d} 00000 n \n"
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    
    with open(filename, 'wb') as f:
        f.write(body.encode('latin1'))
    print(f"Created {filename}")

if __name__ == "__main__":
    make_pdf(
        "c:/Users/K Jayanth/OneDrive/Desktop/Mini-Projects/Knovera/Sampledata/enterprise_ai_guidelines.pdf",
        "KNOVERA ENTERPRISE AI USE GUIDELINES",
        [
            "Document ID: PDF-AI-2026",
            "Effective Date: February 2026",
            "",
            "1. Purpose and Scope",
            "This guideline governs the safe implementation of LLMs and RAG systems.",
            "All generative responses must be backed by verified vector retrieval.",
            "",
            "2. Safety Thresholds",
            "Queries with semantic relevance scores below the safety threshold must",
            "be gracefully refused to prevent ungrounded responses.",
            "",
            "3. Multi-Tenant Protection",
            "Each client workspace maintains distinct MongoDB vector collections.",
            "Data is encrypted at rest and in transit across all endpoints."
        ]
    )
