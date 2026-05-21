# Company Documents Folder

Place your company documents here for ingestion into the GraphRAG knowledge base.

## Supported Formats

| Format | Notes |
|--------|-------|
| `.pdf` | Automatically parsed with pdfplumber |
| `.txt` | Plain text |
| `.md` / `.mdx` | Markdown (section-aware chunking) |
| `.json` | Auto-serialized to text |
| `.docx` | Loaded as plain text |

## Naming Convention → Auto-routing

The filename determines which Qdrant collection the document is ingested into:

| Filename contains | Collection | Purpose |
|---|---|---|
| `competitor*` | `competitor_intel` | Competitive intelligence |
| `regulation*`, `policy*`, `compliance*`, `hipaa*`, `gdpr*` | `regulatory_corpus` | Legal/regulatory docs |
| Anything else | `company_docs` | General company knowledge |

## Example Structure

```
company_docs/
  annual_report_2024.pdf          → company_docs
  competitor_medinsight_q4.pdf    → competitor_intel
  hipaa_compliance_guide.pdf      → regulatory_corpus
  product_roadmap.md              → company_docs
  gdpr_policy.txt                 → regulatory_corpus
  market_research_2025.pdf        → company_docs
```

## Ingesting Documents

```bash
# One-time bulk ingest
./start_rag.sh --ingest

# Auto-watch (add files to this folder and they ingest automatically)
./start_rag.sh --watch
```

Graph entities (companies, risks, regulations, competitors) are extracted
automatically from every document and stored in Neo4j.
