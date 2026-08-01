# Multiplex Hypergraph RAG

A production-oriented architecture that merges **multiplex networks** (layered categorical relations) with **hypergraphs** (many-to-many group containment) to bound graph explosion and token leakage in retrieval-augmented generation.

## Design Document

The complete specification lives in **[DESIGN.md](DESIGN.md)**. It covers:

1. Motivation and failure modes of prior GraphRAG systems  
2. Formal mathematical model \(\mathcal{MH}=(\mathcal{L},\mathcal{V},\mathcal{E},\mathcal{C})\)  
3. Layer 1 (Structural Narrative), Layer 2 (Ontological Entity), Layer 3 (Latent Topological)  
4. Vertical links and soft membership weights  
5. Synchronous ingestion with inline entity resolution  
6. Budget-bounded retrieval (depth, beam, token knapsack)  
7. Formal explosion and token-leakage bounds  
8. Tech stack for 32 GB RAM / 6 GB RTX 4050 (FalkorDB, TEI + BGE-M3, Azure OpenAI GPT-4.1)  
9. Production onboarding and testing roadmap  

## Status

Design phase. Implementation follows the build phases in [DESIGN.md §11](DESIGN.md#11-production-roadmap).

## License

Proprietary / TBD.
