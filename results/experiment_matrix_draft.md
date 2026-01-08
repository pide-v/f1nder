### Experiment Matrix (draft)

| ExpID | Dataset Rev | Index | Doc Text | Retrieval Pipeline | Query Proc | k (retrieve) | RAG | Notes | P@1 | P@5 | P@10 | R@5 | R@10 | nDCG@5 | nDCG@10 | MAP |
|------:|-------------|------:|----------|--------------------|------------|-------------:|-----|-------|-----|-----|------|-----|------|--------|---------|-----|
| E00   | main        | clean | context  | BM25               | none       | 100          | off | sparse baseline |     |     |      |     |      |        |         |     |
| E01   | main        | ocr   | raw_ocr  | BM25               | none       | 100          | off | OCR impact      |     |     |      |     |      |        |         |     |
| E02   | main        | clean | context  | BM25 + RM3         | none       | 100          | off | pseudo RF       |     |     |      |     |      |        |         |     |
| E03   | main        | clean | fields   | BM25F (NER fields) | spaCy NER   | 100          | off | E2 retrieval     |     |     |      |     |      |        |         |     |
| E04   | main        | clean | context  | BM25 -> bi-enc rerank | none    | 100          | off | E1 rerank        |     |     |      |     |      |        |         |     |
| E05   | main        | clean | context  | BM25               | none       | 20           | on  | RAG baseline     |     |     |      |     |      |        |         |     |
| E06   | main        | clean | fields   | BM25F (NER fields) | spaCy NER   | 20           | on  | RAG w/ E2        |     |     |      |     |      |        |         |     |