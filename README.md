# f1nder

Per usare `dataset/build_collection.py` e successivamente `build_index.py` scaricate 
i tre dataset del paper 

`build_collection.py` è il codice del prof messo in un unico file: prende `train`, `test`, `development` e li unisce in un'unica collezione di documenti.
gli oggetti nella collezione sono fatti così: 
```
{
  "para_id": "New_Hampshire_18070804_1",
  "context": "Testo pulito",
  "raw_ocr": "Testo OCR",
  "publication_date": "1807-08-04"
}
```
Inoltre genera il file con le qrels:   
```
{
  "query_id": "test_1",
  "iteration": 0,
  "para_id": "New_Hampshire_18030125_16",
  "relevance": 1
}
```
uno con di test delle queries:
```
{
  "query_id": "test_1",
  "question": "How many lots did Thomas Peirce have"
}
```
e uno con le risposte corrette per le query:
```
{
  "query_id": "test_1",
  "iteration": 0,
  "para_id": "New_Hampshire_18030125_16",
  "relevance": 1,
  "answer": "183",
  "org_answer": "183"
}
```
`build_index.py` costruisce un inverted index a partire dai documenti della collezione. Come metadati vengono salvati `"docno"` e `"publication_date"`
