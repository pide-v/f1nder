# f1nder

Per usare dataset/build_collection.py e successivamente build_index.py scaricate 
i tre dataset del paper 

**build_collection.py** è il codice del prof messo in un unico file: prende train, test, development e li unisce in un'unica collezione di documenti.
gli oggetti nella collezione sono fatti così: 

{
    "para_id": "New_Hampshire_18070804_1",
    "context": "Testo pulito",
    "raw_ocr": "Testo OCR",
    "publication_date": "1807-08-04"
  }

**build_index.py** costruisce un inverted index a partire dai documenti della collezione. Come metadati vengono salvati "docno" e "publication_date"
