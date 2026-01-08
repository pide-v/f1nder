# baseline experiment tf-idf

import json
import os
from pathlib import Path
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()

# load index
INDEX_PATH = Path("data/index/index").resolve()
index = pt.IndexFactory.of(str(INDEX_PATH))

# load queries
with open("data/test_queries.json", encoding="utf-8") as f:
    test_queries = pd.DataFrame(json.load(f))

test_queries = test_queries.rename(columns={"query_id": "qid", "question": "query"})

# load qrels
with open("data/test_qurels.json", encoding="utf-8") as f:
    qrels = pd.DataFrame(json.load(f))

qrels = qrels.drop(columns=["iteration"])
qrels = qrels.rename(columns={"query_id": "qid", "para_id": "docno", "relevance": "label"})

# check qrels and test_queries
print(f"qrels:\n{qrels.head()}\ntest_queries:\n{test_queries.head()}")

# prepare models for retrieval
tfidf = pt.terrier.Retriever(
    index, 
    wmodel="TF_IDF"
)

# export run
res = tfidf.transform(test_queries)
res.to_csv("results/runs/tfidf_run.tsv", sep="\t", index=False, header=False)
print("Run salvata in results/runs/tfidf_run.tsv")

# Oppure volendo fare più esperimenti nello stesso file qui una cosa tipo:
#models = {
#    "tfidf": pt.terrier.Retriever(index, wmodel="TF_IDF"),
#    "bm25": pt.terrier.Retriever(index, wmodel="BM25"),
#    "dph": pt.terrier.Retriever(index, wmodel="DPH")
#}

#for name, model in models.items():
#    print(f"Running {name}...")
#    res = model.transform(test_queries)
#    res.to_csv(f"results/runs/"{name}_run.tsv", sep="\t", index=False, header=False)
#    print(f"✅ {name} run saved at results/runs/{name}_run.tsv")