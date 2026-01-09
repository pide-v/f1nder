### BASELINE EXPERIMENTS ###

import json
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

#load index
index = pt.IndexFactory.of("/mnt/c/Users/lore9/Desktop/f1nder/data/index/index")

#load qrels
with open("dataset/test_qrels.json") as f:
    data = json.load(f)

qrels = pd.DataFrame(data)

#load test queries
with open("dataset/test_queries.json") as f:
    data = json.load(f)

test_queries = pd.DataFrame(data)

#Prepare the dataframes for pt.Experiment pipeline
#NB sul colab del prof c'è scritto di tenere solo i primi 10k, infatti 
#in dataset/test_queries.json ci sono 10k entrate. Le qrels invece sono costruite
#per tutti i documenti. Quando si passano qrels e queries a pt.Experiment 
#vengono valutate solo le 10k entrate presenti in test_queries
test_queries = test_queries.rename(columns={"query_id":"qid", "question":"query"})
qrels = qrels.drop(columns=["iteration"])
qrels = qrels.rename(columns={"query_id":"qid", "para_id":"docno", "relevance":"label"})


#Check qrels and test_queries
print(f"qrels:\n{qrels.head()}\ntest_queries:\n{test_queries.head()}")

#prepare models for retrieval
tfidf = pt.BatchRetrieve(
    index,
    wmodel="TF_IDF"
)

res = pt.Experiment(
    [tfidf],
    test_queries,
    qrels,
    eval_metrics=[P@1]
)

print(res)