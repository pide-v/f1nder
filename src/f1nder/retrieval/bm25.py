### RETRIEVAL RUN WITH BM25 MODEL ###
import json
import pandas as pd
import pyterrier as pt
from argparse import ArgumentParser

def bm25(index_path, qrels_path, test_queries_path, output_path):
    #load index
    index = pt.IndexFactory.of(index_path)

    #load qrels
    with open(qrels_path) as f:
        data = json.load(f)

    qrels = pd.DataFrame(data)

    #load test queries
    with open(test_queries_path) as f:
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
    bm25 = pt.BatchRetrieve(
        index,
        wmodel="BM25"
    )

    results = bm25.transform(test_queries)
    output_path = output_path + "bm25.txt"
    pt.io.write_results(results, output_path)

if __name__ == "__main__":
    pt.init()
    ap = ArgumentParser()
    ap.add_argument("--qrels_path")
    ap.add_argument("--test_queries_path")
    ap.add_argument("--index_path")
    ap.add_argument("--output_path")
    args = ap.parse_args()

    bm25(
        index_path = args.index_path,
        qrels_path=args.qrels_path,
        test_queries_path=args.test_queries_path,
        output_path=args.output_path
    )