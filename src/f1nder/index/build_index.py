import json
import pandas as pd
import os
import shutil
import pyterrier as pt
from argparse import ArgumentParser


def build_index(document_collection_file, index_path):
    #read document_collection
    with open(document_collection_file) as f:
        data = json.load(f)

    #Building a dataframe containing the whole collection
    df = pd.DataFrame(data)
    #rename columns
    df = df.rename(columns={"para_id":"docno", "context":"text"})
    #delete raw_ocr data since we are not using it
    df = df.drop(columns=["raw_ocr"])

    print(df.head().T)
    print(f"Number of documents(compare with indexed documents): {len(df)}")

    #set metadata to be indexed
    meta_cols = {
        "docno" : 30,
        "publication_date" : 15
    }

    #Check if index path already exists
    if os.path.exists(index_path):
        print("W: index path already present. Removing index")
        shutil.rmtree(index_path)
    os.makedirs(index_path, exist_ok=True)

    indexer = pt.IterDictIndexer(
        index_path,
        meta=meta_cols,
        text_attrs=["text"],
        pretokenised=False
    )
    print("I: Creating index")
    indexref = indexer.index(df.to_dict(orient="records"))

    # Open the index to ensure it is valid
    index = pt.IndexFactory.of(indexref)

    # Print a simple summary
    print("Index location:", index_path)
    print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())

    # Retrieve collection statistics
    # to make sure everything is fine and indexed
    stats = index.getCollectionStatistics()

    print("Terrier Collection Statistics")
    print("--------------------------------")
    print(f"Indexed documents:        {stats.getNumberOfDocuments()}")
    print(f"Unique terms (vocabulary): {stats.getNumberOfUniqueTerms()}")
    print(f"Total tokens:             {stats.getNumberOfTokens()}")
    print(f"Average document length:  {stats.getAverageDocumentLength():.2f}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--document_collection_file")
    ap.add_argument("--index_path")
    args = ap.parse_args()

    build_index(args.document_collection_file, args.index_path)