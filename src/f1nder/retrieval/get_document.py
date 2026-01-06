import json


#abbastanza inefficiente ma non credo ci interessi particolarmente migliorarla
#recupera il documento intero dalla collezione
def get_document_by_docno(docno, path):
	with open(path) as f:
		data = json.load(f)
	return [doc for doc in data if doc["para_id"] == docno]
