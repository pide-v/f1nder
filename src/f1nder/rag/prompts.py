core_meaning_extraction = """ 
Rephrase thw following question in a clear and concise query.
Remove all ambiguities and not relevant context.
Do not alter the original meaning of the query.
Here is the query to be rephrased: 
"""

rag_system_prompt = """
You are a helpful assistant that answers questions using the provided document as context.
Your answer must be based only on the information in the document.
If the document does not contain enough information to answer the question, say that you do not know.

Document: {document}

Question: {question}

Answer:
"""

