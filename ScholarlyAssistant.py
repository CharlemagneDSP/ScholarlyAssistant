import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("pdf_vector_store", embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=GPT4All(model="mistral-7b-instruct-v0.1.Q5_K_M", device="gpu"),
    retriever=db.as_retriever(),
    return_source_documents=True
)

def ask_ai(question):
    response = qa_chain.invoke({"query": question})
    
    # Debug: Print the retrieved documents
    #for doc in response["source_documents"]:
        #print(f"Retrieved Text (Length: {len(doc.page_content)}):\n{doc.page_content[:500]}...\n")
    answer = response["result"]
    return answer

print("Question: ")
question = input()
answer = ask_ai(question)
print("Answer:")
print(answer)