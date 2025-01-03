import os
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def load_document(uploaded_file):
    name, extension = os.path.splitext(uploaded_file)

    if extension == ".pdf":
        loader = PyPDFLoader(uploaded_file)
    elif extension == ".txt":
        loader = TextLoader(uploaded_file)
    elif extension == ".docx":
        loader = Docx2txtLoader(uploaded_file)
    else:
        raise ValueError("Uploaded file is not supported")

    data = loader.load()
    return data

def chunk_data(data, chunk_size=512, overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# def create_embedding(chunks):
#     embedding = OllamaEmbeddings(model="llama3.1")
#     vector_store = Chroma.from_documents(chunks, embedding)
#     return vector_store


def create_embedding(chunks):
    # Initialize Chroma client and embedding model
    client = chromadb.Client()
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models

    # Create or get the collection
    collection = client.get_collection(name="chroma_docs")

    # Generate embeddings for the chunks (documents)
    embeddings = model.encode(chunks)  # This returns a list of embeddings for the chunks

    # Add documents and embeddings to Chroma collection
    collection.add(
        documents=chunks,  # The actual text data
        embeddings=embeddings,  # The embeddings
        ids=[str(i) for i in range(len(chunks))]  # Unique IDs for each document
    )

    return collection
def ask_answer(vector_store, q, k=3):
    llm = ChatOllama(model="llama3.1", temperature=0.7)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.invoke(q)
    return answer["result"]
