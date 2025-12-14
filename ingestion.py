
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from app.core.llm import get_embeddings
from app.core.db import get_weaviate_client

def ingest_document(file_path: str, collection_name: str = "LangChain_Collection"):
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    embeddings = get_embeddings()
    client = get_weaviate_client()

    vectorstore = Weaviate(
    client=client,
    index_name=collection_name,
    embedding=embeddings,
    by_text=False
)
    vectorstore.add_documents(splits)
    return {
        "message": f"Ingested {len(splits)} chunks into Weaviate.",
        "chunks": len(splits)
    }
