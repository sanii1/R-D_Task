from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.core.llm import get_llm, get_embeddings
from langchain_community.vectorstores import Weaviate
from app.core.db import get_weaviate_client

class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    use_hyde: bool

def retrieve(state: RagState):
    """
    Retrieve documents relevant to the question.
    If use_hyde is True, generate a hypothetical document first.
    """
    question = state["question"]
    use_hyde = state.get("use_hyde", False)
    llm = get_llm()
    embeddings = get_embeddings()
    
    vectorstore = Weaviate(
        client = get_weaviate_client(),
        index_name="LangChain_Collection", 
        embedding=embeddings,
        by_text=False
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    
    search_query = question
    
    if use_hyde:
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a scientific paper abstract or a detailed paragraph that answers the following question: {question}"
        )
        hyde_chain = hyde_prompt | llm
        hypothetical_doc = hyde_chain.invoke({"question": question})
        search_query = hypothetical_doc.content
        print(f"HyDe generated: {search_query[:100]}...")

    docs = retriever.get_relevant_documents(search_query)
    return {"context": docs}

def generate(state: RagState):
    """
    Generate answer using retrieved context.
    """
    question = state["question"]
    context = state["context"]
    if not context:
        return {
            "answer": "I could not find relevant information in the provided documents to answer this question."
        }
    
    llm = get_llm()
    
    template = """Answer the question based only on the following information below.
    If the answer is not present, say you do not know.

    context:
    {context}
    
    Question: 
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    context_str = "\n\n".join([d.page_content for d in context])
    
    response = chain.invoke({
        "question": question, 
        "context": context_str
        })
    
    return {"answer": response.content}

def build_rag_graph():
    workflow = StateGraph(RagState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
