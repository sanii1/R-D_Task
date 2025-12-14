import os
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.ingestion import ingest_document
from app.services.rag import build_rag_graph
from app.models.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Agentic RAG App")

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = ingest_document(temp_file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    app_graph = build_rag_graph()
    
    inputs = {
        "question": request.query,
        "use_hyde": request.use_hyde,
        "context": [],
        "answer": ""
    }
    
    result = app_graph.invoke(inputs)
    
    processing_time = time.time() - start_time
    
    return ChatResponse(
        answer=result["answer"],
        context=[doc.page_content for doc in result["context"]],
        processing_time=processing_time
    )

@app.get("/")
def read_root():
    return {"message": "RAG API is running"}
