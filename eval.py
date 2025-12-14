import os
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from app.services.rag import build_rag_graph
from app.core.llm import get_llm, get_embeddings

test_questions = [
    "What is the main purpose of this application?",
    "How does the ingestion pipeline work?",
]

test_ground_truths = [
    ["The main purpose is to demonstrate a RAG application with HyDe and Weaviate."],
    ["It loads documents, splits them, generates embeddings, and stores them in Weaviate."],
]

async def run_eval():
    print("Running Evaluation (Standard vs HyDe)...")
    
    app_graph = build_rag_graph()
    llm = get_llm()
    embeddings = get_embeddings()
    
    modes = [("Standard", False), ("HyDe", True)]
    final_results = {}

    for mode_name, use_hyde in modes:
        print(f"\nEvaluating mode: {mode_name}")
        answers = []
        contexts = []
        
        for q in test_questions:
            inputs = {
                "question": q,
                "use_hyde": use_hyde,
                "context": [],
                "answer": ""
            }
            result = app_graph.invoke(inputs)
            answers.append(result["answer"])
            contexts.append([d.page_content for d in result["context"]])
            
        data = {
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": test_ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        results = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=llm,
            embeddings=embeddings
        )
        final_results[mode_name] = results
        print(f"Results for {mode_name}: {results}")

    print("\n--- Comparative Analysis ---")
    for mode, metrics in final_results.items():
        print(f"\nMode: {mode}")
        for m, v in metrics.items():
            print(f"  {m}: {v}")
    
    return final_results

if __name__ == "__main__":
    asyncio.run(run_eval())
