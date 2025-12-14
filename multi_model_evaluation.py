#!/usr/bin/env python3
"""
Multi-Model RAG Evaluation: Test 3 models with Ollama
Compare Qwen2.5:3B, Llama3.2:3B, and Phi3.5:3.8B
"""

from evaluation_framework import MedicalRAGEvaluator, TestQuestion
import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from query_data import detect_query_type, get_optimal_k, get_prompt_template, format_response_by_type, rerank_results
import time
import json
import os

# Force CPU-only mode to avoid CUDA memory issues
os.environ["OLLAMA_NUM_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CHROMA_PATH = "chroma"

# Model configurations
MODELS = {
    "qwen2.5:3b": {
        "name": "Qwen2.5-3B-Instruct",
        "temperature": 0.1,
        "top_p": 0.9
    },
    "llama3.2:3b": {
        "name": "Llama-3.2-3B-Instruct", 
        "temperature": 0.1,
        "top_p": 0.9
    },
    "phi3.5:3.8b": {
        "name": "Phi-3.5-Mini",
        "temperature": 0.1,
        "top_p": 0.9
    }
}

def query_rag_with_model(query_text: str, model_name: str, show_sources: bool = False) -> str:
    """
    Query RAG system using specific Ollama model
    """
    # Detect query type and get optimal k
    query_type = detect_query_type(query_text)
    optimal_k = get_optimal_k(query_type)
    
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    initial_results = db.similarity_search_with_score(query_text, k=optimal_k + 2)

    if len(initial_results) == 0:
        return "Unable to find matching results."

    # Apply semantic re-ranking  
    reranked_results = rerank_results(initial_results, query_text)
    top_results = reranked_results[:optimal_k]
    
    # Build context
    context_chunks = []
    for doc, enhanced_score, confidence, reasoning in top_results:
        context_chunks.append(doc.page_content)
    
    context_text = "\\n\\n---\\n\\n".join(context_chunks)
    
    # Use adaptive prompt template
    prompt_text = get_prompt_template(query_type, query_text, context_text)
    
    # Initialize Ollama LLM (CPU-only mode set via environment variables)
    model_config = MODELS[model_name]
    llm = OllamaLLM(
        model=model_name,
        temperature=model_config["temperature"],
        top_p=model_config["top_p"]
    )
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    
    # Generate response
    try:
        start_time = time.time()
        response_text = llm.invoke(prompt_template.format())
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Apply type-specific formatting
        formatted_response = format_response_by_type(response_text, query_type)
        
        if show_sources:
            print(f"\\n📊 Model: {model_config['name']} | Query Type: {query_type.upper()} | Time: {generation_time:.2f}s")
            for i, (doc, enhanced_score, confidence, reasoning) in enumerate(top_results):
                source_id = doc.metadata.get("id", "Unknown")
                print(f"  {i+1}. Source: {source_id} | Confidence: {confidence:.2f}")
        
        return formatted_response
        
    except Exception as e:
        return f"Error with model {model_name}: {str(e)}"

class MultiModelEvaluator(MedicalRAGEvaluator):
    """Extended evaluator with Ollama model support"""
    
    def __init__(self, models: list = None):
        if models is None:
            models = list(MODELS.keys())
        super().__init__(models)
        
    def evaluate_single_question_with_ollama(self, question: TestQuestion, model_name: str) -> dict:
        """Evaluate single question using Ollama model"""
        
        print(f"Evaluating {question.id} with {MODELS[model_name]['name']}...")
        
        # Measure retrieval time
        retrieval_start = time.time()
        
        # Get retrieved documents
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        retrieval_results = db.similarity_search_with_score(question.question, k=5)
        retrieved_docs = [doc.page_content for doc, score in retrieval_results]
        
        retrieval_end = time.time()
        retrieval_latency = retrieval_end - retrieval_start
        
        # Measure generation time
        generation_start = time.time()
        
        # Generate answer using Ollama model
        generated_answer = query_rag_with_model(question.question, model_name, show_sources=False)
        
        generation_end = time.time()
        generation_latency = generation_end - generation_start
        
        total_latency = retrieval_latency + generation_latency
        
        # Compute metrics
        retrieval_metrics = self.compute_retrieval_metrics(
            retrieved_docs, question.expected_sources, 
            question.ground_truth.get("answer", "")
        )
        
        generation_metrics = self.compute_generation_metrics(
            generated_answer, question.ground_truth
        )
        
        safety_metrics = self.compute_safety_metrics(generated_answer)
        
        return {
            'question_id': question.id,
            'model_name': model_name,
            'model_display_name': MODELS[model_name]['name'],
            'category': question.category,
            'retrieval_latency': retrieval_latency,
            'generation_latency': generation_latency,
            'total_latency': total_latency,
            'generated_answer': generated_answer,
            'ground_truth_answer': question.ground_truth.get("answer", ""),
            **retrieval_metrics,
            **generation_metrics,
            **safety_metrics
        }

def test_single_query():
    """Test single query with all 3 models"""
    print("🔬 TESTING SINGLE QUERY WITH 3 MODELS")
    print("=" * 60)
    
    test_query = "Triệu chứng của viêm phổi ở trẻ em là gì?"
    
    print(f"Query: {test_query}\\n")
    
    for model_name, config in MODELS.items():
        print(f"\\n🤖 Model: {config['name']}")
        print("-" * 40)
        
        start_time = time.time()
        response = query_rag_with_model(test_query, model_name, show_sources=True)
        end_time = time.time()
        
        print(f"Response ({end_time - start_time:.2f}s):")
        print(response[:300] + "..." if len(response) > 300 else response)

def run_comparative_evaluation():
    """Run comprehensive evaluation on all 3 models"""
    print("\\n🏆 COMPARATIVE EVALUATION: 3 MODELS")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = MultiModelEvaluator()
    
    # Load test dataset (limit to 5 questions for demo)
    evaluator.load_test_dataset()
    test_questions = evaluator.test_dataset[:5]  # First 5 questions for quick comparison
    
    print(f"Testing {len(test_questions)} questions across {len(MODELS)} models...\\n")
    
    all_results = []
    
    for model_name in MODELS.keys():
        print(f"\\n🔬 Evaluating {MODELS[model_name]['name']}...")
        print("-" * 50)
        
        model_results = []
        
        for question in test_questions:
            try:
                result = evaluator.evaluate_single_question_with_ollama(question, model_name)
                model_results.append(result)
                all_results.append(result)
                
                print(f"  ✅ {question.id}: {result['total_latency']:.2f}s | Accuracy: {result['medical_accuracy']:.2f}")
                
            except Exception as e:
                print(f"  ❌ {question.id}: Failed - {e}")
                continue
    
    # Generate comparison report
    print("\\n📊 GENERATING COMPARISON REPORT...")
    
    # Aggregate results by model
    model_stats = {}
    for result in all_results:
        model = result['model_name']
        display_name = result['model_display_name']
        
        if model not in model_stats:
            model_stats[model] = {
                'display_name': display_name,
                'questions': 0,
                'total_accuracy': 0,
                'total_latency': 0,
                'total_rouge': 0,
                'exact_matches': 0
            }
        
        stats = model_stats[model]
        stats['questions'] += 1
        stats['total_accuracy'] += result['medical_accuracy']
        stats['total_latency'] += result['total_latency']
        stats['total_rouge'] += result['rouge_l']
        stats['exact_matches'] += int(result['exact_match'])
    
    # Print comparison table
    print("\\n🏆 MODEL COMPARISON RESULTS:")
    print("=" * 80)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Latency':<10} | {'ROUGE-L':<10} | {'Questions'}")
    print("-" * 80)
    
    for model, stats in model_stats.items():
        n = stats['questions']
        avg_accuracy = stats['total_accuracy'] / n
        avg_latency = stats['total_latency'] / n
        avg_rouge = stats['total_rouge'] / n
        
        print(f"{stats['display_name']:<25} | {avg_accuracy:<10.3f} | {avg_latency:<10.2f} | {avg_rouge:<10.3f} | {n}")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"multi_model_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n💾 Detailed results saved to: {results_file}")
    
    # Find best model
    best_accuracy = max(model_stats.items(), key=lambda x: x[1]['total_accuracy'] / x[1]['questions'])
    fastest_model = min(model_stats.items(), key=lambda x: x[1]['total_latency'] / x[1]['questions'])
    
    print("\\n🥇 WINNERS:")
    print(f"   Best Accuracy: {model_stats[best_accuracy[0]]['display_name']} ({best_accuracy[1]['total_accuracy'] / best_accuracy[1]['questions']:.3f})")
    print(f"   Fastest: {model_stats[fastest_model[0]]['display_name']} ({fastest_model[1]['total_latency'] / fastest_model[1]['questions']:.2f}s)")

def main():
    parser = argparse.ArgumentParser(description="Multi-Model RAG Evaluation")
    parser.add_argument("--mode", choices=["single", "full"], default="single", 
                        help="Evaluation mode: single query or full evaluation")
    parser.add_argument("--query", type=str, 
                        help="Custom query for single mode")
    
    args = parser.parse_args()
    
    print("🚀 MULTI-MODEL RAG EVALUATION")
    print("Testing 3 models: Qwen2.5:3B, Llama3.2:3B, Phi3.5:3.8B")
    print("=" * 70)
    
    if args.mode == "single":
        if args.query:
            # Test custom query
            test_query = args.query
            print(f"\\n🔍 Custom Query: {test_query}")
            
            for model_name, config in MODELS.items():
                print(f"\\n🤖 {config['name']}:")
                start_time = time.time()
                response = query_rag_with_model(test_query, model_name)
                end_time = time.time()
                print(f"Time: {end_time - start_time:.2f}s")
                print(f"Response: {response[:200]}...")
        else:
            test_single_query()
    
    elif args.mode == "full":
        run_comparative_evaluation()
    
    print("\\n✅ EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()