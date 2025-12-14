#!/usr/bin/env python3
"""
Demo: How to use the Medical RAG Evaluation Framework
Demonstrates evaluating multiple models and comparing results
"""

from evaluation_framework import MedicalRAGEvaluator, TestQuestion
import pandas as pd

def demo_single_model_evaluation():
    """Demo: Evaluate a single model"""
    print("🔬 DEMO 1: Single Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator for our baseline system
    evaluator = MedicalRAGEvaluator(models=["baseline"])
    
    # Load test dataset
    evaluator.load_test_dataset()
    print(f"📋 Loaded {len(evaluator.test_dataset)} test questions")
    
    # Evaluate the baseline model on first 3 questions (for demo speed)
    evaluator.test_dataset = evaluator.test_dataset[:3]
    print(f"🏃‍♂️ Running evaluation on {len(evaluator.test_dataset)} questions for demo...")
    
    results = evaluator.evaluate_model("baseline")
    
    # Show results
    comparison_df = evaluator.compare_models()
    print("\\n📊 RESULTS:")
    print(comparison_df.round(3))
    
    return evaluator

def demo_metrics_explanation():
    """Demo: Explain what each metric means"""
    print("\\n📚 DEMO 2: Understanding Evaluation Metrics")
    print("=" * 60)
    
    print("""
📈 RETRIEVAL METRICS (How well we find relevant documents):
   • Recall@5: Did we retrieve documents containing the answer? (Higher = Better)
   • Precision@5: Were the top-5 documents relevant? (Higher = Better) 
   • MRR: How high was the first relevant document ranked? (Higher = Better)
   • NDCG: Overall ranking quality (Higher = Better)

📝 GENERATION METRICS (How well we generate answers):
   • ROUGE-L: Text similarity to reference answer (Higher = Better)
   • BERTScore: Semantic similarity (Higher = Better)
   • Exact Match: Did we get exact numbers right? (dosages, etc.)
   • Medical Accuracy: How many medical facts were correct? (Higher = Better)

⚡ SPEED METRICS (How fast is the system):
   • Retrieval Latency: Time to find documents (Lower = Better)
   • Generation Latency: Time to generate answer (Lower = Better)
   • Total Latency: End-to-end response time (Lower = Better)

🛡️ SAFETY METRICS (How safe are the answers):
   • Has Disclaimer: Does answer include medical warnings? (Higher = Better)
   • Cites Sources: Does answer reference source documents? (Higher = Better)
   • Hallucination Rate: % of answers with false info (Lower = Better)
   """)

def demo_custom_test_questions():
    """Demo: Adding custom test questions"""
    print("\\n🧪 DEMO 3: Adding Custom Test Questions")
    print("=" * 50)
    
    # Create a custom test question
    custom_question = TestQuestion(
        id="CUSTOM001",
        question="Liều aspirin an toàn cho trẻ 5 tuổi?",
        category="contraindication", 
        ground_truth={
            "safe": False,
            "reason": "Aspirin chống chỉ định ở trẻ em dưới 12 tuổi",
            "answer": "KHÔNG an toàn. Aspirin không được khuyến nghị cho trẻ dưới 12 tuổi do nguy cơ hội chứng Reye."
        },
        expected_sources=["aspirin", "reye syndrome"],
        difficulty="medium"
    )
    
    print(f"📝 Custom Question: {custom_question.question}")
    print(f"   Category: {custom_question.category}")
    print(f"   Expected Answer: {custom_question.ground_truth['answer']}")
    
    # Evaluate single question
    evaluator = MedicalRAGEvaluator()
    try:
        result = evaluator.evaluate_single_question(custom_question, "baseline")
        print(f"\\n✅ Evaluation Results:")
        print(f"   - Medical Accuracy: {result.medical_accuracy:.3f}")
        print(f"   - ROUGE-L Score: {result.rouge_l:.3f}")
        print(f"   - Response Time: {result.total_latency:.2f}s")
        print(f"   - Has Safety Warning: {result.has_disclaimer}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def demo_performance_analysis():
    """Demo: Analyze performance by category"""
    print("\\n📊 DEMO 4: Performance Analysis by Category")  
    print("=" * 60)
    
    # Sample data showing different performance by question type
    sample_results = {
        'symptom': {'accuracy': 0.25, 'latency': 14.5, 'count': 5},
        'dosage': {'accuracy': 0.67, 'latency': 13.8, 'count': 3}, 
        'procedure': {'accuracy': 0.50, 'latency': 13.9, 'count': 2},
        'contraindication': {'accuracy': 0.75, 'latency': 14.2, 'count': 2}
    }
    
    print("📈 Performance by Medical Question Category:")
    print("-" * 60)
    for category, stats in sample_results.items():
        print(f"   {category.upper():<15} | Accuracy: {stats['accuracy']:.2f} | "
              f"Latency: {stats['latency']:.1f}s | Questions: {stats['count']}")
    
    print("\\n💡 INSIGHTS:")
    best_accuracy = max(sample_results.items(), key=lambda x: x[1]['accuracy'])
    worst_accuracy = min(sample_results.items(), key=lambda x: x[1]['accuracy'])
    fastest = min(sample_results.items(), key=lambda x: x[1]['latency'])
    
    print(f"   • Best Performance: {best_accuracy[0]} questions ({best_accuracy[1]['accuracy']:.2f} accuracy)")
    print(f"   • Needs Improvement: {worst_accuracy[0]} questions ({worst_accuracy[1]['accuracy']:.2f} accuracy)")
    print(f"   • Fastest Category: {fastest[0]} questions ({fastest[1]['latency']:.1f}s)")
    
    print("\\n🎯 RECOMMENDATIONS:")
    print("   • Dosage questions: Good accuracy - maintain current approach")
    print("   • Symptom questions: Low accuracy - improve medical terminology matching") 
    print("   • All categories: Latency >10s - optimize for clinical use")

def demo_benchmark_comparison():
    """Demo: Compare against benchmarks"""
    print("\\n🏆 DEMO 5: Benchmark Comparison")
    print("=" * 50)
    
    # Simulated comparison with other medical RAG systems
    benchmark_data = {
        'System': ['Our Baseline', 'Commercial System A', 'Research System B', 'Target Goals'],
        'Medical Accuracy': [0.36, 0.72, 0.68, 0.80],
        'Avg Latency (s)': [14.4, 3.2, 8.1, 3.0],
        'Safety Score': [0.45, 0.89, 0.76, 0.90],
        'Coverage': ['12 questions', '100 questions', '50 questions', '1000+ questions']
    }
    
    df = pd.DataFrame(benchmark_data)
    print("📊 BENCHMARK COMPARISON:")
    print(df.to_string(index=False))
    
    print("\\n📝 ANALYSIS:")
    print("   ✅ STRENGTHS:")
    print("      - System is functional across multiple medical domains")
    print("      - Retrieval pipeline working effectively")
    print("   ⚠️  AREAS FOR IMPROVEMENT:")
    print("      - Medical accuracy significantly below commercial systems")
    print("      - Response latency 4-5x slower than target") 
    print("      - Safety mechanisms need enhancement")
    print("   🎯 NEXT STEPS:")
    print("      - Implement medical fact verification")
    print("      - Optimize inference pipeline for speed")
    print("      - Expand test coverage to 100+ questions")

def main():
    """Run all evaluation demos"""
    print("🚀 MEDICAL RAG EVALUATION FRAMEWORK - DEMO")
    print("=" * 70)
    print("This demo shows how to use the evaluation framework to benchmark")
    print("medical RAG systems with scientific rigor.\\n")
    
    try:
        # Demo 1: Basic evaluation
        demo_single_model_evaluation()
        
        # Demo 2: Metrics explanation  
        demo_metrics_explanation()
        
        # Demo 3: Custom questions
        demo_custom_test_questions()
        
        # Demo 4: Performance analysis
        demo_performance_analysis()
        
        # Demo 5: Benchmark comparison
        demo_benchmark_comparison()
        
        print("\\n🎉 DEMO COMPLETE!")
        print("=" * 70)
        print("✅ You now know how to:")
        print("   • Evaluate medical RAG systems scientifically")
        print("   • Compare multiple models objectively") 
        print("   • Analyze performance by medical question type")
        print("   • Benchmark against industry standards")
        print("   • Identify specific areas for improvement")
        print("\\n🔬 Ready to benchmark your medical AI system!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()