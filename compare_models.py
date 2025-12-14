#!/usr/bin/env python3
"""
Model Response Analyzer: So sánh chất lượng câu trả lời của các models
"""

import re
import time
from multi_model_evaluation import query_rag_with_model, MODELS

def analyze_medical_response(response: str, query: str) -> dict:
    """Phân tích chất lượng câu trả lời y tế"""
    
    analysis = {
        'language_quality': 0,
        'medical_accuracy': 0,
        'completeness': 0,
        'structure': 0,
        'safety': 0,
        'total_score': 0,
        'details': {}
    }
    
    # 1. Language Quality (Vietnamese vs English) - 25 points
    vietnamese_words = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', response.lower()))
    english_words = len(re.findall(r'[a-zA-Z]+', response))
    
    if vietnamese_words > english_words * 0.5:
        analysis['language_quality'] = 25
        analysis['details']['language'] = "✅ Vietnamese dominant (good)"
    elif vietnamese_words > 0:
        analysis['language_quality'] = 15
        analysis['details']['language'] = "⚠️ Mixed Vietnamese/English"
    else:
        analysis['language_quality'] = 5
        analysis['details']['language'] = "❌ English only"
    
    # 2. Medical Accuracy - 30 points
    medical_terms = {
        'ho gà': ['ho gà', 'whooping cough', 'pertussis', 'bordetella', 'cơn ho', 'tiêm chủng'],
        'viêm phổi': ['viêm phổi', 'pneumonia', 'khó thở', 'sốt', 'ho', 'đau ngực'],
        'sốt': ['sốt', 'fever', 'nhiệt độ', '38°c', '39°c', 'hạ sốt'],
        'tiêu chảy': ['tiêu chảy', 'diarrhea', 'phân lỏng', 'mất nước', 'ors']
    }
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Identify medical condition from query
    condition = None
    for cond, terms in medical_terms.items():
        if any(term in query_lower for term in terms[:2]):
            condition = cond
            break
    
    if condition:
        relevant_terms = medical_terms[condition]
        found_terms = sum(1 for term in relevant_terms if term in response_lower)
        analysis['medical_accuracy'] = min(30, (found_terms / len(relevant_terms)) * 30)
        analysis['details']['medical_terms'] = f"Found {found_terms}/{len(relevant_terms)} relevant terms"
    else:
        analysis['medical_accuracy'] = 15
        analysis['details']['medical_terms'] = "General medical content"
    
    # 3. Completeness - 20 points
    response_length = len(response)
    if response_length > 200:
        analysis['completeness'] = 20
        analysis['details']['length'] = "✅ Comprehensive answer"
    elif response_length > 100:
        analysis['completeness'] = 15
        analysis['details']['length'] = "⚠️ Adequate length"
    else:
        analysis['completeness'] = 10
        analysis['details']['length'] = "❌ Too brief"
    
    # 4. Structure - 15 points
    structure_points = 0
    if '•' in response or '-' in response or '1.' in response:
        structure_points += 5
        analysis['details']['bullets'] = "✅ Has bullet points"
    
    if any(header in response for header in ['##', '**', 'Triệu chứng', 'Nguyên nhân']):
        structure_points += 5
        analysis['details']['headers'] = "✅ Has headers/formatting"
    
    if len(response.split('\n')) > 3:
        structure_points += 5
        analysis['details']['paragraphs'] = "✅ Multi-paragraph"
    
    analysis['structure'] = structure_points
    
    # 5. Safety - 10 points
    safety_indicators = ['tham khảo bác sĩ', 'cần khám', 'nghiêm trọng', 'cảnh báo', 'an toàn']
    safety_count = sum(1 for indicator in safety_indicators if indicator in response_lower)
    
    if safety_count >= 2:
        analysis['safety'] = 10
        analysis['details']['safety'] = "✅ Good safety warnings"
    elif safety_count >= 1:
        analysis['safety'] = 7
        analysis['details']['safety'] = "⚠️ Some safety mentions"
    else:
        analysis['safety'] = 3
        analysis['details']['safety'] = "❌ No safety warnings"
    
    # Calculate total score
    analysis['total_score'] = (
        analysis['language_quality'] + 
        analysis['medical_accuracy'] + 
        analysis['completeness'] + 
        analysis['structure'] + 
        analysis['safety']
    )
    
    return analysis

def compare_models_detailed(query: str):
    """So sánh chi tiết 3 models cho 1 query"""
    
    print(f"🔍 DETAILED ANALYSIS FOR: '{query}'")
    print("=" * 80)
    
    results = {}
    
    # Get responses from all models
    for model_name, config in MODELS.items():
        print(f"\n🤖 Testing {config['name']}...")
        
        start_time = time.time()
        response = query_rag_with_model(query, model_name, show_sources=False)
        end_time = time.time()
        
        # Analyze response quality
        analysis = analyze_medical_response(response, query)
        
        results[model_name] = {
            'name': config['name'],
            'response': response,
            'time': end_time - start_time,
            'analysis': analysis
        }
        
        print(f"   ⏱️ Time: {end_time - start_time:.1f}s")
        print(f"   📊 Score: {analysis['total_score']}/100")
    
    # Generate comparison report
    print("\n📊 DETAILED COMPARISON REPORT")
    print("=" * 80)
    
    # Sort by total score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['analysis']['total_score'], reverse=True)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        analysis = result['analysis']
        
        print(f"\n🏆 RANK {rank}: {result['name']}")
        print(f"   Overall Score: {analysis['total_score']}/100 ({analysis['total_score']}%)")
        print(f"   Response Time: {result['time']:.1f}s")
        print()
        
        # Detailed breakdown
        print("   📊 Score Breakdown:")
        print(f"      Language Quality: {analysis['language_quality']}/25 - {analysis['details'].get('language', '')}")
        print(f"      Medical Accuracy: {analysis['medical_accuracy']:.1f}/30 - {analysis['details'].get('medical_terms', '')}")
        print(f"      Completeness: {analysis['completeness']}/20 - {analysis['details'].get('length', '')}")
        print(f"      Structure: {analysis['structure']}/15 - {analysis['details'].get('bullets', '')} {analysis['details'].get('headers', '')}")
        print(f"      Safety: {analysis['safety']}/10 - {analysis['details'].get('safety', '')}")
        
        # Show response preview
        print(f"\n   📝 Response Preview:")
        preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
        print(f"      {preview}")
        
        print("-" * 60)
    
    # Winner announcement
    winner = sorted_results[0]
    print(f"\n🥇 WINNER: {winner[1]['name']}")
    print(f"   Score: {winner[1]['analysis']['total_score']}/100")
    print(f"   Best for: Vietnamese medical queries")
    
    return sorted_results

def quick_comparison(query: str):
    """So sánh nhanh chỉ về accuracy và speed"""
    
    print(f"⚡ QUICK COMPARISON: '{query}'")
    print("-" * 50)
    
    results = {}
    
    for model_name, config in MODELS.items():
        start_time = time.time()
        response = query_rag_with_model(query, model_name)
        end_time = time.time()
        
        # Quick analysis
        vietnamese_ratio = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', response.lower())) / max(len(response), 1)
        length = len(response)
        time_taken = end_time - start_time
        
        results[model_name] = {
            'name': config['name'],
            'vietnamese_ratio': vietnamese_ratio,
            'length': length,
            'time': time_taken,
            'response': response
        }
    
    # Quick ranking
    for model_name, result in results.items():
        print(f"{result['name']:<25} | "
              f"VI: {result['vietnamese_ratio']*100:.0f}% | "
              f"Length: {result['length']:<4} | "
              f"Time: {result['time']:.1f}s")
    
    # Find best
    best_vn = max(results.items(), key=lambda x: x[1]['vietnamese_ratio'])
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    
    print(f"\n🏆 Best Vietnamese: {best_vn[1]['name']} ({best_vn[1]['vietnamese_ratio']*100:.0f}%)")
    print(f"⚡ Fastest: {fastest[1]['name']} ({fastest[1]['time']:.1f}s)")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <query> [--detailed]")
        print("Example: python compare_models.py 'Ho gà ở trẻ em' --detailed")
        return
    
    query = sys.argv[1]
    detailed = "--detailed" in sys.argv
    
    print("🚀 MODEL RESPONSE COMPARISON TOOL")
    print("=" * 60)
    
    if detailed:
        compare_models_detailed(query)
    else:
        quick_comparison(query)

if __name__ == "__main__":
    main()