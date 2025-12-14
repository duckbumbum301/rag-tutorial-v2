#!/usr/bin/env python3
"""
Model Test Tool: Test 1 câu hỏi với 3 models và xem con nào trả lời tốt nhất
Usage: python test_models.py "câu hỏi của bạn"
"""

import sys
import time
from multi_model_evaluation import query_rag_with_model, MODELS

def test_all_models(query: str):
    """Test 1 query với cả 3 models và hiển thị đầy đủ"""
    
    print(f"🔍 TESTING QUERY: '{query}'")
    print("=" * 80)
    
    results = []
    
    # Test từng model
    for model_name, config in MODELS.items():
        print(f"\\n🤖 MODEL: {config['name']}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            response = query_rag_with_model(query, model_name)
            end_time = time.time()
            time_taken = end_time - start_time
            
            # Quick quality assessment
            vietnamese_chars = sum(1 for c in response if c in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
            total_chars = len(response)
            vn_ratio = vietnamese_chars / max(total_chars, 1) * 100
            
            # Simple scoring
            score = 0
            if vn_ratio > 10:  # Good Vietnamese content
                score += 25
            if len(response) > 100:  # Adequate length
                score += 25
            if time_taken < 40:  # Reasonable speed
                score += 25
            if any(word in response.lower() for word in ['triệu chứng', 'liều', 'thuốc', 'bác sĩ', 'điều trị', 'nguyên nhân']):
                score += 25
            
            print(f"⏱️  Response Time: {time_taken:.1f}s")
            print(f"📊 Quality Score: {score}/100")
            print(f"🇻🇳 Vietnamese: {vn_ratio:.1f}%")
            print(f"📝 Length: {len(response)} characters")
            print()
            print("📋 FULL RESPONSE:")
            print("─" * 60)
            print(response)
            print("─" * 60)
            
            results.append({
                'model_name': config['name'],
                'model_id': model_name,
                'response': response,
                'time': time_taken,
                'score': score,
                'vn_ratio': vn_ratio,
                'length': len(response)
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'model_name': config['name'],
                'model_id': model_name,
                'response': f"Error: {str(e)}",
                'time': 0,
                'score': 0,
                'vn_ratio': 0,
                'length': 0
            })
    
    # Ranking
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\\n🏆 FINAL RANKING:")
    print("=" * 80)
    
    medals = ["🥇 WINNER", "🥈 SECOND", "🥉 THIRD"]
    
    for i, result in enumerate(results):
        medal = medals[i] if i < 3 else f"{i+1}th"
        
        print(f"\\n{medal}: {result['model_name']}")
        print(f"   📊 Quality Score: {result['score']}/100")
        print(f"   ⏱️  Speed: {result['time']:.1f}s")
        print(f"   🇻🇳 Vietnamese: {result['vn_ratio']:.1f}%")
        print(f"   📏 Length: {result['length']} chars")
        
        if result['score'] >= 75:
            print(f"   ✅ Excellent quality")
        elif result['score'] >= 50:
            print(f"   ⚠️ Good quality")  
        elif result['score'] >= 25:
            print(f"   ❌ Poor quality")
        else:
            print(f"   💀 Very poor quality")
    
    # Summary recommendation
    winner = results[0]
    print(f"\\n💡 RECOMMENDATION:")
    print(f"   🏆 Best Model: {winner['model_name']}")
    print(f"   📈 Score: {winner['score']}/100")
    
    if winner['score'] >= 75:
        print(f"   ✅ This model gives excellent answers for your query type!")
    elif winner['score'] >= 50:
        print(f"   ⚠️ This model gives decent answers, but could be better.")
    else:
        print(f"   ❌ All models struggled with this query. Consider rephrasing.")
    
    return results

def main():
    if len(sys.argv) < 2:
        print("🚀 MODEL TESTING TOOL")
        print("=" * 40)
        print("Usage: python test_models.py '<your medical question>'")
        print()
        print("Examples:")
        print("  python test_models.py 'Ho gà ở trẻ em'")
        print("  python test_models.py 'Liều paracetamol cho trẻ 3 tuổi nặng 15kg'") 
        print("  python test_models.py 'Triệu chứng viêm phổi ở trẻ'")
        print("  python test_models.py 'Cách điều trị sốt xuất huyết'")
        print()
        print("📋 This will show you:")
        print("   • Full response from each model")
        print("   • Quality scores and rankings")
        print("   • Speed comparison")
        print("   • Which model is best for your query")
        return
    
    query = sys.argv[1]
    test_all_models(query)

if __name__ == "__main__":
    main()