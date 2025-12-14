import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Query type definitions and configurations
QUERY_TYPES = {
    "symptom": {
        "keywords": ["triệu chứng", "biểu hiện", "dấu hiệu", "symptoms", "signs", "có thể", "thường"],
        "k": 3,  # Need fewer chunks for symptom lists
        "template": """
Dựa trên thông tin y tế sau, hãy liệt kê các triệu chứng chính:

{context}

---

Câu hỏi: {question}

Hướng dẫn trả lời:
1. Liệt kê triệu chứng chính trước
2. Sau đó nêu nguyên nhân (nếu có)
3. Cuối cùng là các dấu hiệu cảnh báo nguy hiểm
4. Sử dụng format danh sách rõ ràng
"""
    },
    "dosage": {
        "keywords": ["liều", "dose", "mg", "kg", "ml", "lần", "ngày", "tuổi", "cân nặng"],
        "k": 4,  # Need more context for dosage calculations
        "template": """
Dựa trên thông tin dược lý sau, tính toán liều dùng thuốc:

{context}

---

Câu hỏi: {question}

Hướng dẫn tính liều:
1. Xác định liều/kg cân nặng
2. Công thức: (liều/kg) × cân nặng bệnh nhân
3. Tần suất sử dụng (số lần/ngày)
4. Thời gian điều trị
5. LƯU Ý an toàn và chống chỉ định

Format bảng: Cân nặng | Liều dùng | Tần suất
"""
    },
    "procedure": {
        "keywords": ["cách", "quy trình", "thủ thuật", "làm thế nào", "how to", "steps", "bước"],
        "k": 5,  # Need detailed step-by-step context
        "template": """
Dựa trên hướng dẫn y tế sau, mô tả quy trình thực hiện:

{context}

---

Câu hỏi: {question}

Hướng dẫn trình bày:
1. Chia thành các bước có đánh số rõ ràng
2. Sử dụng bullet points cho các điểm quan trọng
3. Nêu rõ dụng cụ cần thiết
4. Lưu ý an toàn và biến chứng
5. Khi nào cần tham khảo bác sĩ
"""
    },
    "definition": {
        "keywords": ["là gì", "định nghĩa", "what is", "khái niệm", "có nghĩa là"],
        "k": 2,  # Simple definition needs fewer chunks
        "template": """
Dựa trên kiến thức y khoa sau, định nghĩa thuật ngữ:

{context}

---

Câu hỏi: {question}

Hướng dẫn trả lời:
1. Định nghĩa chính xác và ngắn gọn
2. Giải thích bằng ngôn ngữ dễ hiểu
3. Nêu các đặc điểm quan trọng
4. Đưa ra ví dụ minh họa nếu cần
"""
    },
    "contraindication": {
        "keywords": ["an toàn", "chống chỉ định", "tác dụng phụ", "safe", "side effects", "nguy hiểm"],
        "k": 6,  # Need comprehensive safety information
        "template": """
Dựa trên thông tin an toàn y tế sau, đánh giá độ an toàn:

{context}

---

Câu hỏi: {question}

⚠️ HƯỚNG DẪN AN TOÀN:
1. LUÔN nêu các chống chỉ định trước
2. Liệt kê tác dụng phụ có thể xảy ra
3. Nhóm đối tượng cần thận trọng
4. Liều lượng an toàn
5. Dấu hiệu cần ngừng sử dụng ngay

🚨 KHÔNG BAO GIỜ bỏ qua thông tin về chống chỉ định!
"""
    },
    "interaction": {
        "keywords": ["tương tác", "kết hợp", "dùng chung", "interaction", "together"],
        "k": 4,
        "template": """
Dựa trên dữ liệu tương tác thuốc sau, phân tích khả năng tương tác:

{context}

---

Câu hỏi: {question}

Phân tích tương tác:
1. Loại tương tác (tăng/giảm hiệu quả, độc tính)
2. Mức độ nghiêm trọng (nhẹ/trung bình/nặng)
3. Cơ chế tương tác
4. Khuyến nghị điều chỉnh liều
5. Theo dõi các dấu hiệu bất thường
"""
    }
}

# Default template for unclassified queries
DEFAULT_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--show-sources", action="store_true", help="Show source documents.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text, show_sources=args.show_sources)


def detect_query_type(query: str) -> str:
    """
    Detect query type using keyword matching and pattern recognition.
    """
    query_lower = query.lower()
    
    # Score each query type based on keyword matches
    type_scores = {}
    
    for query_type, config in QUERY_TYPES.items():
        score = 0
        keywords = config["keywords"]
        
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        
        # Bonus scoring for specific patterns
        if query_type == "dosage":
            # Look for weight/age patterns
            import re
            if re.search(r'\d+\s*(kg|tuổi|năm|tháng)', query_lower):
                score += 2
            if re.search(r'(mg|ml|liều)', query_lower):
                score += 2
        
        elif query_type == "procedure":
            # Look for question patterns about "how"
            if any(pattern in query_lower for pattern in ["làm thế nào", "cách nào", "quy trình"]):
                score += 2
        
        elif query_type == "symptom":
            # Look for symptom inquiry patterns
            if any(pattern in query_lower for pattern in ["có triệu chứng gì", "biểu hiện như thế nào"]):
                score += 2
        
        type_scores[query_type] = score
    
    # Return type with highest score, or 'general' if no clear match
    if max(type_scores.values()) > 0:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    else:
        return "general"


def get_prompt_template(query_type: str, query: str, context: str) -> str:
    """
    Get specialized prompt template based on query type.
    """
    if query_type in QUERY_TYPES:
        return QUERY_TYPES[query_type]["template"].format(context=context, question=query)
    else:
        return DEFAULT_PROMPT_TEMPLATE.format(context=context, question=query)


def get_optimal_k(query_type: str) -> int:
    """
    Get optimal number of chunks to retrieve based on query type.
    """
    if query_type in QUERY_TYPES:
        return QUERY_TYPES[query_type]["k"]
    else:
        return 5  # Default k


def format_response_by_type(response: str, query_type: str) -> str:
    """
    Apply type-specific formatting to response.
    """
    if query_type == "symptom":
        # Ensure markdown list formatting
        lines = response.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('*') and not line.startswith('1.'):
                # Check if it looks like a symptom item
                if any(indicator in line.lower() for indicator in ['triệu chứng', 'biểu hiện', 'có thể', 'thường']):
                    if ',' in line or ';' in line:
                        # Split compound symptoms
                        symptoms = [s.strip() for s in line.replace(';', ',').split(',') if s.strip()]
                        for symptom in symptoms:
                            formatted_lines.append(f"• {symptom}")
                    else:
                        formatted_lines.append(f"• {line}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)
    
    elif query_type == "dosage":
        # Try to structure dosage information in table format
        import re
        lines = response.split('\n')
        formatted_lines = ["## 💊 THÔNG TIN LIỀU DÙNG\n"]
        
        dosage_found = False
        for line in lines:
            # Look for dosage patterns
            if re.search(r'\d+.*mg|\d+.*ml|\d+.*kg', line):
                dosage_found = True
                formatted_lines.append(f"📋 **{line.strip()}**")
            elif 'liều' in line.lower() or 'dose' in line.lower():
                formatted_lines.append(f"🔢 {line.strip()}")
            elif line.strip():
                formatted_lines.append(line.strip())
        
        if dosage_found:
            formatted_lines.append("\n⚠️ **Lưu ý**: Luôn tham khảo ý kiến bác sĩ trước khi sử dụng thuốc.")
        
        return '\n'.join(formatted_lines)
    
    elif query_type == "procedure":
        # Format as numbered steps
        lines = response.split('\n')
        formatted_lines = ["## 📋 QUY TRÌNH THỰC HIỆN\n"]
        
        step_counter = 1
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Check if it's a step
                if any(indicator in line.lower() for indicator in ['bước', 'step', 'đầu tiên', 'sau đó', 'cuối cùng']):
                    formatted_lines.append(f"{step_counter}. **{line}**")
                    step_counter += 1
                elif line.startswith('•') or line.startswith('-'):
                    formatted_lines.append(f"   {line}")
                else:
                    formatted_lines.append(line)
        
        formatted_lines.append("\n⚠️ **An toàn**: Thực hiện theo đúng hướng dẫn và tham khảo chuyên gia khi cần thiết.")
        return '\n'.join(formatted_lines)
    
    elif query_type == "contraindication":
        # Add warning formatting
        warning_response = f"""🚨 **THÔNG TIN AN TOÀN QUAN TRỌNG**

{response}

⚠️ **CẢNH BÁO**: Thông tin này chỉ mang tính chất tham khảo. LUÔN tham khảo ý kiến bác sĩ trước khi sử dụng bất kỳ loại thuốc nào."""
        return warning_response
    
    else:
        # Default formatting
        return response


def rerank_results(results, query_text: str, weights=None) -> list:
    """
    Semantic re-ranking layer to improve relevance filtering beyond cosine similarity.
    
    Args:
        results: List of (Document, score) tuples from initial retrieval
        query_text: Original query text
        weights: Custom weights dict for different factors
    
    Returns:
        List of (Document, enhanced_score, confidence, reasoning) tuples
    """
    import re
    from typing import Dict, List, Tuple, Any
    
    if weights is None:
        weights = {
            'medical_keywords': 1.5,
            'medical_units': 1.3,
            'question_type': 1.2,
            'coherence': 1.4,
            'diversity': 1.1,
            'chunk_quality': 1.3
        }
    
    def get_medical_keyword_boost(text: str, query: str) -> float:
        """Calculate boost based on medical keyword matching"""
        medical_keywords = {
            'triệu chứng': ['triệu chứng', 'dấu hiệu', 'biểu hiện', 'có thể'],
            'chẩn đoán': ['chẩn đoán', 'xét nghiệm', 'khám', 'phát hiện', 'nhận biết'],
            'điều trị': ['điều trị', 'chữa trị', 'thuốc', 'phương pháp', 'cách chữa'],
            'liều dùng': ['liều', 'mg', 'ml', 'lần', 'ngày', 'dùng'],
            'phòng ngừa': ['phòng ngừa', 'dự phòng', 'tránh', 'ngăn chặn'],
            'nguyên nhân': ['nguyên nhân', 'do', 'gây ra', 'vì sao', 'tại sao']
        }
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        boost = 1.0
        
        # Identify query intent
        query_intent = None
        for intent, keywords in medical_keywords.items():
            if any(kw in query_lower for kw in keywords):
                query_intent = intent
                break
        
        if query_intent:
            # Check if content matches query intent
            intent_keywords = medical_keywords[query_intent]
            matches = sum(1 for kw in intent_keywords if kw in text_lower)
            if matches > 0:
                boost *= weights['medical_keywords']
        
        return boost
    
    def get_medical_units_boost(text: str) -> float:
        """Calculate boost for medical units and measurements"""
        unit_patterns = [
            r'\d+\s*(mg|ml|kg|g|°C|tuổi|tháng|năm)',
            r'\d+\s*(lần|ngày|giờ|phút)',
            r'\d+-\d+\s*(mg|ml|kg)',
            r'liều\s*lượng',
            r'trọng\s*lượng'
        ]
        
        text_lower = text.lower()
        unit_matches = sum(1 for pattern in unit_patterns if re.search(pattern, text_lower))
        
        if unit_matches > 0:
            return weights['medical_units']
        return 1.0
    
    def get_question_type_boost(text: str, query: str) -> float:
        """Calculate boost based on question type matching"""
        question_patterns = {
            'what': ['là gì', 'gì là', 'định nghĩa', 'khái niệm'],
            'how': ['như thế nào', 'cách nào', 'làm sao', 'quy trình', 'thủ thuật'],
            'when': ['khi nào', 'lúc nào', 'thời điểm', 'thời gian'],
            'why': ['tại sao', 'vì sao', 'nguyên nhân', 'do đâu'],
            'where': ['ở đâu', 'vị trí', 'chỗ nào'],
            'how_much': ['bao nhiêu', 'mức độ', 'số lượng', 'liều']
        }
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Identify question type
        question_type = None
        for qtype, patterns in question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                question_type = qtype
                break
        
        if question_type and question_type in question_patterns:
            # Check if answer pattern matches question type
            answer_indicators = {
                'what': ['là', 'được định nghĩa', 'có nghĩa'],
                'how': ['bước', 'giai đoạn', 'cách', 'phương pháp'],
                'when': ['khi', 'lúc', 'sau khi', 'trước khi'],
                'why': ['do', 'vì', 'nguyên nhân', 'gây ra'],
                'where': ['tại', 'ở', 'vùng', 'khu vực'],
                'how_much': ['mg', 'ml', 'kg', 'lần', 'liều']
            }
            
            if question_type in answer_indicators:
                indicators = answer_indicators[question_type]
                if any(ind in text_lower for ind in indicators):
                    return weights['question_type']
        
        return 1.0
    
    def get_coherence_score(chunk_text: str, query: str) -> float:
        """Calculate coherence score for logical sequence"""
        # Check for structured content
        structure_indicators = [
            r'^\d+\.\s+',  # Numbered lists
            r'[Bb]ước\s+\d+',  # Steps
            r'[Gg]iai\s+đoạn\s+\d+',  # Phases
            r'[Tt]riệu\s+chứng.*:',  # Symptom lists
            r'[Nn]guyên\s+nhân.*:'  # Cause lists
        ]
        
        coherence = 1.0
        
        # Bonus for structured content
        for pattern in structure_indicators:
            if re.search(pattern, chunk_text, re.MULTILINE):
                coherence *= 1.2
                break
        
        # Check for completeness
        if len(chunk_text.strip()) > 100:  # Substantial content
            coherence *= 1.1
        
        # Penalty for fragmented content
        if chunk_text.count('...') > 2:
            coherence *= 0.8
        
        return min(coherence * weights['coherence'], 2.0)
    
    def get_diversity_score(results_list: List, current_doc) -> float:
        """Calculate diversity score to avoid redundant sources"""
        current_source = current_doc.metadata.get('source', '')
        current_page = current_doc.metadata.get('page', -1)
        
        # Check for same source/page redundancy
        same_page_count = sum(1 for doc, _ in results_list 
                             if doc.metadata.get('source', '') == current_source and 
                                doc.metadata.get('page', -1) == current_page)
        
        if same_page_count > 1:
            return 0.7  # Penalty for redundancy
        
        return weights['diversity']
    
    def get_chunk_quality_score(doc) -> float:
        """Score based on chunk metadata quality"""
        metadata = doc.metadata
        
        score = 1.0
        
        # Bonus for high relevance score from chunking
        if 'relevance_score' in metadata:
            relevance = metadata['relevance_score']
            score *= (1 + relevance * 0.5)  # Up to 1.5x boost
        
        # Bonus for important content types
        content_type = metadata.get('content_type', '')
        if '[PROCEDURE]' in content_type or '[TABLE]' in content_type:
            score *= 1.2
        
        return min(score * weights['chunk_quality'], 2.0)
    
    def generate_reasoning(doc, boosts: Dict) -> str:
        """Generate human-readable reasoning for ranking"""
        reasons = []
        
        if boosts.get('medical_keywords', 1.0) > 1.0:
            reasons.append("strong medical keyword match")
        
        if boosts.get('medical_units', 1.0) > 1.0:
            reasons.append("contains specific dosages/measurements")
        
        if boosts.get('question_type', 1.0) > 1.0:
            reasons.append("matches question type pattern")
        
        if boosts.get('coherence', 1.0) > 1.0:
            reasons.append("well-structured content")
        
        if boosts.get('chunk_quality', 1.0) > 1.0:
            reasons.append("high-quality chunk")
        
        content_type = doc.metadata.get('content_type', '')
        if '[PROCEDURE]' in content_type:
            reasons.append("procedural content")
        elif '[TABLE]' in content_type:
            reasons.append("tabular data")
        
        return "; ".join(reasons) if reasons else "general relevance"
    
    # Main re-ranking logic
    enhanced_results = []
    
    for i, (doc, original_score) in enumerate(results):
        # Calculate various boost factors
        boosts = {
            'medical_keywords': get_medical_keyword_boost(doc.page_content, query_text),
            'medical_units': get_medical_units_boost(doc.page_content),
            'question_type': get_question_type_boost(doc.page_content, query_text),
            'coherence': get_coherence_score(doc.page_content, query_text),
            'diversity': get_diversity_score(results[:i], doc),
            'chunk_quality': get_chunk_quality_score(doc)
        }
        
        # Calculate enhanced score
        enhanced_score = original_score
        for boost_value in boosts.values():
            enhanced_score *= boost_value
        
        # Calculate confidence (0-1)
        confidence = min(enhanced_score / (original_score * 2.0), 1.0)
        
        # Generate reasoning
        reasoning = generate_reasoning(doc, boosts)
        
        enhanced_results.append((doc, enhanced_score, confidence, reasoning))
    
    # Sort by enhanced score (descending)
    enhanced_results.sort(key=lambda x: x[1], reverse=True)
    
    return enhanced_results


def query_rag(query_text: str, show_sources: bool = False):
    # Detect query type for adaptive processing
    query_type = detect_query_type(query_text)
    optimal_k = get_optimal_k(query_type)
    
    if show_sources:
        print(f"🔍 Detected query type: {query_type.upper()}")
        print(f"📊 Using k={optimal_k} chunks for this query type")
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with adaptive k parameter
    initial_results = db.similarity_search_with_score(query_text, k=optimal_k + 2)  # Get extra for re-ranking

    if len(initial_results) == 0:
        print("Unable to find matching results.")
        return

    # Apply semantic re-ranking
    reranked_results = rerank_results(initial_results, query_text)
    
    # Take optimal number of results after re-ranking
    top_results = reranked_results[:optimal_k]
    
    # Extract documents and build context
    context_chunks = []
    for doc, enhanced_score, confidence, reasoning in top_results:
        context_chunks.append(doc.page_content)
    
    context_text = "\n\n---\n\n".join(context_chunks)
    
    # Use adaptive prompt template
    prompt = get_prompt_template(query_type, query_text, context_text)

    # For testing purposes without Ollama, create context-based response
    if query_type == "dosage":
        response_text = f"Thông tin liều dùng dựa trên tài liệu y tế:\n\n{context_text[:800]}..."
    elif query_type == "symptom":
        response_text = f"Các triệu chứng được ghi nhận:\n\n{context_text[:800]}..."
    elif query_type == "procedure":
        response_text = f"Quy trình thực hiện:\n\n{context_text[:800]}..."
    elif query_type == "contraindication":
        response_text = f"⚠️ Thông tin an toàn:\n\n{context_text[:800]}..."
    else:
        response_text = f"Based on the provided context, here's what I found:\n\n{context_text[:800]}..."
    
    # Apply type-specific formatting
    formatted_response = format_response_by_type(response_text, query_type)
    
    print(f"Response: {formatted_response}")
    
    if show_sources:
        print(f"\n📊 Detailed Source Analysis (Query Type: {query_type.upper()}):")
        for i, (doc, enhanced_score, confidence, reasoning) in enumerate(top_results):
            source_id = doc.metadata.get("id", "Unknown")
            content_type = doc.metadata.get("content_type", "[CONTENT]")
            original_relevance = doc.metadata.get("relevance_score", 0.5)
            
            print(f"\n  {i+1}. Source: {source_id}")
            print(f"     Type: {content_type}")
            print(f"     Confidence: {confidence:.2f}")
            print(f"     Original Relevance: {original_relevance:.2f}")
            print(f"     Reasoning: {reasoning}")
    
    return formatted_response


if __name__ == "__main__":
    main()
