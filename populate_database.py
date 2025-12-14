import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Advanced chunk splitting strategy for medical documents with:
    1. Section boundary preservation
    2. Table and medical data block detection
    3. Dynamic chunk sizing
    4. Relevance scoring
    5. Enhanced metadata tagging
    """
    import re
    from typing import List, Tuple
    
    def detect_content_type(text: str) -> str:
        """Detect content type for metadata enrichment"""
        # Table detection (common patterns)
        table_patterns = [
            r'\|.*\|.*\|',  # Markdown tables
            r'^\s*\d+\.\s+.*:\s*\d+',  # Numbered lists with values
            r'Liều\s*lượng|Thuốc|mg/kg|ml/kg',  # Dosage patterns
        ]
        
        # Section header detection
        section_patterns = [
            r'^#{1,6}\s+',  # Markdown headers
            r'^[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][^.]*:$',  # Vietnamese section headers
            r'^\d+\.\s*[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]',  # Numbered sections
        ]
        
        # Procedure detection
        procedure_patterns = [
            r'Cách\s+thực\s+hiện|Quy\s+trình|Thủ\s+thuật|Điều\s+trị',
            r'Bước\s+\d+|Giai\s+đoạn\s+\d+',
        ]
        
        text_lower = text.lower()
        
        # Check for tables
        for pattern in table_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return "[TABLE]"
        
        # Check for procedures
        for pattern in procedure_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return "[PROCEDURE]"
        
        # Check for sections
        for pattern in section_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "[SECTION]"
        
        return "[CONTENT]"
    
    def calculate_relevance_score(text: str) -> float:
        """Calculate relevance score for medical content"""
        high_value_terms = [
            'triệu chứng', 'chẩn đoán', 'điều trị', 'thuốc', 'liều lượng',
            'nguyên nhân', 'biến chứng', 'phòng ngừa', 'khám', 'xét nghiệm'
        ]
        
        score = 0.5  # Base score
        text_lower = text.lower()
        
        for term in high_value_terms:
            if term in text_lower:
                score += 0.1
        
        # Bonus for structured content
        if re.search(r'\d+\.\s+', text):  # Numbered lists
            score += 0.15
        
        # Penalty for very short fragments
        if len(text) < 50:
            score -= 0.2
            
        return min(score, 1.0)
    
    def smart_sentence_split(text: str, max_size: int) -> List[str]:
        """Split text at sentence boundaries for Vietnamese"""
        import re
        
        # Vietnamese sentence endings
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + ". " if sentence else ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def preserve_section_boundaries(text: str) -> List[str]:
        """Split text while preserving section boundaries"""
        import re
        
        # Find section headers
        section_pattern = r'(^#{1,6}\s+.*$|^[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][^.]*:$|^\d+\.\s*[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])'
        
        sections = re.split(section_pattern, text, flags=re.MULTILINE)
        result = []
        
        for i, section in enumerate(sections):
            if section and section.strip():
                result.append(section.strip())
        
        return result if result else [text]
    
    # Main splitting logic
    all_chunks = []
    
    for doc in documents:
        content = doc.page_content
        content_type = detect_content_type(content)
        
        # Strategy 1: Short medical facts - keep whole
        if len(content) < 200:
            chunk = Document(
                page_content=content,
                metadata={
                    **doc.metadata,
                    'content_type': content_type,
                    'relevance_score': calculate_relevance_score(content),
                    'chunk_strategy': 'whole_small'
                }
            )
            all_chunks.append(chunk)
            continue
        
        # Strategy 2: Tables and structured data - don't split
        if content_type == "[TABLE]":
            chunk = Document(
                page_content=content,
                metadata={
                    **doc.metadata,
                    'content_type': content_type,
                    'relevance_score': calculate_relevance_score(content),
                    'chunk_strategy': 'whole_table'
                }
            )
            all_chunks.append(chunk)
            continue
        
        # Strategy 3: Section-aware splitting
        if content_type == "[SECTION]":
            sections = preserve_section_boundaries(content)
            for section in sections:
                if len(section) > 800:
                    # Further split large sections
                    sub_chunks = smart_sentence_split(section, 800)
                    for sub_chunk in sub_chunks:
                        chunk = Document(
                            page_content=sub_chunk,
                            metadata={
                                **doc.metadata,
                                'content_type': content_type,
                                'relevance_score': calculate_relevance_score(sub_chunk),
                                'chunk_strategy': 'section_split'
                            }
                        )
                        all_chunks.append(chunk)
                else:
                    chunk = Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            'content_type': content_type,
                            'relevance_score': calculate_relevance_score(section),
                            'chunk_strategy': 'section_whole'
                        }
                    )
                    all_chunks.append(chunk)
            continue
        
        # Strategy 4: Complex procedures and long content
        if len(content) > 500:
            chunks = smart_sentence_split(content, 800)
            for chunk_text in chunks:
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'content_type': content_type,
                        'relevance_score': calculate_relevance_score(chunk_text),
                        'chunk_strategy': 'sentence_split'
                    }
                )
                all_chunks.append(chunk)
        else:
            # Strategy 5: Fallback to standard splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update({
                    'content_type': content_type,
                    'relevance_score': calculate_relevance_score(chunk.page_content),
                    'chunk_strategy': 'standard'
                })
            all_chunks.extend(chunks)
    
    return all_chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
