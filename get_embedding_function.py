from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import time
from typing import List, Dict, Optional
import hashlib
import json
from functools import lru_cache
import re


class HybridEmbeddingFunction:
    """
    Multi-model embedding strategy with ensemble approach, query expansion, and caching.
    """
    
    def __init__(self):
        self.weights = {
            'primary': 0.5,    # all-MiniLM-L6-v2 (general)
            'secondary': 0.3,  # Vietnamese-specific 
            'tertiary': 0.2    # medical domain
        }
        
        # Initialize embedding models
        self.primary_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Vietnamese-specific model (fallback to multilingual if not available)
        try:
            self.secondary_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
        except:
            # Fallback to same as primary if Vietnamese model not available
            self.secondary_model = self.primary_model
            
        # Medical domain model (fallback to BioBERT-like model)
        try:
            self.tertiary_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Using same for now
                model_kwargs={'device': 'cpu'}
            )
        except:
            self.tertiary_model = self.primary_model
        
        # Caching
        self.embedding_cache = {}
        self.max_cache_size = 100
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.success_rate_history = []
        self.weight_adjustments = []
        
        # Vietnamese medical term dictionary
        self.medical_synonyms = {
            'viêm phổi': ['viêm phổi', 'pneumonia', 'phổi viêm', 'nhiễm trùng phổi'],
            'sốt': ['sốt', 'fever', 'nóng sốt', 'có nhiệt'],
            'ho': ['ho', 'cough', 'ho khan', 'ho có đờm'],
            'đau bụng': ['đau bụng', 'stomach pain', 'bụng đau', 'đau dạ dày'],
            'tiêu chảy': ['tiêu chảy', 'diarrhea', 'chảy ruột', 'đi lỏng'],
            'viêm họng': ['viêm họng', 'sore throat', 'đau họng', 'họng viêm'],
            'triệu chứng': ['triệu chứng', 'symptoms', 'dấu hiệu', 'biểu hiện'],
            'chẩn đoán': ['chẩn đoán', 'diagnosis', 'khám bệnh', 'phát hiện bệnh'],
            'điều trị': ['điều trị', 'treatment', 'chữa trị', 'trị liệu'],
            'thuốc': ['thuốc', 'medication', 'dược phẩm', 'y phẩm'],
            'liều lượng': ['liều lượng', 'dosage', 'liều dùng', 'lượng thuốc']
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _update_cache(self, key: str, embedding: List[float]):
        """Update LRU cache with size limit"""
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand Vietnamese medical terms to synonyms"""
        expanded_queries = [query]  # Always include original
        
        query_lower = query.lower()
        
        # Find matching medical terms and add synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                # Create expanded versions
                for synonym in synonyms[1:]:  # Skip the original term
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
                break  # Only expand first matching term to avoid explosion
        
        # Limit to 3 variations to control compute cost
        return expanded_queries[:3]
    
    def get_ensemble_embedding(self, text: str) -> List[float]:
        """Get ensemble embedding from multiple models"""
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]['embedding']
        
        self.cache_misses += 1
        
        # Get embeddings from all models
        try:
            primary_emb = np.array(self.primary_model.embed_query(text))
            secondary_emb = np.array(self.secondary_model.embed_query(text))
            tertiary_emb = np.array(self.tertiary_model.embed_query(text))
            
            # Ensure all embeddings have same dimension (pad/truncate if needed)
            target_dim = len(primary_emb)
            
            if len(secondary_emb) != target_dim:
                if len(secondary_emb) > target_dim:
                    secondary_emb = secondary_emb[:target_dim]
                else:
                    secondary_emb = np.pad(secondary_emb, (0, target_dim - len(secondary_emb)))
            
            if len(tertiary_emb) != target_dim:
                if len(tertiary_emb) > target_dim:
                    tertiary_emb = tertiary_emb[:target_dim]
                else:
                    tertiary_emb = np.pad(tertiary_emb, (0, target_dim - len(tertiary_emb)))
            
            # Weighted ensemble
            ensemble_emb = (
                self.weights['primary'] * primary_emb +
                self.weights['secondary'] * secondary_emb +
                self.weights['tertiary'] * tertiary_emb
            )
            
            # Normalize
            ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb)
            
            # Convert to list and cache
            result = ensemble_emb.tolist()
            self._update_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"Warning: Ensemble embedding failed, using primary only: {e}")
            # Fallback to primary model only
            primary_emb = self.primary_model.embed_query(text)
            self._update_cache(cache_key, primary_emb)
            return primary_emb
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query with expansion and ensemble"""
        # Expand query to variations
        expanded_queries = self.expand_query(query)
        
        # Get embeddings for all variations
        embeddings = []
        for q in expanded_queries:
            emb = self.get_ensemble_embedding(q)
            embeddings.append(np.array(emb))
        
        # Average all embeddings
        if len(embeddings) > 1:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
            return avg_embedding.tolist()
        else:
            return embeddings[0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self.get_ensemble_embedding(text) for text in texts]
    
    def adjust_weights_based_on_performance(self, success_rate: float):
        """Dynamic weight adjustment based on retrieval success"""
        self.success_rate_history.append(success_rate)
        
        # Only adjust if we have enough history
        if len(self.success_rate_history) < 10:
            return
        
        # Calculate recent performance trend
        recent_avg = np.mean(self.success_rate_history[-5:])
        older_avg = np.mean(self.success_rate_history[-10:-5])
        
        # If performance is declining, adjust weights
        if recent_avg < older_avg - 0.1:  # 10% decline threshold
            adjustment = {
                'timestamp': time.time(),
                'old_weights': self.weights.copy(),
                'reason': f"Performance decline: {recent_avg:.3f} vs {older_avg:.3f}"
            }
            
            # Increase Vietnamese model weight if Vietnamese terms detected
            self.weights['secondary'] = min(0.4, self.weights['secondary'] + 0.05)
            self.weights['primary'] = max(0.3, self.weights['primary'] - 0.03)
            self.weights['tertiary'] = max(0.15, self.weights['tertiary'] - 0.02)
            
            adjustment['new_weights'] = self.weights.copy()
            self.weight_adjustments.append(adjustment)
            
            print(f"🔄 Auto-adjusted embedding weights: {adjustment['reason']}")
    
    def get_cache_stats(self) -> Dict:
        """Get caching performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.embedding_cache),
            'weight_adjustments': len(self.weight_adjustments)
        }


def get_embedding_function():
    """Return the hybrid embedding function"""
    return HybridEmbeddingFunction()
