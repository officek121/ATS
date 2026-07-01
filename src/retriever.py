import os
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

class HybridRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the Hybrid Retriever with BM25 and a Sentence Transformer model.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS Index: IndexFlatIP for Inner Product (Cosine Similarity if normalized)
        self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
        
        self.bm25_corpus = []
        self.bm25 = None
        
        self.metadata_store = [] # Maps integer index to resume metadata (e.g. filename, full text)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Adds parsed resumes to both FAISS and BM25 indices.
        documents should be a list of dicts: {'id': str, 'text': str, 'metadata': dict}
        """
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        
        # 1. Update BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_corpus.extend(tokenized_corpus)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        # 2. Update FAISS
        # Compute embeddings
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        # Add to FAISS index
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        
        # 3. Update Metadata Store
        self.metadata_store.extend(documents)

    def retrieve(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search (BM25 + Semantic FAISS) using Reciprocal Rank Fusion (RRF).
        """
        if len(self.metadata_store) == 0:
            return []

        # Bound top_k to the number of documents we actually have
        top_k = min(top_k, len(self.metadata_store))

        # 1. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top indices for BM25
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 2. FAISS Semantic Search
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        faiss_scores, faiss_top_indices = self.faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
        faiss_top_indices = faiss_top_indices[0] # First query's results

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {i: 0.0 for i in range(len(self.metadata_store))}
        
        # Add BM25 RRF scores
        for rank, doc_idx in enumerate(bm25_top_indices):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)
            
        # Add FAISS RRF scores
        for rank, doc_idx in enumerate(faiss_top_indices):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)
            
        # Sort combined scores
        sorted_indices = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Return top candidates with their RRF scores
        results = []
        for doc_idx, score in sorted_indices[:top_k]:
            if score > 0: # Only return docs that actually appeared in the top K of either search
                doc_data = self.metadata_store[doc_idx].copy()
                doc_data['hybrid_score'] = score
                results.append(doc_data)
                
        return results

if __name__ == "__main__":
    retriever = HybridRetriever()
    print("Retriever initialized.")
