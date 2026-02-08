"""
Evaluation Metrics for Hybrid RAG System
Implements MRR (mandatory) + ROUGE-L + NDCG@K (custom metrics)
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for RAG systems
    """
    
    def __init__(self):
        """Initialize metric calculators"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # ============================================================================
    # MANDATORY METRIC: Mean Reciprocal Rank (MRR) at URL Level
    # ============================================================================
    
    def calculate_mrr(self, retrieved_results: List[Dict], ground_truth_url: str) -> float:
        """
        Calculate Mean Reciprocal Rank at URL level (Mandatory Metric - 2 marks)
        
        **Justification:**
        MRR measures how quickly the system identifies the correct source document.
        It's critical for RAG systems as finding the right source early in the ranking
        directly impacts answer quality.
        
        **Calculation Method:**
        For each question, find the rank position (1-indexed) of the first occurrence
        of the correct Wikipedia URL in retrieved results.
        MRR = 1/rank if found, else 0
        
        **Interpretation:**
        - MRR = 1.0: Perfect - correct URL always ranked first
        - MRR = 0.5: Correct URL typically at rank 2
        - MRR = 0.0: System never retrieves correct URL
        - Higher is better, range [0, 1]
        
        Args:
            retrieved_results: List of retrieved chunks with metadata
            ground_truth_url: Correct source Wikipedia URL
            
        Returns:
            Reciprocal rank (1/rank if found, 0 otherwise)
        """
        for rank, result in enumerate(retrieved_results, start=1):
            if result.get("url") == ground_truth_url:
                return 1.0 / rank
        return 0.0
    
    def calculate_mean_mrr(self, results: List[Tuple[List[Dict], str]]) -> float:
        """
        Calculate mean MRR across all questions
        
        Args:
            results: List of (retrieved_results, ground_truth_url) tuples
            
        Returns:
            Mean MRR score
        """
        mrr_scores = [self.calculate_mrr(retrieved, gt_url) for retrieved, gt_url in results]
        return np.mean(mrr_scores) if mrr_scores else 0.0
    
    # ============================================================================
    # CUSTOM METRIC 1: ROUGE-L for Answer Quality (2 marks)
    # ============================================================================
    
    def calculate_rouge_l(self, generated_answer: str, ground_truth: str) -> float:
        """
        Calculate ROUGE-L F1 score (Custom Metric 1 - 2 marks)
        
        **Justification:**
        ROUGE-L measures the longest common subsequence between generated and
        ground truth answers, capturing word order and fluency. It's ideal for
        RAG systems as it evaluates both content overlap and answer coherence,
        unlike simple word overlap metrics.
        
        **Calculation Method:**
        ROUGE-L finds the longest common subsequence (LCS) between two texts:
        - Recall_LCS = LCS_length / ground_truth_length
        - Precision_LCS = LCS_length / generated_length
        - F1_LCS = 2 * (Precision * Recall) / (Precision + Recall)
        
        We use the harmonic mean (F1) to balance precision and recall.
        
        **Interpretation:**
        - ROUGE-L = 1.0: Perfect match in word order and content
        - ROUGE-L = 0.7-0.9: High quality, captures main information
        - ROUGE-L = 0.4-0.7: Moderate quality, partial information
        - ROUGE-L < 0.4: Poor quality, significant information loss
        - Higher is better, range [0, 1]
        
        Args:
            generated_answer: System-generated answer
            ground_truth: Reference ground truth answer
            
        Returns:
            ROUGE-L F1 score
        """
        if not generated_answer or not ground_truth:
            return 0.0
        
        scores = self.rouge_scorer.score(ground_truth, generated_answer)
        return scores['rougeL'].fmeasure
    
    # ============================================================================
    # CUSTOM METRIC 2: NDCG@K for Retrieval Quality (2 marks)
    # ============================================================================
    
    def calculate_ndcg_at_k(self, retrieved_results: List[Dict], 
                            ground_truth_url: str, k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K (Custom Metric 2 - 2 marks)
        
        **Justification:**
        NDCG@K evaluates both the relevance and ranking quality of retrieved documents.
        Unlike MRR which only considers the first correct result, NDCG accounts for
        multiple relevant documents and their positions. This is crucial for RAG
        systems where context quality depends on having multiple relevant chunks
        in top positions.
        
        **Calculation Method:**
        1. Assign relevance scores: 1 if URL matches ground truth, 0 otherwise
        2. Calculate DCG: DCG@K = Σ(rel_i / log2(i+1)) for i=1 to K
        3. Calculate IDCG (ideal DCG with perfect ranking)
        4. NDCG@K = DCG@K / IDCG@K
        
        The log2(i+1) discount emphasizes top-ranked results while still 
        rewarding relevant documents in lower positions.
        
        **Interpretation:**
        - NDCG@5 = 1.0: Perfect ranking, all relevant docs at top
        - NDCG@5 = 0.7-0.9: Good ranking, most relevant docs highly ranked
        - NDCG@5 = 0.4-0.7: Moderate ranking, some relevant docs in top-K
        - NDCG@5 < 0.4: Poor ranking, relevant docs ranked low
        - Higher is better, range [0, 1]
        
        Args:
            retrieved_results: List of retrieved chunks with metadata
            ground_truth_url: Correct source Wikipedia URL
            k: Number of top results to consider (default: 5)
            
        Returns:
            NDCG@K score
        """
        if not retrieved_results:
            return 0.0
        
        # Limit to top K results
        retrieved_results = retrieved_results[:k]
        
        # Calculate relevance scores (1 for correct URL, 0 otherwise)
        relevance_scores = [
            1.0 if result.get("url") == ground_truth_url else 0.0
            for result in retrieved_results
        ]
        
        # Calculate DCG@K
        dcg = sum(
            rel / np.log2(idx + 2)  # idx+2 because idx starts at 0
            for idx, rel in enumerate(relevance_scores)
        )
        
        # Calculate IDCG@K (ideal ranking - all relevant docs at top)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(
            rel / np.log2(idx + 2)
            for idx, rel in enumerate(ideal_relevance)
        )
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    # ============================================================================
    # Additional Helper Metrics
    # ============================================================================
    
    def calculate_precision_at_k(self, retrieved_results: List[Dict], 
                                  ground_truth_url: str, k: int = 5) -> float:
        """
        Calculate Precision@K: proportion of top-K results from correct URL
        
        Args:
            retrieved_results: List of retrieved chunks
            ground_truth_url: Correct source URL
            k: Number of top results
            
        Returns:
            Precision@K score
        """
        top_k = retrieved_results[:k]
        relevant = sum(1 for r in top_k if r.get("url") == ground_truth_url)
        return relevant / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self, retrieved_results: List[Dict], 
                              ground_truth_chunks: List[str], k: int = 5) -> float:
        """
        Calculate Recall@K: proportion of relevant chunks retrieved in top-K
        
        Args:
            retrieved_results: List of retrieved chunks
            ground_truth_chunks: List of relevant chunk IDs
            k: Number of top results
            
        Returns:
            Recall@K score
        """
        if not ground_truth_chunks:
            return 0.0
        
        top_k = retrieved_results[:k]
        retrieved_ids = {r.get("chunk_id") for r in top_k}
        relevant_retrieved = len(retrieved_ids.intersection(set(ground_truth_chunks)))
        
        return relevant_retrieved / len(ground_truth_chunks)
    
    def calculate_f1_score(self, generated_answer: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score
        
        Args:
            generated_answer: Generated answer text
            ground_truth: Ground truth answer text
            
        Returns:
            F1 score
        """
        if not generated_answer or not ground_truth:
            return 0.0
        
        # Tokenize and normalize
        gen_tokens = set(self._normalize_tokens(generated_answer))
        gt_tokens = set(self._normalize_tokens(ground_truth))
        
        if not gen_tokens or not gt_tokens:
            return 0.0
        
        # Calculate overlap
        common = gen_tokens.intersection(gt_tokens)
        
        if not common:
            return 0.0
        
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(gt_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _normalize_tokens(self, text: str) -> List[str]:
        """Normalize and tokenize text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def calculate_exact_match(self, generated_answer: str, ground_truth: str) -> float:
        """
        Calculate Exact Match score (binary: 1 if exact match, 0 otherwise)
        
        Args:
            generated_answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        gen_normalized = self._normalize_answer(generated_answer)
        gt_normalized = self._normalize_answer(ground_truth)
        
        return 1.0 if gen_normalized == gt_normalized else 0.0
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # ============================================================================
    # Comprehensive Evaluation
    # ============================================================================
    
    def evaluate_single_question(self, question_data: Dict, 
                                 retrieved_chunks: List[Dict],
                                 generated_answer: str) -> Dict:
        """
        Evaluate a single question with all metrics
        
        Args:
            question_data: Question metadata with ground truth
            retrieved_chunks: Retrieved chunks from RAG system
            generated_answer: Generated answer from LLM
            
        Returns:
            Dictionary with all metric scores
        """
        ground_truth_url = question_data.get("source_url")
        ground_truth_answer = question_data.get("ground_truth", "")
        
        results = {
            "question_id": question_data.get("question_id"),
            "question": question_data.get("question"),
            "question_type": question_data.get("question_type"),
            "generated_answer": generated_answer,
            "ground_truth": ground_truth_answer,
            
            # Mandatory metric
            "mrr": self.calculate_mrr(retrieved_chunks, ground_truth_url),
            
            # Custom metric 1: Answer quality
            "rouge_l": self.calculate_rouge_l(generated_answer, ground_truth_answer),
            
            # Custom metric 2: Retrieval quality
            "ndcg_at_5": self.calculate_ndcg_at_k(retrieved_chunks, ground_truth_url, k=5),
            
            # Additional metrics for analysis
            "precision_at_5": self.calculate_precision_at_k(retrieved_chunks, ground_truth_url, k=5),
            "f1_score": self.calculate_f1_score(generated_answer, ground_truth_answer),
            "exact_match": self.calculate_exact_match(generated_answer, ground_truth_answer),
            
            # Metadata
            "source_url": ground_truth_url,
            "retrieved_urls": [r.get("url") for r in retrieved_chunks[:5]]
        }
        
        return results
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all questions
        
        Args:
            results: List of per-question results
            
        Returns:
            Aggregated statistics
        """
        if not results:
            return {}
        
        # Overall metrics
        aggregated = {
            "total_questions": len(results),
            
            # Main metrics (mean)
            "mean_mrr": np.mean([r["mrr"] for r in results]),
            "mean_rouge_l": np.mean([r["rouge_l"] for r in results]),
            "mean_ndcg_at_5": np.mean([r["ndcg_at_5"] for r in results]),
            
            # Additional metrics
            "mean_precision_at_5": np.mean([r["precision_at_5"] for r in results]),
            "mean_f1_score": np.mean([r["f1_score"] for r in results]),
            "exact_match_rate": np.mean([r["exact_match"] for r in results]),
            
            # Standard deviations
            "std_mrr": np.std([r["mrr"] for r in results]),
            "std_rouge_l": np.std([r["rouge_l"] for r in results]),
            "std_ndcg_at_5": np.std([r["ndcg_at_5"] for r in results]),
        }
        
        # Per question type breakdown
        type_results = defaultdict(list)
        for r in results:
            qtype = r.get("question_type", "unknown")
            type_results[qtype].append(r)
        
        aggregated["by_question_type"] = {}
        for qtype, qresults in type_results.items():
            aggregated["by_question_type"][qtype] = {
                "count": len(qresults),
                "mean_mrr": np.mean([r["mrr"] for r in qresults]),
                "mean_rouge_l": np.mean([r["rouge_l"] for r in qresults]),
                "mean_ndcg_at_5": np.mean([r["ndcg_at_5"] for r in qresults]),
            }
        
        return aggregated


def main():
    """Test metrics with sample data"""
    metrics = EvaluationMetrics()
    
    # Sample data
    retrieved = [
        {"url": "https://en.wikipedia.org/wiki/Test1", "chunk_id": "1"},
        {"url": "https://en.wikipedia.org/wiki/Test2", "chunk_id": "2"},
        {"url": "https://en.wikipedia.org/wiki/Correct", "chunk_id": "3"},
    ]
    
    ground_truth_url = "https://en.wikipedia.org/wiki/Correct"
    
    print("Testing MRR:")
    mrr = metrics.calculate_mrr(retrieved, ground_truth_url)
    print(f"MRR Score: {mrr:.4f}")
    
    print("\nTesting ROUGE-L:")
    gen_answer = "The capital of France is Paris."
    gt_answer = "Paris is the capital city of France."
    rouge_l = metrics.calculate_rouge_l(gen_answer, gt_answer)
    print(f"ROUGE-L Score: {rouge_l:.4f}")
    
    print("\nTesting NDCG@5:")
    ndcg = metrics.calculate_ndcg_at_k(retrieved, ground_truth_url, k=5)
    print(f"NDCG@5 Score: {ndcg:.4f}")


if __name__ == "__main__":
    main()
