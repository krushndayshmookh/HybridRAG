"""
Automated Evaluation Pipeline for Hybrid RAG System
Single-command execution: python evaluation_pipeline.py
"""

import json
import time
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from evaluation_metrics import EvaluationMetrics
from HybridRag import (
    Preprocessing, DenseRetriever, SparseRetriever, 
    RRF, ResponseGenerator
)


class InnovativeEvaluation:
    """
    Innovative Evaluation Components (4 marks)
    Includes: Ablation Studies + Error Analysis
    """
    
    def __init__(self, chunks, dense, sparse, rrf, generator):
        self.chunks = chunks
        self.dense = dense
        self.sparse = sparse
        self.rrf = rrf
        self.generator = generator
        self.metrics = EvaluationMetrics()
    
    def ablation_study(self, questions: List[Dict], top_k=10, final_n=5) -> Dict:
        """
        Ablation Study: Compare Dense-only vs Sparse-only vs Hybrid
        
        Tests which retrieval method contributes most to performance
        """
        print("\n" + "="*70)
        print("ABLATION STUDY: Comparing Retrieval Methods")
        print("="*70)
        
        results = {
            "dense_only": [],
            "sparse_only": [],
            "hybrid_rrf": []
        }
        
        for question_data in tqdm(questions[:50], desc="Ablation Study"):  # Use subset for speed
            query = question_data["question"]
            ground_truth_url = question_data["source_url"]
            
            # Dense-only retrieval
            dense_chunks = self.dense.retrieve(query, top_k=final_n)
            dense_mrr = self.metrics.calculate_mrr(dense_chunks, ground_truth_url)
            dense_ndcg = self.metrics.calculate_ndcg_at_k(dense_chunks, ground_truth_url, k=5)
            results["dense_only"].append({"mrr": dense_mrr, "ndcg": dense_ndcg})
            
            # Sparse-only retrieval
            sparse_chunks = self.sparse.retrieve(query, top_k=final_n)
            sparse_mrr = self.metrics.calculate_mrr(sparse_chunks, ground_truth_url)
            sparse_ndcg = self.metrics.calculate_ndcg_at_k(sparse_chunks, ground_truth_url, k=5)
            results["sparse_only"].append({"mrr": sparse_mrr, "ndcg": sparse_ndcg})
            
            # Hybrid RRF
            hybrid_chunks = self.rrf.retrieve(query, top_k=top_k, final_n=final_n)
            hybrid_mrr = self.metrics.calculate_mrr(hybrid_chunks, ground_truth_url)
            hybrid_ndcg = self.metrics.calculate_ndcg_at_k(hybrid_chunks, ground_truth_url, k=5)
            results["hybrid_rrf"].append({"mrr": hybrid_mrr, "ndcg": hybrid_ndcg})
        
        # Aggregate results
        summary = {}
        for method, scores in results.items():
            summary[method] = {
                "mean_mrr": sum(s["mrr"] for s in scores) / len(scores),
                "mean_ndcg": sum(s["ndcg"] for s in scores) / len(scores),
                "count": len(scores)
            }
        
        print("\nAblation Study Results:")
        print("-" * 70)
        for method, stats in summary.items():
            print(f"{method:15s} | MRR: {stats['mean_mrr']:.4f} | NDCG@5: {stats['mean_ndcg']:.4f}")
        print("-" * 70)
        
        return {
            "summary": summary,
            "detailed_results": results
        }
    
    def error_analysis(self, evaluation_results: List[Dict]) -> Dict:
        """
        Error Analysis: Categorize failures by question type
        
        Identifies patterns in system failures for targeted improvement
        """
        print("\n" + "="*70)
        print("ERROR ANALYSIS: Failure Pattern Detection")
        print("="*70)
        
        # Define failure thresholds
        MRR_THRESHOLD = 0.5
        ROUGE_THRESHOLD = 0.3
        
        failures = {
            "retrieval_failure": [],  # Low MRR
            "generation_failure": [],  # Low ROUGE-L
            "complete_failure": [],    # Both low
            "success": []              # Both good
        }
        
        for result in evaluation_results:
            mrr = result.get("mrr", 0)
            rouge = result.get("rouge_l", 0)
            
            if mrr < MRR_THRESHOLD and rouge < ROUGE_THRESHOLD:
                failures["complete_failure"].append(result)
            elif mrr < MRR_THRESHOLD:
                failures["retrieval_failure"].append(result)
            elif rouge < ROUGE_THRESHOLD:
                failures["generation_failure"].append(result)
            else:
                failures["success"].append(result)
        
        # Aggregate by question type
        type_analysis = {}
        for failure_type, examples in failures.items():
            by_type = {}
            for ex in examples:
                qtype = ex.get("question_type", "unknown")
                if qtype not in by_type:
                    by_type[qtype] = []
                by_type[qtype].append(ex)
            type_analysis[failure_type] = by_type
        
        # Print summary
        print("\nError Distribution:")
        print("-" * 70)
        total = len(evaluation_results)
        for failure_type, examples in failures.items():
            count = len(examples)
            pct = (count / total * 100) if total > 0 else 0
            print(f"{failure_type:20s}: {count:3d} ({pct:5.1f}%)")
        print("-" * 70)
        
        # Most common failure patterns
        print("\nFailure Patterns by Question Type:")
        print("-" * 70)
        for failure_type in ["retrieval_failure", "generation_failure", "complete_failure"]:
            examples = failures[failure_type]
            if examples:
                type_counts = {}
                for ex in examples:
                    qtype = ex.get("question_type", "unknown")
                    type_counts[qtype] = type_counts.get(qtype, 0) + 1
                
                print(f"\n{failure_type}:")
                for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {qtype:15s}: {count} failures")
        print("-" * 70)
        
        return {
            "failure_categories": {k: len(v) for k, v in failures.items()},
            "type_analysis": type_analysis,
            "failure_examples": {
                k: v[:3] for k, v in failures.items() if v  # Top 3 examples each
            }
        }
    
    def parameter_sensitivity(self, questions: List[Dict]) -> Dict:
        """
        Parameter Sensitivity: Test different K, N, and RRF k values
        
        Finds optimal hyperparameters for the hybrid system
        """
        print("\n" + "="*70)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        test_configs = [
            {"top_k": 5, "final_n": 3, "rrf_k": 60},
            {"top_k": 10, "final_n": 5, "rrf_k": 60},
            {"top_k": 15, "final_n": 7, "rrf_k": 60},
            {"top_k": 10, "final_n": 5, "rrf_k": 30},
            {"top_k": 10, "final_n": 5, "rrf_k": 90},
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\nTesting: top_k={config['top_k']}, final_n={config['final_n']}, rrf_k={config['rrf_k']}")
            
            # Create RRF with custom k
            rrf_temp = RRF(self.dense, self.sparse, k=config['rrf_k'])
            
            mrr_scores = []
            for question_data in tqdm(questions[:30], desc="Testing config", leave=False):
                query = question_data["question"]
                ground_truth_url = question_data["source_url"]
                
                chunks = rrf_temp.retrieve(query, top_k=config['top_k'], final_n=config['final_n'])
                mrr = self.metrics.calculate_mrr(chunks, ground_truth_url)
                mrr_scores.append(mrr)
            
            avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
            results.append({
                "config": config,
                "mean_mrr": avg_mrr
            })
            
            print(f"  → Mean MRR: {avg_mrr:.4f}")
        
        # Find best config
        best = max(results, key=lambda x: x["mean_mrr"])
        print(f"\nBest Configuration: {best['config']} with MRR: {best['mean_mrr']:.4f}")
        
        return {
            "configurations_tested": results,
            "best_config": best
        }


class EvaluationPipeline:
    """
    Main Evaluation Pipeline
    """
    
    def __init__(self, questions_file="evaluation_questions.json", 
                 chunks_file="wiki_chunks.jsonl"):
        """Initialize pipeline components"""
        print("="*70)
        print("INITIALIZING HYBRID RAG EVALUATION PIPELINE")
        print("="*70)
        
        # Load questions
        print("\n[1/5] Loading evaluation questions...")
        with open(questions_file, "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        print(f"✓ Loaded {len(self.questions)} questions")
        
        # Load chunks
        print("\n[2/5] Loading preprocessed chunks...")
        preprocess = Preprocessing()
        self.chunks = preprocess.load_chunks(chunks_file)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Initialize retrievers
        print("\n[3/5] Initializing dense retriever...")
        self.dense = DenseRetriever(self.chunks)
        print("✓ Dense retriever ready")
        
        print("\n[4/5] Initializing sparse retriever...")
        self.sparse = SparseRetriever(self.chunks)
        print("✓ Sparse retriever ready")
        
        print("\n[5/5] Initializing RRF and generator...")
        self.rrf = RRF(self.dense, self.sparse, k=60)
        self.generator = ResponseGenerator()
        print("✓ All components ready")
        
        # Metrics
        self.metrics = EvaluationMetrics()
        
        print("\n" + "="*70)
        print("PIPELINE INITIALIZATION COMPLETE")
        print("="*70 + "\n")
    
    def run_evaluation(self, top_k=10, final_n=5) -> List[Dict]:
        """
        Run evaluation on all questions
        
        Args:
            top_k: Retrieve top-K from each method
            final_n: Final number of chunks after RRF
            
        Returns:
            List of evaluation results
        """
        print("\n" + "="*70)
        print("EVALUATING RAG SYSTEM ON ALL QUESTIONS")
        print("="*70)
        
        results = []
        
        for question_data in tqdm(self.questions, desc="Evaluating questions"):
            query = question_data["question"]
            
            # Time the retrieval and generation
            start_time = time.time()
            
            # Retrieve chunks
            retrieved_chunks = self.rrf.retrieve(query, top_k=top_k, final_n=final_n)
            
            # Generate answer
            generated_answer = self.generator.generate(query, retrieved_chunks)
            
            elapsed_time = time.time() - start_time
            
            # Evaluate
            result = self.metrics.evaluate_single_question(
                question_data,
                retrieved_chunks,
                generated_answer
            )
            result["response_time"] = elapsed_time
            
            results.append(result)
        
        print(f"\n✓ Evaluated {len(results)} questions")
        return results
    
    def run_innovative_evaluation(self, results: List[Dict]) -> Dict:
        """
        Run innovative evaluation components
        
        Args:
            results: Standard evaluation results
            
        Returns:
            Innovative evaluation results
        """
        innovative = InnovativeEvaluation(
            self.chunks, self.dense, self.sparse, self.rrf, self.generator
        )
        
        # Ablation study
        ablation_results = innovative.ablation_study(self.questions)
        
        # Error analysis
        error_analysis = innovative.error_analysis(results)
        
        # Parameter sensitivity (optional - can be slow)
        # param_results = innovative.parameter_sensitivity(self.questions)
        
        return {
            "ablation_study": ablation_results,
            "error_analysis": error_analysis,
            # "parameter_sensitivity": param_results
        }
    
    def save_results(self, results: List[Dict], innovative_results: Dict,
                     output_file="evaluation_results.json"):
        """Save detailed results to JSON"""
        
        # Aggregate metrics
        aggregated = self.metrics.aggregate_results(results)
        
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(results),
            "aggregated_metrics": aggregated,
            "detailed_results": results,
            "innovative_evaluation": innovative_results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved detailed results to {output_file}")
        
        # Also save CSV for easy viewing
        df = pd.DataFrame(results)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV results to {csv_file}")
        
        return aggregated
    
    def print_summary(self, aggregated: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nTotal Questions Evaluated: {aggregated['total_questions']}")
        
        print("\n" + "-"*70)
        print("MANDATORY METRIC:")
        print("-"*70)
        print(f"Mean Reciprocal Rank (MRR):        {aggregated['mean_mrr']:.4f} ± {aggregated['std_mrr']:.4f}")
        
        print("\n" + "-"*70)
        print("CUSTOM METRICS:")
        print("-"*70)
        print(f"ROUGE-L (Answer Quality):          {aggregated['mean_rouge_l']:.4f} ± {aggregated['std_rouge_l']:.4f}")
        print(f"NDCG@5 (Retrieval Quality):        {aggregated['mean_ndcg_at_5']:.4f} ± {aggregated['std_ndcg_at_5']:.4f}")
        
        print("\n" + "-"*70)
        print("ADDITIONAL METRICS:")
        print("-"*70)
        print(f"Precision@5:                        {aggregated['mean_precision_at_5']:.4f}")
        print(f"F1 Score:                           {aggregated['mean_f1_score']:.4f}")
        print(f"Exact Match Rate:                   {aggregated['exact_match_rate']:.4f}")
        
        print("\n" + "-"*70)
        print("PERFORMANCE BY QUESTION TYPE:")
        print("-"*70)
        for qtype, stats in aggregated['by_question_type'].items():
            print(f"\n{qtype.upper()} ({stats['count']} questions):")
            print(f"  MRR:     {stats['mean_mrr']:.4f}")
            print(f"  ROUGE-L: {stats['mean_rouge_l']:.4f}")
            print(f"  NDCG@5:  {stats['mean_ndcg_at_5']:.4f}")
        
        print("\n" + "="*70)


def main():
    """Main function to run complete evaluation pipeline"""
    
    # Check if questions exist, if not generate them
    import os
    if not os.path.exists("evaluation_questions.json"):
        print("Questions not found. Generating questions first...")
        from question_generator import QuestionGenerator
        qg = QuestionGenerator()
        qg.generate_questions(n_questions=100)
        print("\n")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(
        questions_file="evaluation_questions.json",
        chunks_file="wiki_chunks.jsonl"
    )
    
    # Run standard evaluation
    results = pipeline.run_evaluation(top_k=10, final_n=5)
    
    # Run innovative evaluation
    innovative_results = pipeline.run_innovative_evaluation(results)
    
    # Save results
    aggregated = pipeline.save_results(results, innovative_results)
    
    # Print summary
    pipeline.print_summary(aggregated)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review evaluation_results.json for detailed results")
    print("  2. Review evaluation_results.csv for tabular view")
    print("  3. Run: python report_generator.py to generate PDF report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
