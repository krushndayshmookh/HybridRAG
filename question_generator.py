"""
Question Generator for Hybrid RAG Evaluation
Generates 100 diverse Q&A pairs from Wikipedia corpus using T5-based QG model
"""

import json
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import List, Dict
import re
from tqdm import tqdm


class QuestionGenerator:
    def __init__(self, chunks_file="wiki_chunks.jsonl", model_name="google/flan-t5-base"):
        """
        Initialize question generator with T5 model
        
        Args:
            chunks_file: Path to preprocessed chunks
            model_name: HuggingFace model for QG
        """
        self.chunks = self.load_chunks(chunks_file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Question type distribution
        self.question_types = {
            "factual": 0.4,      # Who, What, When, Where
            "comparative": 0.2,   # Compare, difference, similar
            "inferential": 0.2,   # Why, How, Explain
            "multi-hop": 0.2      # Requires multiple chunks
        }
    
    def load_chunks(self, filename):
        """Load preprocessed chunks from JSONL file"""
        chunks = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks from {filename}")
        return chunks
    
    def select_informative_chunks(self, n_chunks=150):
        """
        Select most informative chunks for question generation
        Prefers longer chunks with more content
        """
        # Score chunks by length and content diversity
        scored_chunks = []
        for chunk in self.chunks:
            text = chunk["text"]
            score = len(text.split())  # Word count
            # Bonus for proper nouns (capitalized words)
            score += len(re.findall(r'\b[A-Z][a-z]+\b', text)) * 2
            # Bonus for numbers (dates, quantities)
            score += len(re.findall(r'\b\d+\b', text))
            scored_chunks.append((score, chunk))
        
        # Sort and select top N
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        selected = [chunk for _, chunk in scored_chunks[:n_chunks]]
        
        print(f"Selected {len(selected)} informative chunks for QG")
        return selected
    
    def generate_factual_question(self, context: str) -> Dict:
        """Generate factual questions (Who, What, When, Where)"""
        prompts = [
            f"Generate a factual question about: {context[:400]}",
            f"What question can be answered from: {context[:400]}",
            f"Create a who/what/when/where question from: {context[:400]}"
        ]
        
        prompt = random.choice(prompts)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "type": "factual"
        }
    
    def generate_comparative_question(self, context: str) -> Dict:
        """Generate comparative questions"""
        prompts = [
            f"Generate a comparison question from: {context[:400]}",
            f"What can be compared in: {context[:400]}",
            f"Create a question about similarities or differences: {context[:400]}"
        ]
        
        prompt = random.choice(prompts)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "type": "comparative"
        }
    
    def generate_inferential_question(self, context: str) -> Dict:
        """Generate inferential questions (Why, How, Explain)"""
        prompts = [
            f"Why or how question from: {context[:400]}",
            f"Generate an explanatory question: {context[:400]}",
            f"Create a reasoning question from: {context[:400]}"
        ]
        
        prompt = random.choice(prompts)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "type": "inferential"
        }
    
    def generate_multihop_question(self, chunks: List[Dict]) -> Dict:
        """Generate multi-hop question requiring multiple chunks"""
        # Select 2-3 related chunks from same URL
        same_url_chunks = {}
        for chunk in chunks:
            url = chunk["url"]
            if url not in same_url_chunks:
                same_url_chunks[url] = []
            same_url_chunks[url].append(chunk)
        
        # Find URL with multiple chunks
        multi_chunk_urls = {url: chs for url, chs in same_url_chunks.items() if len(chs) >= 2}
        
        if not multi_chunk_urls:
            # Fallback to regular factual
            chunk = random.choice(chunks)
            return self.generate_factual_question(chunk["text"])
        
        url = random.choice(list(multi_chunk_urls.keys()))
        selected_chunks = random.sample(multi_chunk_urls[url], min(2, len(multi_chunk_urls[url])))
        
        combined_context = " ".join([c["text"] for c in selected_chunks])[:800]
        
        prompt = f"Generate a complex question requiring multiple pieces of information from: {combined_context}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.generate_answer(question, combined_context)
        
        return {
            "question": question,
            "answer": answer,
            "type": "multi-hop",
            "source_chunks": [c["chunk_id"] for c in selected_chunks]
        }
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate ground truth answer from context"""
        prompt = f"Answer the question based on the context.\n\nContext: {context[:500]}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False  # Greedy for consistency
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def generate_questions(self, n_questions=100, output_file="evaluation_questions.json"):
        """
        Generate N diverse questions according to type distribution
        
        Args:
            n_questions: Total number of questions to generate
            output_file: Output JSON file path
        """
        print(f"\nGenerating {n_questions} questions...")
        
        # Select informative chunks
        selected_chunks = self.select_informative_chunks(n_chunks=min(150, len(self.chunks)))
        
        questions_data = []
        
        # Calculate number of each type
        type_counts = {
            qtype: int(n_questions * ratio)
            for qtype, ratio in self.question_types.items()
        }
        
        # Adjust for rounding
        total = sum(type_counts.values())
        if total < n_questions:
            type_counts["factual"] += (n_questions - total)
        
        print(f"Question distribution: {type_counts}")
        
        question_id = 1
        
        # Generate factual questions
        print("\nGenerating factual questions...")
        for _ in tqdm(range(type_counts["factual"])):
            chunk = random.choice(selected_chunks)
            try:
                qa = self.generate_factual_question(chunk["text"])
                questions_data.append({
                    "question_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "question_type": qa["type"],
                    "source_url": chunk["url"],
                    "source_chunk_id": chunk["chunk_id"],
                    "title": chunk["title"]
                })
                question_id += 1
            except Exception as e:
                print(f"Error generating factual question: {e}")
        
        # Generate comparative questions
        print("\nGenerating comparative questions...")
        for _ in tqdm(range(type_counts["comparative"])):
            chunk = random.choice(selected_chunks)
            try:
                qa = self.generate_comparative_question(chunk["text"])
                questions_data.append({
                    "question_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "question_type": qa["type"],
                    "source_url": chunk["url"],
                    "source_chunk_id": chunk["chunk_id"],
                    "title": chunk["title"]
                })
                question_id += 1
            except Exception as e:
                print(f"Error generating comparative question: {e}")
        
        # Generate inferential questions
        print("\nGenerating inferential questions...")
        for _ in tqdm(range(type_counts["inferential"])):
            chunk = random.choice(selected_chunks)
            try:
                qa = self.generate_inferential_question(chunk["text"])
                questions_data.append({
                    "question_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "question_type": qa["type"],
                    "source_url": chunk["url"],
                    "source_chunk_id": chunk["chunk_id"],
                    "title": chunk["title"]
                })
                question_id += 1
            except Exception as e:
                print(f"Error generating inferential question: {e}")
        
        # Generate multi-hop questions
        print("\nGenerating multi-hop questions...")
        for _ in tqdm(range(type_counts["multi-hop"])):
            try:
                qa = self.generate_multihop_question(selected_chunks)
                questions_data.append({
                    "question_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "question_type": qa["type"],
                    "source_url": selected_chunks[0]["url"],  # First chunk URL
                    "source_chunk_id": qa.get("source_chunks", [selected_chunks[0]["chunk_id"]]),
                    "title": selected_chunks[0]["title"]
                })
                question_id += 1
            except Exception as e:
                print(f"Error generating multi-hop question: {e}")
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Generated {len(questions_data)} questions")
        print(f"✓ Saved to {output_file}")
        
        # Print statistics
        type_stats = {}
        for q in questions_data:
            qtype = q["question_type"]
            type_stats[qtype] = type_stats.get(qtype, 0) + 1
        
        print("\nFinal distribution:")
        for qtype, count in type_stats.items():
            print(f"  {qtype}: {count}")
        
        return questions_data


def main():
    """Main function to generate questions"""
    generator = QuestionGenerator(
        chunks_file="wiki_chunks.jsonl",
        model_name="google/flan-t5-base"
    )
    
    questions = generator.generate_questions(
        n_questions=100,
        output_file="evaluation_questions.json"
    )
    
    print("\n" + "="*60)
    print("Question generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
