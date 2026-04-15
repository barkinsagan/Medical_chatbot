"""
scripts/test_retrieval.py — Inspect RAG retrieval quality on sample questions

Loads the FAISS index and prints the top-k retrieved chunks for a handful
of test questions so you can judge whether retrieval is helping or adding noise.

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --n_questions 10 --k 5 --index_dir outputs/rag_index
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import load_index, retrieve
from src.data import load_doktorsitesi


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect RAG retrieval quality")
    parser.add_argument("--index_dir",   default="outputs/rag_index")
    parser.add_argument("--n_questions", type=int, default=5,
                        help="Number of test questions to inspect")
    parser.add_argument("--k",           type=int, default=3,
                        help="Number of chunks to retrieve per question")
    parser.add_argument("--chunk_preview", type=int, default=300,
                        help="How many characters of each chunk to print")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading index...")
    faiss_index, chunks_df, embed_model = load_index(args.index_dir)

    print("Loading test questions...")
    ds = load_doktorsitesi()
    test = ds["test"].select(range(args.n_questions))

    for i, example in enumerate(test):
        question  = example["question_content"]
        reference = example["question_answer"]

        print(f"\n{'='*70}")
        print(f"QUESTION {i+1}")
        print(f"{'='*70}")
        print(f"  {question}")
        print(f"\n  Reference answer (first 200 chars):")
        print(f"  {reference[:200]}")
        print(f"\n  Top-{args.k} retrieved chunks:")

        hits = retrieve(question, faiss_index, chunks_df, embed_model, k=args.k)

        for j, hit in enumerate(hits):
            print(f"\n  [{j+1}] score={hit['score']:.4f} | hospital={hit['hospital']}")
            print(f"       title : {hit['article_title']}")
            print(f"       url   : {hit['url']}")
            print(f"       chunk : {hit['chunk_text'][:args.chunk_preview]}")

        # Verdict hint
        scores = [h["score"] for h in hits]
        avg_score = sum(scores) / len(scores) if scores else 0
        if avg_score >= 0.4:
            verdict = "GOOD  — chunks are semantically close"
        elif avg_score >= 0.25:
            verdict = "WEAK  — chunks loosely related, may add noise"
        else:
            verdict = "POOR  — chunks likely irrelevant"
        print(f"\n  Avg retrieval score: {avg_score:.4f}  →  {verdict}")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("  score >= 0.40  : chunk is semantically close to the question")
    print("  score 0.25–0.40: loosely related, may or may not help")
    print("  score <  0.25  : likely irrelevant noise")
    print()
    print("If most scores are below 0.30, retrieval is the problem —")
    print("the hospital article corpus does not match the Q&A domain well.")


if __name__ == "__main__":
    main()
