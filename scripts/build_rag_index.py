"""
scripts/build_rag_index.py — One-time RAG index builder (Phase 3)

Loads the turkish-hospital-medical-articles corpus, chunks it, embeds
with mpnet, and writes a FAISS index + chunk metadata to output_dir.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --output_dir outputs/rag_index --batch_size 128 --force

Requires:
    export HF_TOKEN=hf_...
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import build_corpus, build_index


def parse_args():
    parser = argparse.ArgumentParser(description="Build RAG FAISS index")
    parser.add_argument("--output_dir",  default="outputs/rag_index",
                        help="Where to save index.faiss and chunks.parquet")
    parser.add_argument("--cache_dir",   default=None,
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Embedding batch size (increase if you have more VRAM/RAM)")
    parser.add_argument("--force",       action="store_true",
                        help="Rebuild index even if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BUILD RAG INDEX — Phase 3")
    print("=" * 60)
    print(f"  Output dir : {args.output_dir}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Force      : {args.force}")
    print()

    corpus = build_corpus(cache_dir=args.cache_dir)
    build_index(corpus, output_dir=args.output_dir, batch_size=args.batch_size, force=args.force)

    print("\nDone. Run eval_rag.py to test retrieval.")


if __name__ == "__main__":
    main()
