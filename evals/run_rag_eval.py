"""Run the RAG retrieval evaluation against a real Chroma collection.

Usage:
    python evals/run_rag_eval.py
    python evals/run_rag_eval.py --top-k 5 --json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import chromadb

from app.rag.evaluation import evaluate, load_cases, load_jsonl
from app.rag.ingest import ingest_document
from app.skills.knowledge import KnowledgeSearchSkill


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus.jsonl")
CASES_PATH = os.path.join(BASE_DIR, "rag_cases.jsonl")
COLLECTION_NAME = "rag_eval_docs"


def build_collection(path: str):
    client = chromadb.PersistentClient(path=path)

    return client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_corpus(collection, corpus_path: str = CORPUS_PATH) -> int:
    documents = load_jsonl(corpus_path)

    for document in documents:
        ingest_document(
            collection,
            document["text"],
            permission=document["permission"],
            title=document["title"],
            source="eval",
        )

    return len(documents)


def build_runner(collection, top_k: int):
    skill = KnowledgeSearchSkill(collection, None, top_k=top_k)

    def runner(case):
        return skill.search(case.query, case.role)

    return runner


def print_report(report: dict) -> None:
    print(
        f"{'result':<8} {'case':<28} {'rank':<5} {'recall':<7} "
        f"{'retrieved'}"
    )
    print("-" * 100)

    for result in report["results"]:
        if result["leaks"]:
            status = "LEAK"
        elif result["rank"] is None:
            status = "MISS" if result["relevant_titles"] else "SKIP"
        else:
            status = "HIT"

        print(
            f"{status:<8} {result['name']:<28} "
            f"{str(result['rank'] or '-'):<5} "
            f"{result['recall']:<7} "
            f"{', '.join(result['retrieved_titles']) or '-'}"
        )

        for leak in result["leaks"]:
            print(f"{'':<8} ! {leak}")

    print("-" * 100)
    print(
        f"cases={report['cases']} "
        f"top_k={report['top_k']} "
        f"hit_rate={report['hit_rate']} "
        f"recall@{report['top_k']}={report['recall_at_k']} "
        f"mrr={report['mrr']} "
        f"no_result_rate={report['no_result_rate']} "
        f"permission_leaks={report['permission_leaks']}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality and permission safety.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Chroma persistence path, defaults to a temporary directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of sources returned to the LLM (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw report as JSON",
    )
    args = parser.parse_args(argv)

    temporary_path = None
    path = args.path

    if path is None:
        temporary_path = tempfile.mkdtemp(prefix="rag_eval_")
        path = temporary_path

    try:
        collection = build_collection(path)
        document_count = ingest_corpus(collection)
        cases = load_cases(CASES_PATH)
        report = evaluate(
            build_runner(collection, args.top_k),
            cases,
            top_k=args.top_k,
        )
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)

    print(f"corpus documents: {document_count}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    return 1 if report["permission_leaks"] else 0


if __name__ == "__main__":
    sys.exit(main())
