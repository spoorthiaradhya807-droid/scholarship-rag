"""Check retrieval quality and find a good 'not enough evidence' cutoff."""
import time

from src.embeddings import EmbeddingError, embed_query
from src.vectorstore import search

# (question, file-name fragments; one must appear in the top 3 results)
IN_SCOPE = [
    ("What documents are required for post matric scholarship?", ["ssp_postmatric_documents_required"]),
    ("Income limit for Arya Vysya fee reimbursement", ["arya_vysya"]),
    ("Which scholarships does AICTE offer for girl students?", ["aicte_pragati"]),
    ("How do I create an SSP account?", ["create_account", "user_guide", "application_user_manual", "portal_screenshot"]),
    ("What is the eligibility for the Ishan Uday scholarship?", ["nsp_ishan_uday"]),
    ("What is the PM-USP central sector scholarship?", ["nsp_pm_usp_csss_faq"]),
    ("Which department gives a scholarship for tailors?", ["sahyadri"]),
    ("What is the income limit for minority fee reimbursement?", ["minority"]),
    ("Is there a scholarship for students with disabilities?", ["disability"]),
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who is the best cricket player of all time?",
    "How do I fix a flat bicycle tyre?",
    "What is the HDFC Parivartan scholarship amount?",
    "What is the Fulbright scholarship deadline for the USA?",
    "How do I reset my Instagram password?",
]


def ask(question: str):
    hits = search(embed_query(question), 3)
    time.sleep(1)
    return hits


def describe(hits):
    return ", ".join(f"{h['metadata']['source']} p{h['metadata']['page']} ({h['similarity']:.2f})" for h in hits)


def main():
    in_scores, out_scores, misses = [], [], 0
    try:
        print("IN-SCOPE QUESTIONS (the right file should be in the top 3)\n")
        for question, expected in IN_SCOPE:
            hits = ask(question)
            sources = [h["metadata"]["source"] for h in hits]
            ok = any(any(e in s for e in expected) for s in sources)
            misses += 0 if ok else 1
            in_scores.append(hits[0]["similarity"])
            print(f"{'PASS' if ok else 'MISS'} | top1 {hits[0]['similarity']:.3f} | {question}")
            print(f"      {describe(hits)}\n")

        print("OUT-OF-SCOPE QUESTIONS (the documents should NOT answer these)\n")
        for question in OUT_OF_SCOPE:
            hits = ask(question)
            out_scores.append(hits[0]["similarity"])
            print(f"top1 {hits[0]['similarity']:.3f} | {question}")
            print(f"      {describe(hits[:1])}\n")
    except EmbeddingError as exc:
        print(f"\nStopped: {exc}")
        return

    print(f"Retrieval: {len(IN_SCOPE) - misses}/{len(IN_SCOPE)} in-scope questions found the right file")
    print(f"Lowest in-scope top-1 score : {min(in_scores):.3f}")
    print(f"Highest out-of-scope top-1  : {max(out_scores):.3f}")


if __name__ == "__main__":
    main()