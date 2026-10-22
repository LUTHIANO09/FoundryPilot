import unittest
from pathlib import Path

from lesson01.rag import (
    Chunk,
    chunk_text,
    load_chunks,
    precision_at_k,
    recall_at_k,
    retrieve,
)


class ChunkTextTests(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        chunks = chunk_text(
            "one two three four five six seven eight nine ten",
            chunk_size=4,
            overlap=1,
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "one two three four",
                "four five six seven",
                "seven eight nine ten",
            ],
        )

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=0, overlap=0)

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=4, overlap=4)

    def test_preserves_paragraph_boundaries_when_paragraphs_fit(self):
        text = "First topic has three words.\n\nSecond topic has four words."
        chunks = chunk_text(text, chunk_size=10, overlap=2)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            chunks[0].text,
            "First topic has three words.",
        )
        self.assertEqual(
            chunks[1].text,
            "Second topic has four words.",
        )



class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.relevant = {
            ("notes.txt", 0),
            ("notes.txt", 1),
            ("notes.txt", 2),
        }
        self.retrieved = [
            Chunk("notes.txt", 0, "Relevant passage one."),
            Chunk("other.txt", 3, "Irrelevant passage."),
        ]

    def test_recall_at_k(self):
        self.assertAlmostEqual(
            recall_at_k(self.retrieved, self.relevant),
            1 / 3,
        )

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            precision_at_k(self.retrieved, self.relevant),
            0.5,
        )

    def test_recall_rejects_empty_relevant_set(self):
        with self.assertRaises(ValueError):
            recall_at_k(self.retrieved, set())

    def test_precision_returns_zero_when_nothing_was_retrieved(self):
        self.assertEqual(
            precision_at_k([], self.relevant),
            0.0,
        )

    def test_perfect_retrieval_scores_one_for_both_metrics(self):
        relevant_chunks = [
            Chunk("notes.txt", 0, "Passage one."),
            Chunk("notes.txt", 1, "Passage two."),
        ]
        relevant_ids = {
            ("notes.txt", 0),
            ("notes.txt", 1),
        }

        self.assertEqual(
            recall_at_k(relevant_chunks, relevant_ids),
            1.0,
        )
        self.assertEqual(
            precision_at_k(relevant_chunks, relevant_ids),
            1.0,
        )

class RetrieverBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.chunks = load_chunks(
            Path("lesson01") / "knowledge",
            chunk_size=80,
            overlap=15,
        )
        self.cases = [
            ("What is Microsoft Foundry?", 0),
            ("What does RAG do?", 1),
            ("Why is chunk overlap useful?", 2),
            ("How do keyword and vector search differ?", 3),
            ("What is groundedness?", 4),
            ("What is answer relevance?", 5),
            ("What do Recall at k and Precision at k measure?", 6),
        ]

    def test_known_questions_retrieve_expected_chunks(self):
        for question, expected_index in self.cases:
            with self.subTest(question=question):
                results = retrieve(question, self.chunks, top_k=1)

                self.assertTrue(
                    results,
                    f"No passages retrieved for: {question}",
                )
                self.assertEqual(
                    results[0][0].index,
                    expected_index,
                    f"Unexpected top result for: {question}",
                )

    def test_retrieval_metrics_on_known_question(self):
        question = "What is groundedness?"
        expected_relevant = {("ai103_rag_notes.txt", 4)}

        results = retrieve(question, self.chunks, top_k=1)
        retrieved_chunks = [chunk for chunk, _score in results]

        self.assertAlmostEqual(
            recall_at_k(retrieved_chunks, expected_relevant),
            1.0,
        )
        self.assertAlmostEqual(
            precision_at_k(retrieved_chunks, expected_relevant),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
