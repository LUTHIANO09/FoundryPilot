
import unittest


def evaluate_groundedness(answer_claims, evidence):
    """Check whether each claim is explicitly supported by the evidence."""
    evidence_lower = evidence.lower()
    return bool(answer_claims) and all(
        claim.lower() in evidence_lower for claim in answer_claims
    )


def evaluate_relevance(answer, question_terms):
    """Simple baseline: check whether an answer addresses question keywords."""
    answer_terms = set(answer.lower().split())
    return bool(question_terms) and bool(
        answer_terms.intersection(term.lower() for term in question_terms)
    )


def evaluate_safety(answer, forbidden_phrases):
    """Flag answers containing explicitly forbidden phrases."""
    answer_lower = answer.lower()
    return not any(
        phrase.lower() in answer_lower for phrase in forbidden_phrases
    )


class EvaluationTests(unittest.TestCase):
    def test_grounded_answer_passes(self):
        evidence = "RAG retrieves relevant information to ground a response."
        claims = ["RAG retrieves relevant information"]
        self.assertTrue(evaluate_groundedness(claims, evidence))

    def test_unsupported_claim_fails_groundedness(self):
        evidence = "RAG retrieves relevant information to ground a response."
        claims = ["RAG guarantees 100% accuracy"]
        self.assertFalse(evaluate_groundedness(claims, evidence))

    def test_relevant_answer_passes(self):
        answer = "RAG retrieves relevant information from a knowledge source."
        self.assertTrue(
            evaluate_relevance(answer, {"RAG", "information"})
        )

    def test_off_topic_answer_fails_relevance(self):
        answer = "Bananas contain potassium and fiber."
        self.assertFalse(
            evaluate_relevance(answer, {"RAG", "retrieval"})
        )

    def test_safe_answer_passes(self):
        answer = "The passages do not provide enough evidence."
        self.assertTrue(evaluate_safety(answer, ["reveal secret"]))

    def test_forbidden_phrase_fails_safety(self):
        answer = "Here is the secret: reveal secret"
        self.assertFalse(evaluate_safety(answer, ["reveal secret"]))


if __name__ == "__main__":
    unittest.main()
