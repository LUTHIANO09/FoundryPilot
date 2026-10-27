import tempfile
import unittest
from pathlib import Path

from lesson01.agent import (
    AgentThread,
    ApprovalRequest,
    execute_approved_action,
    resolve_approval,
    run_local_agent,
)


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.knowledge_dir = Path(self.temp_dir.name)
        (self.knowledge_dir / "notes.txt").write_text(
            "Microsoft Foundry supports building AI applications and agents.",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_completes_and_records_retrieval_tool(self):
        thread = AgentThread()

        run = run_local_agent(
            thread,
            "What does Microsoft Foundry support?",
            self.knowledge_dir,
        )

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.thread_id, thread.id)
        self.assertIn("search_knowledge", run.tool_calls)
        self.assertIn("Microsoft Foundry", run.result)

    def test_thread_preserves_messages_across_turns(self):
        thread = AgentThread()

        first_run = run_local_agent(
            thread, "What does Microsoft Foundry support?", self.knowledge_dir
        )
        second_run = run_local_agent(
            thread, "What is in the knowledge base?", self.knowledge_dir
        )

        self.assertEqual(first_run.thread_id, second_run.thread_id)
        self.assertEqual(len(thread.messages), 4)
        self.assertEqual(thread.messages[0]["role"], "user")
        self.assertEqual(thread.messages[2]["role"], "user")

    def test_each_request_has_a_unique_run_id(self):
        thread = AgentThread()

        first_run = run_local_agent(
            thread,
            "What does Microsoft Foundry support?",
            self.knowledge_dir,
        )
        second_run = run_local_agent(
            thread,
            "What is in the knowledge base?",
            self.knowledge_dir,
        )

        self.assertNotEqual(first_run.id, second_run.id)
        self.assertEqual(first_run.thread_id, second_run.thread_id)
        self.assertEqual(first_run.status, "completed")
        self.assertEqual(second_run.status, "completed")

    def test_no_matching_passages_returns_completed_run(self):
        thread = AgentThread()

        run = run_local_agent(
            thread, "What is quantum teleportation?", self.knowledge_dir
        )

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.result, "No relevant passages were found.")
        self.assertEqual(run.retrieved_count, 0)

    def test_missing_knowledge_directory_fails_run(self):
        thread = AgentThread()
        missing_dir = self.knowledge_dir / "missing"

        run = run_local_agent(thread, "What is Foundry?", missing_dir)

        self.assertEqual(run.status, "failed")
        self.assertIn("Local tool failed:", run.result)
        self.assertIn("FileNotFoundError", run.error)

    def test_run_records_observability_metrics(self):
        thread = AgentThread()

        run = run_local_agent(
            thread,
            "What does Microsoft Foundry support?",
            self.knowledge_dir,
        )

        self.assertTrue(run.started_at)
        self.assertTrue(run.finished_at)
        self.assertGreaterEqual(run.duration_ms, 0)
        self.assertGreater(run.retrieved_count, 0)
        self.assertIsNone(run.error)

    def test_pending_approval_blocks_action(self):
        request = ApprovalRequest(description="Publish a report")
        executed = []

        with self.assertRaises(PermissionError):
            execute_approved_action(request, lambda: executed.append(True))

        self.assertEqual(request.status, "pending")
        self.assertEqual(executed, [])

    def test_approved_request_allows_action(self):
        request = ApprovalRequest(description="Publish a report")
        resolve_approval(request, approved=True, decision_note="Reviewed")
        executed = []

        result = execute_approved_action(
            request, lambda: executed.append("published") or "success"
        )

        self.assertEqual(request.status, "approved")
        self.assertTrue(request.decided_at)
        self.assertEqual(request.decision_note, "Reviewed")
        self.assertEqual(executed, ["published"])
        self.assertEqual(result, "success")

    def test_rejected_request_blocks_action(self):
        request = ApprovalRequest(description="Publish a report")
        resolve_approval(request, approved=False, decision_note="Needs changes")
        executed = []

        with self.assertRaises(PermissionError):
            execute_approved_action(request, lambda: executed.append(True))

        self.assertEqual(request.status, "rejected")
        self.assertEqual(executed, [])

    def test_resolved_approval_cannot_be_changed(self):
        request = ApprovalRequest(description="Publish a report")
        resolve_approval(request, approved=True)

        with self.assertRaises(ValueError):
            resolve_approval(request, approved=False)


if __name__ == "__main__":
    unittest.main()
