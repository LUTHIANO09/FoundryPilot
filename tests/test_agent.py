import tempfile
import unittest
from pathlib import Path

from lesson01.agent import AgentThread, run_local_agent


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

    def test_no_matching_passages_returns_completed_run(self):
        thread = AgentThread()

        run = run_local_agent(
            thread, "What is quantum teleportation?", self.knowledge_dir
        )

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.result, "No relevant passages were found.")

    def test_missing_knowledge_directory_fails_run(self):
        thread = AgentThread()
        missing_dir = self.knowledge_dir / "missing"

        run = run_local_agent(
            thread, "What is Foundry?", missing_dir
        )

        self.assertEqual(run.status, "failed")
        self.assertIn("Local tool failed:", run.result)


if __name__ == "__main__":
    unittest.main()
