import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lesson01 import app


class MainTests(unittest.TestCase):
    def make_client(self, answer="A noun names a person, place, thing, or idea."):
        client = MagicMock()
        client.responses.create.return_value = SimpleNamespace(
            output_text=answer,
            status="completed",
            incomplete_details=None,
        )
        return client

    def test_normal_uses_default_instruction(self):
        client = self.make_client()

        with patch(
            "builtins.input",
            side_effect=["normal", "What is a noun?"],
        ), patch.object(app, "create_client", return_value=client):
            with redirect_stdout(io.StringIO()):
                app.main()

        client.responses.create.assert_called_once_with(
            model=app.deployment_name,
            instructions=app.DEFAULT_INSTRUCTION,
            input="What is a noun?",
            reasoning={"effort": "low"},
            max_output_tokens=200,
        )

    def test_blank_instruction_uses_default_instruction(self):
        client = self.make_client()

        with patch(
            "builtins.input",
            side_effect=["", "What is a noun?"],
        ), patch.object(app, "create_client", return_value=client):
            with redirect_stdout(io.StringIO()):
                app.main()

        self.assertEqual(
            client.responses.create.call_args.kwargs["instructions"],
            app.DEFAULT_INSTRUCTION,
        )

    def test_blank_question_does_not_create_client(self):
        with patch(
            "builtins.input",
            side_effect=["normal", "   "],
        ), patch.object(app, "create_client") as create_client:
            with redirect_stdout(io.StringIO()) as output:
                app.main()

        create_client.assert_not_called()
        self.assertIn("Please enter a question", output.getvalue())

    def test_response_is_printed(self):
        answer = "A noun names a person, place, thing, or idea."
        client = self.make_client(answer)

        with patch(
            "builtins.input",
            side_effect=["normal", "What is a noun?"],
        ), patch.object(app, "create_client", return_value=client):
            with redirect_stdout(io.StringIO()) as output:
                app.main()

        self.assertIn(answer, output.getvalue())


if __name__ == "__main__":
    unittest.main()