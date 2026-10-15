import os

from openai import APIConnectionError, APIStatusError, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


# Microsoft Foundry configuration.
# Environment variables allow the project to be used with other deployments.
endpoint = os.getenv(
    "AZURE_AI_ENDPOINT",
    "https://olamideibrahim-6254-resource.services.ai.azure.com/openai/v1",
)
deployment_name = os.getenv("AZURE_AI_DEPLOYMENT", "ai103-gpt5mini-test")

DEFAULT_INSTRUCTION = (
    "Answer in one or two concise sentences using simple language "
    "for a beginner."
)


def create_client():
    """Create a Microsoft Foundry client using Microsoft Entra ID."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://ai.azure.com/.default",
    )

    return OpenAI(
        base_url=endpoint,
        api_key=token_provider,
    )


def main():
    instruction = input("How should the AI behave? ").strip()

    if not instruction or instruction.lower() in {"normal", "default"}:
        instruction = DEFAULT_INSTRUCTION

    question = input("\nAsk the AI a question: ").strip()

    if not question:
        print("\nPlease enter a question and run the app again.")
        return

    try:
        client = create_client()

        print("\nContacting Microsoft Foundry...")

        response = client.responses.create(
            model=deployment_name,
            instructions=instruction,
            input=question,
            reasoning={"effort": "low"},
            max_output_tokens=200,
        )

        answer = response.output_text.strip()

        if answer:
            print("\nAI response:")
            print(answer)
        else:
            print("\nThe model did not return visible text.")

        if response.status == "incomplete":
            reason = (
                response.incomplete_details.reason
                if response.incomplete_details
                else "unknown"
            )
            print(f"\nNote: The response was incomplete ({reason}).")
        elif response.status != "completed":
            print(f"\nResponse status: {response.status}")

    except APIConnectionError:
        print(
            "\nCould not connect to Microsoft Foundry. "
            "Check your internet connection and endpoint."
        )
    except APIStatusError as exc:
        print(f"\nMicrosoft Foundry returned HTTP {exc.status_code}.")
        print("Check your authentication, deployment and Azure permissions.")


if __name__ == "__main__":
    main()
