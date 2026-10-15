# FoundryPilot

**A hands-on Microsoft Foundry learning project for the Microsoft AI-103 certification.**

FoundryPilot is a Python application that sends a user's question and instructions to a deployed model in Microsoft Foundry and displays the model's response. It is being developed incrementally to practise AI application development on Azure.

## Current Features

- Interactive command-line interface.
- Custom instructions that guide the model's behaviour.
- Microsoft Foundry model inference using the OpenAI Python SDK.
- Microsoft Entra ID authentication through Azure Identity.
- Configurable endpoint and deployment name through environment variables.
- Basic handling of connection errors, API errors, and incomplete responses.

## Technology Stack

- Python
- Microsoft Foundry
- Azure Identity
- OpenAI Python SDK
- Microsoft Entra ID

## Project Structure

```text
FoundryPilot/
├── lesson01/
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `.venv` directory is created locally and is intentionally excluded from Git.

## Prerequisites

- Python 3.11 or later.
- An Azure subscription with access to Microsoft Foundry.
- A deployed model that supports the Responses API.
- Permission to authenticate and invoke the deployed model.
- Azure CLI installed and signed in.

## Setup on Windows PowerShell

### 1. Clone the repository

```powershell
git clone https://github.com/LUTHIANO09/FoundryPilot.git
cd FoundryPilot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Authenticate with Azure

```powershell
az login
```

Sign in with an account that has the necessary permissions to access your Foundry resource and invoke the deployed model.

### 5. Configure the model endpoint

```powershell
$env:AZURE_AI_ENDPOINT = "https://YOUR-RESOURCE.services.ai.azure.com/openai/v1"
$env:AZURE_AI_DEPLOYMENT = "YOUR-DEPLOYMENT-NAME"
```

Replace the example values with your own endpoint and deployment name.

### 6. Run the application

```powershell
python .\lesson01\app.py
```

Enter an instruction for the AI, then ask a question. Enter `normal` or `default` to use the default instruction.

## Authentication and Security

- Uses Microsoft Entra ID instead of a hard-coded API key.
- Never commit access tokens, passwords, API keys, or `.env` files.
- Assign only the Azure permissions needed to run the application.
- The endpoint and deployment name are configuration values, not authentication credentials.

## Azure Cost Awareness

Azure resources and model deployments may incur charges depending on their configuration and usage. Monitor your Azure cost dashboard and review deployment pricing before running workloads. A budget alert is a notification, not a spending limit.

## Learning Roadmap

Planned areas of exploration:

- Improve configuration and error handling.
- Ground responses in trusted information using retrieval-augmented generation (RAG).
- Explore model selection and evaluation.
- Build and evaluate AI agents.
- Add tracing, observability, and responsible AI safeguards.
- Explore Azure AI services for text and image analysis.

These are learning goals, not completed features.

## Project Status

**Early-stage learning prototype.**

FoundryPilot is being built step by step alongside preparation for the Microsoft AI-103 certification.

## Author

**LUTHIANO09**

Follow the repository to see the project evolve as new AI development concepts are implemented and tested.
