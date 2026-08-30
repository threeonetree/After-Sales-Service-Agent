# After-Sales Service Agent

An after-sales assistant for robot vacuums built with LangGraph, Qwen, RAG, and Streamlit.

## Features

- Answers troubleshooting, maintenance, and product questions from a local knowledge base.
- Uses tools for weather, user profiles, and robot usage records.
- Generates personalized usage reports.
- Includes tool-contract evaluations for checking tool order and required arguments.

## Project Structure

- `agent/`: LangGraph ReAct agent and tools.
- `rag/`: Chroma-based knowledge retrieval.
- `services/`: weather and user-data services.
- `prompts/`: system and report prompts.
- `evals/`: tool-contract cases and evaluation runner.
- `app.py`: Streamlit application entry point.

## Model configuration

The application calls Alibaba Cloud Model Studio only; it does not run a local
model and has no paid-model fallback.

- Chat: `qwen3.7-flash-2026-07-15`
- Embedding: `qwen3.7-text-embedding`
- Credential: one shared `DASHSCOPE_API_KEY`

Keep "stop when free quota is exhausted" enabled for both models. When Bailian
returns `403 AllocationQuota.FreeTierOnly`, the UI shows a quota-exhausted
message and stops the request.

## Windows 10 setup

Python 3.11 is recommended. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the `DASHSCOPE_API_KEY` already configured in Windows, or copy
`.env.example` to `.env` and replace the placeholder locally. Never commit the
real key.

Run three small live checks for chat, text embeddings, and tool calling:

```powershell
python -m scripts.probe_models
```

Expected result:

```text
[PASS] chat
[PASS] embedding
[PASS] tool_calling
```

The repository does not commit generated Chroma data. Build the local knowledge
index once after first install or whenever the embedding model changes:

```powershell
python -m rag.rebuild_index --yes
```

This command replaces only the generated `chroma_db` index; it keeps all source
files under `data/`. Start the application after the index is ready:

```powershell
streamlit run app.py
```

Run contract checks without calling the model:

```bash
python -m evals.run_contract_evals --dry-run
```

## Notes

Do not commit API keys, local vector databases, logs, or evaluation outputs.
An index manifest stored inside `chroma_db/` prevents vectors from different
embedding models from being mixed.
