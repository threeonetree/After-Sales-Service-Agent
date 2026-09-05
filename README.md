# After-Sales Service Agent

An after-sales assistant for robot vacuums built with LangGraph, Qwen, RAG, and Streamlit.

## Features

- Answers troubleshooting, maintenance, and product questions from a local knowledge base.
- Uses tools for weather, user profiles, and robot usage records.
- Resolves current, previous, or explicit-month record lookups deterministically,
  without spending chat-model quota or letting the model choose another month.
- Generates personalized reports only for one explicitly selected month that
  has a usage record.
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
- Chat protocol: Bailian's OpenAI-compatible multimodal endpoint
- Embedding integration: `DashScopeEmbeddings`

Keep "stop when free quota is exhausted" enabled for both models. When Bailian
returns `403 AllocationQuota.FreeTierOnly`, the UI shows a quota-exhausted
message and stops the request.

## Windows 10 setup

Python 3.10 or newer is supported. The example below uses Python 3.10 in
PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the `DASHSCOPE_API_KEY` already configured in Windows, or copy
`.env.example` to `.env` and replace the placeholder locally. Never commit the
real key.

The default chat endpoint is the mainland China endpoint. If the API Key belongs
to another Bailian region, set `DASHSCOPE_BASE_URL` to that region's
OpenAI-compatible `/compatible-mode/v1` endpoint as well.

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

Install the small development-only dependency and run the offline tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Pytest also collects the repository's existing `unittest.TestCase` tests, so
both styles can coexist. Useful commands while learning:

```powershell
# Run one test file and show each case name
python -m pytest tests/test_personal_data_route.py -v

# Run one specific test method
python -m pytest tests/test_personal_data_route.py::test_current_month_without_record_stops_before_model -v
```

The new routing tests demonstrate three pytest basics:

- `@pytest.fixture`: prepares an isolated temporary CSV for each test.
- `@pytest.mark.parametrize`: runs the same assertion for several questions.
- Plain `assert`: compares the actual response with the expected behavior.

Start with `test_current_month_without_record_stops_before_model`: its comments
separate Arrange (prepare data), Act (call the code), and Assert (check output).
Then read `tests/test_react_agent_routing.py`, which uses the real LangGraph
execution with a scripted model to test tool guards, conversation memory and
user isolation. Pytest blocks network connections during tests, so testing does
not consume model quota or need a real API key.

Personal record and report behavior:

- "查询本月的使用记录" reads the selected user's current-month record directly.
- If that month has no record, the app lists the user's available months and
  does not generate a report.
- "生成使用报告" asks for one month before using the chat model.
- "生成2025年12月使用报告" enters report generation only if that record exists.
- Multi-month reports are not supported yet; this prevents silent date-range
  expansion and invented comparisons.
- An ambiguous date such as "12月" asks for a full year and month. After a
  month-selection prompt, replying "2025-12" resumes the record/report request.
- Switching the selected user starts a new conversation. Report generation uses
  only the approved user's month and does not read older reports as source data.

For an existing installation, this routing update requires no vector-index
rebuild or runtime dependency reinstall. Pull once, optionally install
`requirements-dev.txt` to run pytest, then restart Streamlit.

## Notes

Do not commit API keys, local vector databases, logs, or evaluation outputs.
An index manifest stored inside `chroma_db/` prevents vectors from different
embedding models from being mixed.
