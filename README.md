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

## Run

Configure the required model credentials in your local environment, then run:

```bash
streamlit run app.py
```

Run contract checks without calling the model:

```bash
python -m evals.run_contract_evals --dry-run
```

## Notes

Do not commit API keys, local vector databases, logs, or evaluation outputs.
