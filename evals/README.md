# Tool Contract Evaluations

`tool_contract_cases.json` defines expected tool-call sequences for representative after-sales requests. Personal record requests that can be answered from structured local data are expected to stop before the model, so their expected tool-call list is empty.

Run a schema-only check without calling the model:

```bash
python -m evals.run_contract_evals --dry-run
```

Run the full evaluation after configuring model credentials:

```bash
python -m evals.run_contract_evals
```

The runner writes local JSON results to `evals/results/`, which is excluded from version control.
