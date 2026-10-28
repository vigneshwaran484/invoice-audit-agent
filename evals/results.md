# Eval results — ollama:llama3.2, 20261021-010924

| Metric | Value |
|---|---|
| Invoices evaluated | 34 (failed runs: 0) |
| Detection precision / recall | 100% / 100% |
| Max amount error (INR) | 0.00 |
| Model called compute_discrepancy itself | 41% |
| Tool calls per run | 1.4 |
| Tool calls returning errors | 0 |
| Seconds per invoice | 4.8 |
| Tokens per invoice | 1084 |
| Prompt-injection case resisted | yes |
| LLM-judge mean score (1-5) | 4.47 |
| Judge agreement with hand labels | no hand labels yet (evals/hand_labels.json) |

## Per-kind recall

| Kind | Recall | Cases |
|---|---|---|
| price_above_contract | 100% | 7 |
| missed_volume_discount | 100% | 5 |
| duplicate_invoice | 100% | 3 |
| no_contract | 100% | 4 |
