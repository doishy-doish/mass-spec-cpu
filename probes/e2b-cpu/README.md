# e2b CPU probe (public-repo, M159a pre-authorized)

Runs the small model + open-source half of the stack on GitHub CPU:
`google/gemma-4-E2B-it` via HF transformers (CPU) + `google-adk==2.15.1`
(the same ADK version the Kaggle harness wheelhouse ships).

## Why this probe (the better one)

Two binary gates block the e2b decision:

1. **ParallelAgent mechanics** (OPEN) — source says concurrency is real
   (`asyncio.TaskGroup`, `parallel_agent.py` 9196B), isolation is
   conversation-history-only (shared session state, shared `/workspace`
   -> worktree isolation required), but the runner is private so the
   **call-cap shared-vs-per-branch** accounting is unproven. This probe
   exercises it behaviorally on CPU: overlap timing (concurrent vs
   serial), shared-counter starvation shape, state/file sharing.
2. **ThinkingTokenBudget payload** (ANSWERED) — Kaggle traces already
   prove `thinking_budget` is not a completion cap (max 2262 observed
   with budget 768). The CPU rerun here only confirms the model loads
   and decodes on CPU (tok/s) — it does NOT replicate vLLM's
   `ThinkingTokenBudgetLogitsProcessor` (TransformersServer path
   differs by design).

So: follow probe 1, carry probe 2 as a health check. That is the
"better probe" order this workflow implements (`both` default).

## What full-stack still needs Kaggle for

The private `swegemma`/`adk-submission`/`adk-eval-core` wheels (PyPI 404,
Kaggle-dataset-only) cannot install here, so the 6-task guard run
(2/6 guards) stays a Kaggle GPU job (M159 explicit-GO). This probe
unblocks the ParallelAgent decision with zero quota.

## Reading the logs

- `PHASE0` env (CPU/RAM/torch/ADK versions).
- `PHASE1` overlap wall: ~5s = CONCURRENT, ~10s = SERIAL (2x5s sleeps).
- `PHASE1` shared-counter: which branch starves first under a 6-call budget.
- `PHASE2` model load + decode tok/s + token counts.
- `VERDICT` JSON at end, also saved to `probes/e2b-cpu/summary.json`
  (uploaded as artifact `e2b-cpu-probe-summary`).
