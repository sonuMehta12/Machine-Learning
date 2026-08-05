# CLAUDE.md

This repo is a **learning project**, not a production codebase. Sonu is learning machine
learning from zero. Claude Code is the teacher, not just the code writer.

Read these three files before teaching:

| File | What it holds |
|---|---|
| [USER.md](USER.md) | Who Sonu is — background, strengths, gaps, how he learns |
| [TEACHER.md](TEACHER.md) | How to teach him. Deliberately open-ended. |
| [ROADMAP.md](ROADMAP.md) | The full 6–12 month plan, phase by phase |
| [USE_CASES.md](USE_CASES.md) | Product ideas and use cases as they come up — future project candidates |

## Where we are right now

**Phase 1 — Classical ML and the Data Stack.**
Currently learning the Python scientific stack (NumPy → Pandas → Matplotlib) because
*Hands-On ML* assumes it. See [PROGRESS.md](PROGRESS.md) for the live status.

## The goal

Go from "LLM app developer who calls models" → "ML engineer who trains and ships models."
Target domains: industrial defect detection, construction inspection, security/surveillance.

## Repo layout

```
ROADMAP.md              the plan
PROGRESS.md             high-level progress across all phases
USER.md / TEACHER.md    who the student is, how to teach
LEARNING_LOG.md         session-by-session journal (what I did, what broke, what clicked)
phaseN-*/               one folder per roadmap phase
  PROGRESS.md           detailed progress for that phase
Books/                  reference PDFs (gitignored)
```

## Tools and environment

- **Package manager:** `uv` (not pip/conda). Add deps with `uv add <pkg>`, run with `uv run <cmd>`.
- **Python:** 3.12 in `.venv` (system Python is 3.14 — too new for the ML stack, don't use it).
- **Notebooks:** JupyterLab — `uv run jupyter lab`, or launch via `.claude/launch.json` preview.
- **Deep learning:** PyTorch with **MPS** (Apple Silicon GPU). Not CUDA. `torch.backends.mps.is_available()` is `True`.
- **Experiment tracking:** Weights & Biases, logged in locally. Project name: `ml-journey`.
- **Free GPU when needed:** Kaggle Notebooks (30 hrs/week), account `SonuKumar1223`.
- **Remote:** GitHub `sonuMehta12/Machine-Learning`, branch `main`.

## Working rules for Claude

- **Commit every session**, even broken or half-finished work. The git history is evidence of trajectory.
- **Update [PROGRESS.md](PROGRESS.md) and the phase's `PROGRESS.md`** when something is finished.
- **Update [LEARNING_LOG.md](LEARNING_LOG.md)** at the end of a session — what was learned, what was confusing.
- Notebooks are written to disk directly (Write tool), not typed into the browser — JupyterLab's
  auto-closing brackets corrupt simulated keystrokes.
- Don't commit data, model weights, or `Books/` — already in `.gitignore`.
