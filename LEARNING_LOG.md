# Learning Log

One entry per session. What I did, what broke, what I figured out.

## 2026-08-01 — Phase 0: Environment setup

- Installed `uv`, created a Python 3.12 venv (skipped system Python 3.14 — too new for stable PyTorch/scientific package support).
- Installed numpy, pandas, matplotlib, scikit-learn, jupyterlab.
- Installed PyTorch + torchvision. Confirmed MPS (Apple Silicon GPU) backend works on this M5 chip.
- Trained scikit-learn's iris classifier end to end — Phase 0 done-when gate passed.
- Still open: Kaggle account, Weights & Biases account, GitHub repo push.

## 2026-08-02 — Phase 0: accounts and repo, done

- Created Kaggle account (SonuKumar1223). Learned what it's for: free GPU notebooks (30 hrs/week) and datasets — not needed heavily until Phase 2+.
- Created GitHub repo `sonuMehta12/Machine-Learning`, pushed initial commit (env, README, phase folders with per-phase READMEs).
- Signed up for Weights & Biases, logged in locally via `wandb login` (API key entered directly in terminal, never through chat). Ran a smoke-test script (`phase0-setup/wandb_smoke_test.py`) and confirmed the loss chart renders on the dashboard.
- Already enrolled in Andrew Ng's ML Specialization on Coursera — using that instead of Kaggle Learn for Phase 1's conceptual backbone.
- **Phase 0 is complete.** Next: start Phase 1 (classical ML) alongside Month 1 math track (3Blue1Brown linear algebra).
