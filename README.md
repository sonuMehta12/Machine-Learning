# ml-journey

Sonu's path from LLM app developer to ML engineer. Full plan in [ROADMAP.md](ROADMAP.md).

## Environment

Managed with [uv](https://docs.astral.sh/uv/). Python 3.12, PyTorch with Apple Silicon MPS acceleration.

```bash
uv sync                 # install/sync dependencies
uv run jupyter lab       # launch Jupyter Lab
uv add <package>         # add a new dependency
```

## Structure

One folder per phase, matching ROADMAP.md:

- `phase0-setup/` — environment and orientation
- `phase1-classical-ml/` — scikit-learn, tabular data, data discipline
- `phase2-deep-learning/` — PyTorch from scratch
- `phase3-computer-vision/` — classification, detection, segmentation, anomaly detection
- `phase4-production/` — MLOps, serving, monitoring
- `phase5-capstone/` — end-to-end capstone project

See [LEARNING_LOG.md](LEARNING_LOG.md) for a running log of what was done, what broke, and what clicked.
