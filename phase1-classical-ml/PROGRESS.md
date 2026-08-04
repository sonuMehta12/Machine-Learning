# Phase 1 — Progress

**Status: 🔵 In progress** (started 2 August 2026)

---

## Part 0 — The Python data stack (prerequisite)

*Hands-On ML* assumes NumPy, Pandas and Matplotlib are already known. They aren't yet,
so they come first. NumPy especially — it is the foundation under Pandas, scikit-learn,
and PyTorch. Every tensor operation in deep learning is this same model.

### NumPy — the plan

Goal is **not** to memorize functions. Goal is to understand the model underneath, so that
shape errors and performance problems are readable instead of mysterious.

| # | Lesson | The real question it answers |
|---|---|---|
| 1 | Why NumPy exists | Why is a Python list not good enough for numbers? |
| 2 | The ndarray model: buffer + metadata | What *is* an array, physically? Why do some operations copy and some don't? |
| 3 | Axes | What does `axis=0` actually mean, and why does it confuse everyone? |
| 4 | Broadcasting | Why can I add a `(3,1)` to a `(1,4)` and get a `(3,4)`? |
| 5 | Indexing and boolean masks | How do I select the rows I care about — the daily work of data cleaning? |
| 6 | An image is an array | Ties it to the goal: computer vision is NumPy with pictures. |

- [x] Lesson 1 — Why NumPy exists ✅ (2 Aug) — vectorization, boxing, why looping over an array is worse than a list
- [ ] Lesson 2 — The ndarray model
- [ ] Lesson 3 — Axes
- [ ] Lesson 4 — Broadcasting
- [ ] Lesson 5 — Indexing and masks
- [ ] Lesson 6 — An image is an array

### Pandas — not started
### Matplotlib — not started

---

## Part 1 — Classical ML proper

Concepts, resources and the three projects are in the "Phase 1" section of
[ROADMAP.md](../ROADMAP.md).

- [ ] Hands-On ML, Chapters 1–9
- [ ] Andrew Ng ML Specialization (in progress, running in parallel)
- [ ] Project 1.1 — Tabular baseline
- [ ] Project 1.2 — Deliberately break it (data leakage)
- [ ] Project 1.3 — Imbalanced classification

## Gate

> Given any CSV and a target column, produce a defensible model with an honest error
> analysis in an afternoon — and explain to a non-technical person why 94% accuracy is
> either impressive or meaningless.
