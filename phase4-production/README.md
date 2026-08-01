# Phase 4 — Production, MLOps, and Deployment

Weeks 32–41. What's different about deployment when the artifact is weights, not an API call: silent degradation, statistical correctness, hard reproducibility.

Project 4.1 — Productionize the Phase 3 defect detector:
- [ ] Versioned and tracked in W&B or MLflow
- [ ] Exported to ONNX, benchmarked FP32 vs. FP16 vs. INT8
- [ ] Served behind FastAPI, Dockerized, with health checks
- [ ] Load-tested (p50/p99 latency, throughput ceiling)
- [ ] Drift monitor on incoming image statistics
- [ ] Feedback UI: reviewer marks predictions right/wrong → retraining dataset
- [ ] One-command retraining pipeline
- [ ] README a stranger could follow

Done when: you can take a trained model from a notebook to a monitored, containerized, optimized service.

See the "Phase 4" section of [ROADMAP.md](../ROADMAP.md) for resources and detail.
