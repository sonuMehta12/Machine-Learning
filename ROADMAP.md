# From LLM App Developer → ML Engineer Who Trains and Ships Models

**Personalized roadmap for Sonu · Target: 6–12 months · Started August 2026**

---

## Read this part first

You already have something most people starting this path don't: you can write production software, you understand APIs and deployment, and you've shipped systems that use models. That is roughly 40% of the job of an ML engineer, and it's the half that takes longest to learn.

What you're missing is a specific, learnable set of skills:

1. **The math vocabulary** — not to derive proofs, but so that "the gradients exploded" or "your loss is NaN" or "this needs a different loss function" are diagnosable statements rather than mysteries.
2. **The data discipline** — the actual hard part of production ML. Labeling, splitting, leakage, class imbalance, distribution shift. Models are commodities; datasets are not.
3. **The training loop** — writing one from scratch once, then knowing what every knob does.
4. **The evaluation instinct** — knowing that 99% accuracy on a defect dataset with 1% defects means your model learned nothing.
5. **The deployment path for weights, not APIs** — quantization, ONNX, batching, latency budgets, monitoring for drift.

### The one rule that decides whether this works

**Ratio: 30% reading, 70% running code.** The failure mode for someone in your position is watching 40 hours of lectures and feeling like you understand, then freezing at a blank `train.py`. Every phase below ends with a thing you build. If you skip the build, skip the phase — it did nothing for you.

### How to use this document

- Phases are sequential, but the **math track runs in parallel** the whole way through. Don't stop everything for two months to do math; you'll quit.
- Weeks are estimates for ~8–10 hours/week. Compress or stretch freely. Since you're not time-bounded, **prioritize completing the project over hitting the week number.**
- Each phase has a **"You're done when"** gate. It's a behavioral test, not a feeling. Be strict with yourself — this is the single most important part of the document.
- One repo per project, on GitHub, with a real README. By month 10 this is your portfolio and your proof.

### The honest cost of "not superficial"

You said you want skills that let you train and deploy production models, not surface-level skills. That means three things you should agree to now:

- **You will write training loops by hand before using high-level libraries.** Slower for the first two months, dramatically faster forever after.
- **You will spend more time on data than on models.** This feels wrong and it's correct.
- **You will read some math.** Not proofs. But you will sit with the chain rule until backprop is obvious rather than magic.

---

# The Parallel Track: Mathematics (runs Months 1–5)

**Budget: 25–30% of your study time. Roughly 2–3 hours per week. Do not front-load this.**

You are a beginner in math and willing to learn — good. But math learned without a model to apply it to does not stick. So this track is deliberately paced to stay *slightly ahead* of what you need in the main track.

### What you actually need (and what you don't)

| Need | Depth required | Why |
|---|---|---|
| Linear algebra — vectors, matrices, matmul, dot products | Solid intuition + can compute | Every tensor operation. Understanding shapes is 50% of debugging PyTorch. |
| Calculus — derivatives, chain rule, partial derivatives, gradients | Solid intuition | Backpropagation. Learning rates. Why gradients vanish. |
| Probability — distributions, conditional probability, Bayes, expectation | Solid intuition | Loss functions, calibration, why cross-entropy is what it is. |
| Statistics — mean/variance, sampling, hypothesis testing basics | Working knowledge | Evaluation, confidence in results, A/B testing models. |
| Optimization — gradient descent, convexity (conceptually), momentum | Conceptual | Choosing optimizers and schedules. |
| Information theory — entropy, KL divergence, cross-entropy | Light conceptual | Understanding the default loss functions. |
| Matrix calculus, measure theory, proofs, functional analysis | **Skip** | Research territory. Not your goal. |

### Math resources, in order

**Month 1 — Linear algebra intuition**

- **[3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)** (15 videos, ~4 hours total). Watch the whole series. This is the single highest-leverage 4 hours in this entire document. It replaces intuition you'd otherwise spend a semester building.
- **[Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra)** — do the exercises for vectors, matrix multiplication, and transformations only. You need to *compute*, not just *watch*.
- Practice: in NumPy, implement matrix multiply with loops, then with `@`. Verify they match. Reshape tensors until shape errors stop scaring you.

**Month 2 — Calculus intuition**

- **[3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus)** (12 videos). Focus hard on chain rule.
- **[3Blue1Brown — Neural Networks series](https://www.3blue1brown.com/topics/neural-networks)** — chapters 3 and 4 (backpropagation, and backprop calculus). Watch these twice.
- Practice: by hand, on paper, compute the derivative of a 2-layer network with one neuron each. Then verify against PyTorch autograd. Do this until it's boring.

**Month 3 — Probability and statistics**

- **[StatQuest with Josh Starmer](https://www.youtube.com/@statquest)** — the "Statistics Fundamentals" playlist. Starmer is aggressively beginner-friendly and genuinely rigorous. Also do his playlists on ML concepts as you hit them.
- Specifically watch: probability vs likelihood, expected value, normal distribution, maximum likelihood, cross-entropy, R², confusion matrix, ROC/AUC, precision/recall.

**Months 4–5 — Consolidation (reference, not cover-to-cover)**

- **[Mathematics for Machine Learning](https://mml-book.github.io/)** — Deisenroth, Faisal, Ong. Free PDF. Read **Chapter 5 (Vector Calculus)** and **Chapter 6 (Probability and Distributions)**. Skim the rest as reference. Do not attempt to read this book front to back as a beginner — you will stall.
- Optional if you want more structure: **[Imperial College — Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning)** on Coursera. Three courses, well-taught, but the 3Blue1Brown + StatQuest combo covers most of it faster.

**Math track gate:** You can explain, out loud and without notes, why a neural network's gradient is a product of derivatives, and what happens to that product when the network gets deep. If you can do that, the math track has done its job.

---

# Phase 0 — Environment and Orientation
**Week 1 · ~8 hours**

Short phase. The goal is to remove all friction so that later, "run an experiment" costs you 30 seconds, not 30 minutes.

### Do this

1. **Local setup.** Python 3.11+, `uv` or `conda` for environments, VS Code or Cursor with the Jupyter extension. Install `numpy pandas matplotlib scikit-learn jupyterlab`.
2. **PyTorch.** Install PyTorch — CPU build if you're on a Mac without CUDA (MPS backend works for small models), otherwise the CUDA build. Verify: `torch.cuda.is_available()` or `torch.backends.mps.is_available()`.
3. **Cloud GPU account.** You said you're willing to pay — set it up now, before you need it. Options, in the order I'd try them:
   - **[Google Colab](https://colab.research.google.com/)** free tier for learning; **Colab Pro** (~$10/mo) when free-tier disconnects start annoying you.
   - **[Kaggle Notebooks](https://www.kaggle.com/code)** — 30 free GPU hours/week, no card required, and gets you near the datasets. Excellent value; start here.
   - **[RunPod](https://www.runpod.io/)** or **[Lambda](https://lambda.ai/)** — rent an A100/H100 by the hour ($0.50–$2/hr) when you need a real multi-hour training run. You won't need this until Phase 4.
   - Budget guidance: expect **$0 for months 1–3**, **$10–30/mo for months 4–8**, occasional **$20–50 one-off** for capstone training runs.
4. **Experiment tracking account.** Sign up for **[Weights & Biases](https://wandb.ai/)** (free personal tier). You'll start using it in Phase 3, but having it now means no excuse later.
5. **GitHub repo.** Create `ml-journey`. One folder per phase. Commit every session, even broken code. This becomes evidence of trajectory, which matters more than any certificate.

### Also do this — it will save you months

Read **[Chapter 1 of Hands-On Machine Learning (3rd ed.)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** — "The Machine Learning Landscape." It's ~30 pages and gives you the taxonomy (supervised/unsupervised, classification/regression, batch/online, instance/model-based) that makes every other resource easier to parse.

### You're done when
You can open a terminal, create a fresh environment, launch Jupyter, `import torch`, and train scikit-learn's iris classifier — in under five minutes, without googling anything.

---

# Phase 1 — Classical ML and the Data Stack
**Weeks 2–9 · ~8 weeks**

**Why this before deep learning:** because 80% of your future debugging is data debugging, and classical ML teaches you data discipline with fast feedback loops (seconds, not hours). Also because a shocking number of "AI problems" are correctly solved by gradient boosting, and knowing when *not* to use a neural network is a senior skill.

Skipping this phase is the single most common mistake for people coming from software into ML. Don't.

### Concepts to own

- The full supervised learning workflow: problem framing → data → features → model → evaluate → iterate
- Train/validation/test splits, and **why** — plus cross-validation, stratification
- **Data leakage** — the #1 cause of models that work in your notebook and fail in production. Learn to hunt it.
- Overfitting/underfitting, bias-variance tradeoff, regularization (L1/L2)
- Core algorithms and their intuition: linear/logistic regression, decision trees, random forests, gradient boosting (XGBoost/LightGBM), k-NN, SVM, k-means, PCA
- **Evaluation metrics and when each lies to you:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix. Specifically: why accuracy is useless for defect detection.
- Class imbalance: resampling, class weights, threshold tuning
- Feature engineering, scaling, encoding categoricals, handling missing data
- Hyperparameter tuning: grid search, random search, Optuna

### Primary resources

| Resource | What it's for | Notes |
|---|---|---|
| **[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)** — Andrew Ng, DeepLearning.AI/Stanford (3 courses) | Conceptual backbone | The best paced explanation of ML fundamentals that exists. Audit free, or pay for certificate. ~2 months at your pace, but you can move faster. |
| **[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** — Aurélien Géron (2022) | Your primary book | **Chapters 1–9.** The 3rd edition is still current as of 2026. Buy the physical copy. [Free notebooks on GitHub](https://github.com/ageron/handson-ml3) — run every one. |
| **[StatQuest ML playlist](https://www.youtube.com/@statquest)** | When a concept won't click | Go here whenever Géron or Ng loses you. |
| **[Kaggle Learn](https://www.kaggle.com/learn)** — Intro to ML, Intermediate ML, Feature Engineering, Data Cleaning | Fast hands-on drills | Each is 3–5 hours. Do all four. |

**Note on the book:** Géron uses TensorFlow/Keras in the second half. That's fine — read Chapters 1–9 (classical ML with scikit-learn, which is framework-neutral) and then **stop**. We're going PyTorch for deep learning, because that's what industry and the research ecosystem use, and because it's what you'll find on Hugging Face.

### Projects — build all three

**1.1 — Tabular baseline (Week 3–4)**
Pick a real dataset with business meaning — e.g. [UCI Machine Learning Repository](https://archive.ics.uci.edu/) or a Kaggle Playground competition. Build a full pipeline: EDA → cleaning → feature engineering → 3 models compared → hyperparameter tuning → held-out test evaluation → written analysis of errors.
*Deliverable: a notebook plus a 1-page README explaining what you'd tell a stakeholder.*

**1.2 — Deliberately break it (Week 5)**
Take project 1.1 and **introduce data leakage on purpose.** Fit your scaler on the full dataset before splitting. Include a feature computed using the target. Watch your validation score jump to something absurd. Then fix it and watch it fall. Write down what you saw.
*This exercise is worth more than three lectures. Most people never do it.*

**1.3 — Imbalanced classification (Weeks 6–8)**
Find a dataset with ~1–5% positive class (credit card fraud on Kaggle is the classic; a predictive-maintenance dataset is closer to your goals). Get a model that's actually useful. You will discover that accuracy is a lie, that you need PR-AUC, that threshold selection is a business decision, and that class weights matter.
*This is a direct rehearsal for industrial defect detection, where defects are rare. Take it seriously.*

### You're done when
Given any CSV and a target column, you can produce a defensible model with an honest error analysis in an afternoon — and you can explain to a non-technical person why your model's 94% accuracy is either impressive or meaningless.

---

# Phase 2 — Deep Learning Foundations (PyTorch, From Scratch)
**Weeks 10–19 · ~10 weeks**

This is the phase that closes your gap. You currently *call* models. After this phase you *build* them.

**Non-negotiable:** you write a training loop from scratch, by hand, before touching any high-level trainer. This is the difference between the skills you have and the skills you said you want.

### Concepts to own

- Tensors, autograd, computational graphs — what `.backward()` actually does
- The anatomy of a training loop: forward → loss → `zero_grad` → backward → `step`
- Neurons, layers, activation functions (ReLU, GELU, sigmoid, softmax) and why each exists
- Loss functions: MSE, cross-entropy, BCE — and how to choose
- Optimizers: SGD, momentum, Adam, AdamW — and what each is compensating for
- Learning rate schedules, warmup, the LR-finder technique
- Batch size, epochs, and their interaction with learning rate
- Initialization, batch/layer normalization, dropout, weight decay
- Vanishing/exploding gradients, residual connections
- The convolution operation — kernels, stride, padding, receptive field, pooling
- Transfer learning and fine-tuning — the workhorse technique of applied CV
- Data augmentation
- Reading and debugging training curves: what overfitting looks like, what a too-high LR looks like, what a dead network looks like

### Primary resources — do these in this order

**Step 1 (Weeks 10–13): [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** — Andrej Karpathy, free on YouTube.

Start with **"The spelled-out intro to neural networks and backpropagation: building micrograd."** You will build an automatic differentiation engine from nothing, in ~100 lines of Python. Then "makemore" parts 1–3.

**Type every line with him. Pause. Do not watch passively.** Budget 3–4 hours per 2-hour video. When you finish micrograd, backprop will never be mysterious again, and you'll be able to read PyTorch's source and recognize what it's doing.

This is, in my view, the best deep learning teaching material ever produced, and it's free.

**Step 2 (Weeks 13–16): [PyTorch official tutorials](https://pytorch.org/tutorials/)** + **[Zero to Mastery: Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)** (Daniel Bourke, free online book + YouTube).

Bourke's course is code-first and long-form; it's the most practical PyTorch onboarding available. Do sections 00–04 minimum (fundamentals → workflow → classification → computer vision → custom datasets).

**Step 3 (Weeks 16–19): [Practical Deep Learning for Coders](https://course.fast.ai/)** — fast.ai, Jeremy Howard. Free, 9 lessons, ~90 min each. Companion book free at [the fastbook repo](https://github.com/fastai/fastbook).

Deliberately placed *after* Karpathy. fast.ai is top-down — it shows you results in lesson 1 and explains later. That's the right approach *once you already know what's underneath*, and it'll teach you a working practitioner's habits (LR finding, progressive resizing, sensible defaults) that pure-PyTorch courses skip.

### Reference books (for depth, when a topic needs it)

- **[Understanding Deep Learning](https://udlbook.github.io/udlbook/)** — Simon Prince, MIT Press. Free PDF. Beautifully illustrated, mathematically careful without being punishing. **The best reference book for your level.** Use it topic-by-topic.
- **[Dive into Deep Learning (d2l.ai)](https://d2l.ai/)** — free, interactive, every concept has runnable PyTorch code alongside the math. Great as a second explanation when one source doesn't land.

### Projects — build all four

**2.1 — micrograd, from memory (Week 13)**
A week after finishing Karpathy's video, rebuild the autograd engine from scratch **without rewatching**. Struggle. Check yourself only when stuck. This is the difference between having watched and knowing.

**2.2 — MNIST/FashionMNIST from raw PyTorch (Week 14)**
No `nn.Sequential` shortcuts, no Lightning. Write the `Dataset`, the `DataLoader`, the model class, the training loop, the validation loop, the metric logging. Plot the loss curves. Then deliberately break things: set LR to 10, set it to 0.000001, remove the `zero_grad()`. Observe each failure mode and write down its signature.

**2.3 — Your pet classifier (Weeks 15–17)**
The example you mentioned. Do it properly, and do it **twice**:
- First: train a small CNN from scratch on the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 breeds, ~7,400 images). Note your accuracy.
- Then: fine-tune a pretrained ResNet-50 or EfficientNet on the same data. Note the enormous gap.
- Write up *why* transfer learning wins. That understanding is the foundation of everything you'll do in Phase 3.

**2.4 — Custom dataset, end to end (Weeks 18–19)**
Collect or scrape your own small image dataset (300–1,000 images, 3–5 classes — something you care about). Label it. Split it. Train it. This is your first taste of the real bottleneck: **data collection and labeling is most of the work.** Feel that pain now, when the stakes are zero.

### You're done when
You can write a complete PyTorch training loop from a blank file, without reference. You can look at a loss curve and diagnose the problem. You can fine-tune a pretrained model on a custom dataset and get a sensible result.

**At this point you have crossed the gap you described.** Everything after this is specialization and production hardening.

---

# Phase 3 — Computer Vision Deep Dive
**Weeks 20–31 · ~12 weeks**

Now we go directly at your three target use cases. The order below is deliberate: each task type builds on the previous, and each has a lower labeling cost than the next.

### The four vision task types — know which one your problem is

| Task | Output | Labeling cost | Your use case |
|---|---|---|---|
| **Classification** | One label per image | Cheapest | "Is this part defective?" |
| **Object detection** | Boxes + labels | Moderate | "Where are the cracks, and how many?" · "Is a person in the restricted zone?" |
| **Segmentation** | Per-pixel mask | Expensive | "What's the exact area of spalling?" |
| **Anomaly detection** | Anomaly score + heatmap | Cheapest (normal images only!) | "Flag anything that doesn't look like a good part" |

**This table is the most practically important thing in this phase.** Choosing the right task type for a business problem is where junior and senior ML engineers diverge — and note that anomaly detection needs *only good samples* to train, which is often the difference between a feasible and infeasible industrial project.

### Sub-phase 3A — Classification and CNN architectures (Weeks 20–22)

- **[Stanford CS231n](https://cs231n.stanford.edu/)** — lecture notes are free and are the canonical CV reference. Read the notes on convolutional networks, training neural networks (parts 1–3), and transfer learning. Lecture videos are on YouTube.
- Understand the architecture lineage and *why each innovation happened*: LeNet → AlexNet → VGG → ResNet (residual connections) → EfficientNet (compound scaling) → Vision Transformer (ViT) → ConvNeXt.
- Learn **[timm](https://huggingface.co/docs/timm/)** (PyTorch Image Models) — the standard library for pretrained vision backbones. Hundreds of architectures, one API. You'll use this constantly.
- Practice: benchmark 4 backbones on your Phase 2 pet dataset. Compare accuracy vs. inference latency vs. model size. **Start thinking about that tradeoff now** — it's the central tension in edge deployment.

### Sub-phase 3B — Object detection (Weeks 23–26)

- Concepts: bounding boxes, IoU, anchor boxes, non-max suppression, mAP@0.5 and mAP@0.5:0.95, one-stage vs. two-stage detectors.
- **[Ultralytics YOLO](https://docs.ultralytics.com/)** — the practical industry standard. As of 2026 the current model is **YOLO26** (released January 2026), which added NMS-free end-to-end inference and notably faster CPU performance — relevant for edge deployment on a factory floor. YOLO11 remains widely used and well-documented; either is a fine starting point.
  - ⚠️ **Licensing:** Ultralytics YOLO is AGPL-3.0. Free for learning and open-source, but **commercial deployment requires a paid license.** Know this before you build a client product on it. Permissive alternatives: [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [torchvision's detection models](https://pytorch.org/vision/stable/models.html#object-detection), or [Hugging Face Transformers' DETR](https://huggingface.co/docs/transformers/model_doc/detr).
- **[Roboflow](https://roboflow.com/)** — dataset management, annotation, augmentation, format conversion. Their [blog and YouTube channel](https://blog.roboflow.com/) are excellent free applied-CV teaching material.
- **Annotation tools:** [CVAT](https://www.cvat.ai/) (open source, powerful) or [Label Studio](https://labelstud.io/). Learn one properly. You will label data. Everyone labels data.

**Project 3.1 — Construction defect detector (Weeks 24–26)**
Your first target use case. Datasets to start from:
- [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) — 56,000+ annotated concrete crack images (bridge decks, walls, pavements), cracks from 0.06mm to 25mm, deliberately including shadows, surface roughness and debris. **CC BY 4.0 — commercially usable.** ~500 MB.
- Concrete Crack Images for Classification (Özgenel) — ~40,000 images, on Mendeley Data
- Search Roboflow Universe for "crack detection" and "construction defect" — many community datasets with boxes already drawn

Train a detector, evaluate with mAP, do **error analysis**: look at your false negatives with your own eyes. What kinds of cracks does it miss? Thin ones? Low contrast? Shadows mistaken for cracks? This looking-at-failures habit is what separates people who ship working models from people who ship benchmarks.

### Sub-phase 3C — Segmentation and anomaly detection (Weeks 27–31)

**Segmentation:**
- Concepts: semantic vs. instance vs. panoptic segmentation; U-Net; Dice loss and IoU loss; why per-pixel labels are expensive.
- **[Segment Anything (SAM 2)](https://ai.meta.com/sam2/)** — use it as an *annotation accelerator*. Click a defect, get a mask. This can cut labeling time by 10x and is a genuinely important practical trick.
- **[segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)** — clean library, many architectures and encoders.

**Anomaly detection — pay special attention here:**

This is likely the highest-value technique for your industrial use case, and most self-taught people never learn it. The premise: train **only on defect-free images**, then flag anything statistically unlike them. No defect labels needed.

- **[Anomalib](https://github.com/open-edge-platform/anomalib)** — Intel's open-source library, Apache 2.0, built on PyTorch Lightning. 20+ algorithms behind one API, auto-downloads benchmark datasets, produces standardized metrics. Learn PatchCore, PaDiM, and EfficientAD specifically.
- **[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)** — the benchmark dataset. 5,000+ high-res images across 15 object and texture categories, with pixel-precise anomaly annotations. There is also a harder **[MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)** for when you want a real challenge.
  - ⚠️ **License: CC BY-NC-SA 4.0 — non-commercial use only.** Perfect for learning and portfolio work; you cannot build a commercial product on it. Get in the habit of checking dataset licenses now — it's a real constraint in industrial work, and it's the kind of thing that quietly kills projects late.
- Other industrial datasets: **VisA** (Amazon, 10,821 images), **NEU Surface Defect Database** (steel, 6 defect classes), **DAGM 2007** (synthetic textures), **[Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)** (Kaggle, real production data with segmentation masks).

**Project 3.2 — Industrial defect detection, both ways (Weeks 28–31)**
Take one MVTec AD category. Solve it **twice**:
1. As supervised classification/segmentation using the labeled defects.
2. As unsupervised anomaly detection using only the good images.

Compare AUROC, per-pixel localization quality, and — critically — **how much labeling each approach required**. Write a short decision memo: given a client with 10,000 good parts and 50 known defects, which approach do you recommend and why?

**That memo is the artifact that makes you employable.** It demonstrates judgment, not just capability.

### You're done when
Given a photo dataset and a business question, you can correctly identify which of the four task types applies, pick an appropriate model, train it, and report metrics that a domain expert would find credible — plus explain the labeling budget your approach implies.

---

# Phase 4 — Production, MLOps, and Deployment
**Weeks 32–41 · ~10 weeks**

You know software deployment. This phase is about what's *different* when the artifact is a set of weights: models degrade silently, correctness is statistical, and reproducibility is genuinely hard.

Your existing engineering background makes this phase go faster than it would for most people. Lean into that advantage.

### Concepts to own

**Experiment discipline**
- Tracking runs, params, metrics, artifacts: **[Weights & Biases](https://wandb.ai/)** or **[MLflow](https://mlflow.org/)**
- Data and model versioning: **[DVC](https://dvc.org/)**, or [Hugging Face Hub](https://huggingface.co/docs/hub/) for both datasets and model weights
- Reproducibility: seeds, environment pinning, deterministic ops — and why you'll still never get bit-exact reproducibility on GPU

**Inference optimization** (this is where a lot of the real engineering lives)
- **[ONNX](https://onnx.ai/) / [ONNX Runtime](https://onnxruntime.ai/)** — the portable format. Export from PyTorch, run anywhere. Learn this properly; it's the lingua franca of model deployment.
- **[TensorRT](https://developer.nvidia.com/tensorrt)** for NVIDIA hardware; **[OpenVINO](https://docs.openvino.ai/)** for Intel CPUs and edge — highly relevant for factory-floor deployments
- **Quantization** (FP32 → FP16 → INT8), pruning, knowledge distillation. Expect 2–4x speedup for ~1% accuracy loss. Learn to measure that tradeoff honestly.
- Batching, dynamic batching, and the latency/throughput tradeoff

**Serving**
- FastAPI + PyTorch for a simple service (you'll find this easy)
- **[NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)** for serious multi-model serving
- **[TorchServe](https://pytorch.org/serve/)**, [BentoML](https://bentoml.com/), [LitServe](https://lightning.ai/docs/litserve/)
- Edge deployment: Jetson, Raspberry Pi + Coral, or an industrial PC. Relevant because factory-floor vision usually can't round-trip to the cloud.
- Docker + GPU containers (`nvidia-container-toolkit`)

**Monitoring and the lifecycle**
- Data drift, concept drift, and how models rot. **[Evidently AI](https://www.evidentlyai.com/)** is a good open-source tool.
- Shadow deployment, canary releases, A/B testing models
- Feedback loops and retraining triggers — closing the loop from production failures back into your training set
- Human-in-the-loop review for low-confidence predictions (essential in defect inspection, where a false negative may ship a broken part)

### Primary resources

| Resource | Why |
|---|---|
| **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)** — Chip Huyen (O'Reilly, 2022) | **Read this cover to cover.** The best book on production ML thinking. Based on her Stanford CS 329S course. Free summaries and resources at [github.com/chiphuyen/dmls-book](https://github.com/chiphuyen/dmls-book). |
| **[Made With ML](https://madewithml.com/)** — Goku Mohandas | Free, hands-on MLOps course. Testing, CI/CD, and orchestration for ML. Excellent complement to Huyen's more conceptual book. |
| **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)** | Free course focused specifically on the gap between "model works in notebook" and "model works in production." Materials are from 2022 but the lessons are durable. |
| **[Machine Learning in Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)** — DeepLearning.AI | Optional. Good structure if you want a guided path; skippable if you're doing Huyen + Made With ML. |
| **[AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)** — Chip Huyen (2025) | You'll find this the *easiest* book here, because it's your current job. Read it for the vocabulary that connects your LLM work to your new model-training work. |

### Project 4.1 — Productionize Phase 3 (Weeks 36–41)

Take your defect detector and make it real:

- [ ] Model versioned and tracked in W&B or MLflow, with every experiment logged
- [ ] Exported to ONNX, benchmarked FP32 vs. FP16 vs. INT8 — with a table of latency, size, and accuracy for each
- [ ] Served behind a FastAPI endpoint, Dockerized, with health checks
- [ ] Load-tested: what's your p50 and p99 latency? What's your throughput ceiling?
- [ ] A drift monitor that alerts when incoming image statistics shift
- [ ] A basic feedback UI: reviewer marks predictions right/wrong, corrections flow to a retraining dataset
- [ ] A one-command retraining pipeline
- [ ] A README a stranger could follow to run the whole thing

### You're done when
You can take a trained model from a notebook to a monitored, containerized, optimized service — and you can answer "what happens when this model starts getting worse?" with a specific mechanism rather than a shrug.

---

# Phase 5 — Capstone and Specialization
**Weeks 42–52 · ~10 weeks**

### Capstone: one system, built as if a client paid for it

Choose one of your three use cases and build it properly, end to end:

1. **Problem framing** — write the spec first. What's the business metric? What's the cost of a false positive vs. a false negative? (For defect inspection, these are wildly asymmetric, and that asymmetry should drive your threshold, your architecture, and your human-review policy.)
2. **Data strategy** — sourcing, labeling protocol, an inter-annotator agreement check, a documented split strategy that reflects deployment reality (e.g. split by *production batch* or *site*, not randomly — otherwise you're leaking).
3. **Baseline first** — the dumbest thing that could work. Classical CV, or a pretrained model zero-shot. Never skip this; it tells you whether ML is even needed.
4. **Iterate** — three model approaches, honestly compared on the same held-out set.
5. **Error analysis** — categorize every failure. Look at the images. Group the failures. Fix the biggest bucket.
6. **Deploy** — optimized, containerized, monitored, per Phase 4.
7. **Document** — model card, dataset card, decision log, known limitations, and a plain-English summary for a non-technical stakeholder.

Publish it. Write the blog post. This single project, done to this standard, is worth more in a job conversation than any certificate in this document.

### Then specialize — pick based on where you want to work

- **Industrial/manufacturing inspection:** deeper anomaly detection, few-shot learning, synthetic data generation for rare defects, hardware/camera/lighting considerations (which matter more than model choice — bad lighting kills more industrial CV projects than bad architectures do).
- **Security/surveillance:** video understanding, multi-object tracking (ByteTrack, BoT-SORT), action recognition, re-identification, real-time streaming pipelines. Also: get comfortable with the privacy and legal dimension — this is a domain where "can we" and "should we" genuinely diverge, and clients will expect you to raise it.
- **Bridging back to your LLM strength:** multimodal models (VLMs), fine-tuning open-weight models (LoRA/QLoRA via [Hugging Face PEFT](https://huggingface.co/docs/peft/)), the [Hugging Face courses](https://huggingface.co/learn) generally. This is the highest-leverage specialization for you specifically, because it compounds with skills you already have — very few people can both train a vision model and architect an agent system.

---

# Quick-Reference Resource Index

### Free courses
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — Karpathy. **Start here for DL.**
- [Practical Deep Learning for Coders](https://course.fast.ai/) — fast.ai
- [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) — Daniel Bourke
- [CS231n: CNNs for Visual Recognition](https://cs231n.stanford.edu/) — Stanford
- [Made With ML](https://madewithml.com/) — MLOps
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Hugging Face Courses](https://huggingface.co/learn)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Paid courses (audit-free on Coursera)
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) — Andrew Ng
- [MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) — DeepLearning.AI

### Books
| Book | Cost | When |
|---|---|---|
| [Hands-On ML, 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Géron | Paid | Phase 1, chapters 1–9 |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — Prince | **Free** | Phase 2 reference |
| [Dive into Deep Learning](https://d2l.ai/) | **Free** | Phase 2 reference |
| [Mathematics for ML](https://mml-book.github.io/) — Deisenroth et al. | **Free** | Math track, ch. 5–6 |
| [Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Huyen | Paid | Phase 4, cover to cover |
| [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) — Huyen | Paid | Phase 4, easy read for you |

### YouTube channels
- [3Blue1Brown](https://www.youtube.com/@3blue1brown) — math intuition
- [StatQuest](https://www.youtube.com/@statquest) — stats and ML concepts, beginner-friendly
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) — building from scratch
- [Roboflow](https://www.youtube.com/@Roboflow) — applied computer vision
- [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) — paper explanations, when you're ready

### Tools by category
- **Training:** PyTorch, PyTorch Lightning, timm, Hugging Face Transformers
- **Vision:** Ultralytics (AGPL — check licensing), torchvision, segmentation_models_pytorch, Anomalib, SAM 2
- **Data:** Roboflow, CVAT, Label Studio, FiftyOne, DVC
- **Tracking:** Weights & Biases, MLflow
- **Serving:** ONNX Runtime, TensorRT, OpenVINO, Triton, FastAPI, BentoML
- **Monitoring:** Evidently AI, Grafana
- **Compute:** Kaggle (free 30 GPU hrs/wk), Colab Pro, RunPod, Lambda

### Datasets for your use cases
- **General vision:** ImageNet, COCO, Oxford-IIIT Pets, CIFAR-10/100
- **Industrial defects:** [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) (non-commercial license), [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), VisA, NEU Surface Defect, DAGM 2007, [Severstal Steel](https://www.kaggle.com/c/severstal-steel-defect-detection)
- **Construction:** [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/), Concrete Crack Images (Özgenel), CrackForest
- **Security:** COCO (person class), MOT Challenge, roboflow universe PPE-detection datasets
- **Discovery:** [Roboflow Universe](https://universe.roboflow.com/), [Hugging Face Datasets](https://huggingface.co/datasets), [Kaggle Datasets](https://www.kaggle.com/datasets), [Papers with Code sota tables](https://huggingface.co/papers)

---

# Timeline at a glance

| Phase | Weeks | Focus | Key deliverable |
|---|---|---|---|
| 0 | 1 | Setup | Working environment |
| 1 | 2–9 | Classical ML | Imbalanced classifier + leakage experiment |
| 2 | 10–19 | Deep learning, PyTorch | micrograd rebuilt + pet classifier both ways |
| 3 | 20–31 | Computer vision | Crack detector + MVTec anomaly detection + decision memo |
| 4 | 32–41 | Production/MLOps | Fully deployed, monitored defect service |
| 5 | 42–52 | Capstone | One client-grade end-to-end system |

**Math track runs alongside, weeks 1–20.**

Compress if you're moving fast. Extend if a project is teaching you a lot — a project that's still teaching you is never a reason to move on.

---

# Habits that decide whether this works

1. **Ship something every week**, even if tiny. Momentum beats intensity.
2. **Keep a learning log.** One markdown file. What you did, what broke, what you figured out. In month 8 you will reread month 2 and be shocked at your progress — and that's the fuel that gets people through the middle.
3. **When stuck for 30+ minutes, change medium.** Read → video → code → someone else's notebook. Different explanation, not more of the same one.
4. **Read other people's code.** Kaggle competition winners publish solutions. Read them. This is how you learn the tricks nobody teaches.
5. **Look at your data. With your eyes.** Every serious ML engineer says this and every beginner skips it. Open the images your model got wrong and stare at them.
6. **Don't chase SOTA.** A well-understood ResNet-50 that you can debug beats a paper-fresh architecture you can't.
7. **Build in public.** Blog posts, GitHub, LinkedIn. Not for vanity — writing forces you to find out what you don't understand.

---

# What to watch out for

- **Tutorial hell.** The trap for exactly your profile. You're a capable engineer, so following tutorials feels comfortable and productive. It isn't. The gate criteria in each phase exist to catch this — take them literally.
- **Skipping Phase 1.** Classical ML feels like a detour from deep learning. It isn't; it's where you learn data discipline cheaply.
- **Framework tourism.** Pick PyTorch. Stay with it for the full year.
- **Certificate collecting.** One good capstone > six certificates. Nobody has ever hired anyone for a Coursera certificate.
- **Comparing yourself to researchers.** You're not trying to publish. You're trying to ship models that work. Different sport, different training.

---

# The first week, concretely

If you want to start today:

1. **Today:** Phase 0 environment setup. Get PyTorch importing. Create the GitHub repo. Sign up for Kaggle and W&B.
2. **This week, 4 hours:** [3Blue1Brown Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra), all 15 videos.
3. **This week, 2 hours:** Chapter 1 of Hands-On ML.
4. **This week, 3 hours:** [Kaggle's Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.
5. **End of week:** commit *something* to the repo. A notebook, notes, anything.

Then Phase 1 starts.

---

*Last updated: 1 August 2026. Resource versions verified as current at that date — YOLO26 released Jan 2026; Hands-On ML 3rd ed. (2022) is still the latest edition; fast.ai's 2022 course remains the current version.*
