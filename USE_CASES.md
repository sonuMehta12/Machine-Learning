# USE_CASES.md

Ideas for things to build. Some become projects, some stay ideas. Add freely — an idea
written down badly is worth more than one remembered vaguely.

**How to read a use case:** the important part is not the industry, it's the **task shape**.
Almost every ML product is one of five shapes:

| Shape | Question it answers |
|---|---|
| **Classification** | Which category is this? |
| **Detection** | Where is it, and how many? |
| **Segmentation** | Exactly which pixels? |
| **Forecasting / regression** | What number comes next? |
| **Anomaly detection** | Does this look unlike normal? |

---

# My ideas

## 1. Sales conversion prediction from call transcripts

*Added 4 Aug 2026. From real experience at a previous company.*

**Context:** worked with sales call transcripts. The goal was to find what patterns make a
call successful — offer made, time of call, tone of customer and rep. Never trained a model;
it was all LLM prompting.

**This is actually two problems, and they need different approaches:**

| Question | Type | Use |
|---|---|---|
| Which lead is likely to buy? | Prediction (lead scoring) | Rank the call queue |
| What makes a call successful? | Explanation / causal | Change rep behaviour |

The first works well with standard ML. The second needs **experiments (A/B tests)**, not just
a model on historical data — a model finds correlation, and correlation copied blindly
produces no improvement.

**Task shape:** binary classification → but really a **ranking** problem. You don't need
"will they buy yes/no", you need "sort 500 leads so I call the best 50 first."

**Data:** one row per call. Target = `converted` (1/0).

- *Structured:* time of day, day of week, location, industry, company size, lead source,
  number of prior touches, rep ID
- *From transcript:* talk/listen ratio, question count, was pricing discussed, objection type,
  sentiment, topic
- *From audio:* tone, speaking rate, interruptions

**Model:** gradient boosting (XGBoost / LightGBM) on tabular features. Not a neural net —
boosting usually wins on tabular business data.

**My advantage:** use an LLM to extract structured fields from each transcript
("did they mention budget?", "was a decision-maker present?"), then feed those into the
boosting model. Hybrid LLM-extraction → classical-model. This plays directly to existing
agent skills.

**⚠️ Leakage risks — the thing that would kill this project:**
Ask of every feature: *would I know this before deciding to make the call?*
- `call_duration` — only known **after** the call
- `rep_sent_proposal` — happens *because* it's going well; it's the outcome in disguise
- anything logged after the deal closed

**⚠️ Imbalance:** if ~3% convert, 97% accuracy is achieved by always saying "no."
Needs PR-AUC and threshold tuning, not accuracy.

**⚠️ Ethics/legal:** targeting on location + demographic profile can become discriminatory
targeting. Design around it early.

**Why it's a good project:** it hits *both* Phase 1 danger zones — leakage (Project 1.2) and
class imbalance (Project 1.3) — on a problem I actually understand from the business side.

**Blocker:** no longer have access to that data. Would need a public substitute
(e.g. Kaggle lead-scoring or bank-marketing datasets) or synthetic data.

---

# Industry reference map

Not my ideas — a survey of what's already working, kept for pattern-spotting.

### Manufacturing *(target domain)*
- Defect detection on parts — classification / detection / segmentation
- **Anomaly detection** — train only on good parts, flag deviations. Huge when defect
  examples are rare or expensive to collect.
- Predictive maintenance — sensor time-series predicts failure before it happens
- Robotic pick-and-place — detection + pose estimation

### Construction *(target domain)*
- Site safety — detect missing PPE, workers in restricted zones
- Structural inspection — crack and spalling detection on bridges and buildings
- Progress tracking — compare drone photos over time against the plan
- Equipment utilization — detect and count machinery on site

### Security / surveillance *(target domain)*
- Perimeter intrusion — person/vehicle detection in restricted areas
- Re-identification and tracking across multiple cameras
- Behavioural anomaly — unusual crowd movement, abandoned objects
- ⚠️ Real privacy/legal tension in this domain — "can we" vs "should we"

### Agriculture
- Crop disease detection from leaf photos
- Yield prediction from satellite + weather + soil
- Precision spraying — detect weeds vs crops, spray only weeds
- Livestock health monitoring from gait/behaviour

### Healthcare
- Medical imaging — tumour detection, diabetic retinopathy, skin cancer
- Signal anomaly detection — ECG/EEG
- Drug discovery — predict molecule binding
- Clinical NLP — structure doctors' free-text notes

### Retail / e-commerce
- Recommendation systems
- Visual search — photograph an item, find similar products
- Demand forecasting per store/SKU

### Finance
- Fraud detection — the classic imbalanced classification problem
- Credit scoring
- Algorithmic trading (noisy, hard)

### Energy / environment
- Solar/wind generation forecasting
- Satellite monitoring — deforestation, methane leaks, flood mapping

---

# How to judge an idea

The useful question is **not** "where could AI help?" (everywhere, vaguely).

It's: **"Where does someone currently pay a human to look at something repetitive and decide
pass/fail — or watch a number and guess what happens next?"**

That's where:
- labelled data already half-exists (past human decisions = training set)
- the ROI argument is easy
- the workflow being replaced is obvious

Products that fail usually had a good model and no clear workflow it was replacing.
