# Proactive Elderly Safety System for Pre-Impact Fall Anticipation using Temporal Reasoning

## Team
- Vivek Sathish Poojary — Lead, M4 (Decision Engine, Alerting & Integration)
- Prajwal — M3 (Temporal Reasoning Model)
- Praveen — M1 (Data Collection & Pose Estimation)
- Vinay V S — M2 (Feature Engineering & Context Detection)
- Faculty Guide: Prof. Soumya Santhosha

## Pipeline
Camera → M1 (Pose) → M2 (Features) → M3 (Temporal Model) → M4 (Decision Engine & Alerts)

## Structure
- `m1-pose-estimation/` — Pose detection module
- `m2-feature-engineering/` — Motion features + context detection
- `m3-temporal-model/` — LSTM/GRU fall-risk prediction model
- `m4-decision-engine/` — Decision logic, alerting, and full pipeline integration
- `shared/schema.md` — Agreed data formats passed between modules
- `main.py` — Runs the full pipeline end-to-end (owned by M4)

## Setup
Each module has its own README with setup instructions specific to that part.

##rules
-daily updates has to be made here.
-everyone learn what u are doing.
