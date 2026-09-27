# LUMEN — Explainable AI Academic Advisor

Course-recommendation prototype that pairs a **LightGBM** fit score with **SHAP**-driven, plain-language explanations for every suggestion.

## Tech stack

- Python, pandas, scikit-learn
- LightGBM (course-fit model)
- SHAP (explainability)
- Streamlit (interface)

## Quick start

```bash
cd Lumen
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

**macOS:** LightGBM needs OpenMP. If import fails, install it with Homebrew:

```bash
brew install libomp
```

# Train model & generate synthetic data (optional — app does this on first run)
python -m lumen.model.train

# Launch advisor UI
streamlit run app.py
```

## Project layout

| Path | Role |
|------|------|
| `lumen/data/synthetic.py` | **Faker**-generated student database, course catalog, training pairs |
| `lumen/features.py` | Feature engineering for (student, course) pairs |
| `lumen/model/train.py` | Train & persist LightGBM regressor |
| `lumen/explain.py` | SHAP → readable explanation bullets |
| `lumen/recommend.py` | Rank courses for a profile |
| `app.py` | Streamlit app: rankings + explanations side by side |

## Design principles (from project brief)

1. **Explain, don't just rank** — every recommendation includes reasons.
2. **Privacy** — local synthetic data only.
3. **Trust** — transparency for students and advisors, not black-box scores.

Group Cartel · Chandigarh University
