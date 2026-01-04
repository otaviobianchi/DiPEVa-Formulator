# DiPEVa Formulator (Streamlit) — Integrated

This repository provides a research-oriented formulation screening tool based on Hansen-style metrics:

- **Ra** (Hansen distance)
- **Δδa** where δa = sqrt(δP² + δH²)
- **ΔσL** (surface tension difference)
- **Π (article-style, no weights)**: Π = Δδa/δa,max + ΔσL/σL,max

## What’s integrated
- **Formulator**: pair metrics, ranking, optional stoichiometry helpers (PU / Epoxy)
- **Hotmaps**: organized by purpose (Polyester / PU / Epoxy / Vinylics / Silanes)
- **Figures 7–8**: Ra vs Δδa and Ra vs Π, with exportable pair tables

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
- Streamlit Community Cloud: set the app file to `app.py`.
- Optional: place `Database_final.xlsx` in repo root for automatic load.
