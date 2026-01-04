# DiPEVa — Formulator & Compatibility Maps (Streamlit)

DiPEVa is a formulation-screening and compatibility-visualization app based on Solubility Parameters and the complementary indices used in the associated manuscript:

- **Ra** (Hansen distance)
- **Δδₐ** (difference in the associative parameter, δₐ = √(δP² + δH²))
- **Π** (normalized interfacial affinity index defined in the manuscript; **no weights**)

> Research and formulation support only. Experimental validation is required.

## What the app does

The interface is organized by **purpose**, mirroring the manuscript:

- **Polyesters**: acids/anhydrides × polyols
- **Polyurethanes (PU)**: isocyanates × polyols and isocyanates × chain extenders
- **Epoxies**: resins × hardeners and resins × reactive diluents
- **Vinylics**: monomers × solvents/plasticizers
- **Silanes**: silanes/siloxanes × (monomers/solvents)

The app includes:
- A didactic **Overview** with equations
- A **Formulator** to compare chosen components and compute Ra / Δδₐ / ΔσL / Π
- **Heatmaps** (Ra, Δδₐ, Π) by purpose
- **FIGURES**: scatter plots **Ra vs Δδₐ** and **Ra vs Π** by purpose

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (keep `Database_final.xlsx` in the repo root).
2. On Streamlit Cloud, select the repo and set the entrypoint to `app.py`.

