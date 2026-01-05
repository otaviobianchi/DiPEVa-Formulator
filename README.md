# DiPEVa Formulator

A Streamlit app for formulation screening using Hansen solubility parameters and interfacial proxies, built on the **DiPEVa** database.

## What it does
- **Heatmaps** for four formulation families:
  - **Polyesters** (acids/anhydrides × polyols)
  - **Polyurethanes** (isocyanates × polyols; isocyanates × chain extenders)
  - **Epoxies** (resins × hardeners; resins × reactive diluents)
  - **Vinyls** (monomers × solvents/plasticizers)
  - **Silanes** (silane coupling agents × selected targets)
- **FIGURES**:
  - **Ra vs Δδa**
  - **Ra vs Π**
- **Formulator**:
  - PU stoichiometry (NCO/OH index) with total mass targeting
  - Epoxy stoichiometry (EEW/AHEW) with total mass targeting
  - Automatic **ranking** of best candidates by a **compatibility score (%)** that combines **Ra**, **Δδa**, and **Π** (equal weighting)

## Research / academic use only
This tool is intended for academic research and screening, not as a standalone decision device.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
The app ships with `Database_final.xlsx` in the repo root (internal database).
