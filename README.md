# DiPEVa — Formulator & Compatibility Maps (Streamlit)

Focused formulation tool using HSP + manuscript indices:

- **Ra** (Hansen distance)
- **Δδa**
- **Π** (normalized interfacial index; **no weights**)

## Key Formulator Features
- **Editable equivalents library** (internal table): store **%NCO**, **OH#**, **EEW**, **AHEW** by abbreviation.
- **Total batch mass target**: compute stoichiometric masses such that *total mass includes all components*.
- **Ranking**: automatically ranks candidate combinations (e.g., all hardeners × selected resin) by a **compatibility % score**.

> The equivalents values are placeholders unless you enter supplier-confirmed values.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
Keep these files in repo root:
- `app.py`
- `Database_final.xlsx`
- `equivalents_library.csv`
- `requirements.txt`
