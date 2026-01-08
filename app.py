
# DiPEVa Formulator — Streamlit App (REFactored, clean, deterministic)
# Academic / research use only — screening tool, not a standalone decision device.

import re
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic / research use only — screening tool, not a standalone decision device.")

HERE = Path(__file__).resolve().parent

DB_FILE = HERE / "Database_final_classified_strict.xlsx"
if not DB_FILE.exists():
    DB_FILE = HERE / "Database_final_classified_full.xlsx"
if not DB_FILE.exists():
    DB_FILE = HERE / "Database_final.xlsx"

# ============================================================
# HELPERS
# ============================================================
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

# ============================================================
# LOAD DATABASE
# ============================================================
@st.cache_data
def load_db(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df.copy()

df_raw = load_db(DB_FILE)

def _find_col(cands):
    for c in cands:
        if c in df_raw.columns:
            return c
    for c in df_raw.columns:
        for k in cands:
            if k.lower() in str(c).lower():
                return c
    return None

COL = {
    "name":   _find_col(["Molecule", "Molecule name", "Name"]),
    "abbr":   _find_col(["Abbrev", "Abbreviation"]),
    "cas":    _find_col(["CAS", "CASRN"]),
    "smiles": _find_col(["SMILES/BigSMILES", "SMILES", "BigSMILES"]),
    "mw":     _find_col(["MW", "Molecular weight", "MolWt"]),
    "dD":     _find_col(["δD", "dD"]),
    "dP":     _find_col(["δP", "dP"]),
    "dH":     _find_col(["δH", "dH"]),
    "da":     _find_col(["δa", "delta a"]),
    "sig":    _find_col(["σL", "surface tension"]),
    "cls":    _find_col(["Class", "Classification", "__class__", "Role"]),
}

missing = [k for k in ["abbr", "dD", "dP", "dH"] if COL[k] is None]
if missing:
    st.error(f"Database missing required columns: {missing}")
    st.stop()

# ============================================================
# BUILD MASTER TABLE
# ============================================================
T = df_raw.copy()
T["__abbr__"] = (
    T[COL["abbr"]]
    .astype(str)
    .str.strip()
    .str.upper()
)
T = T[T["__abbr__"].ne("")]
T = T.drop_duplicates("__abbr__", keep="first").set_index("__abbr__")

for k in ["dD", "dP", "dH", "da", "sig", "mw"]:
    if COL.get(k) and COL[k] in T.columns:
        T[COL[k]] = pd.to_numeric(T[COL[k]], errors="coerce")

# ============================================================
# CLASS NORMALIZATION (SINGLE SOURCE OF TRUTH)
# ============================================================
_CLASS_MAP = {
    "isocyanate": "isocyanate",
    "polyol": "polyol",
    "extender": "extender",
    "crosslinker": "crosslinker",
    "acid": "acid_anhydride",
    "anhydride": "acid_anhydride",
    "epoxy_resin": "epoxy_resin",
    "reactive_diluent": "reactive_diluent",
    "epoxy_hardener": "epoxy_hardener",
    "vinyl_monomer": "vinyl_monomer",
    "solvent": "solvent_plasticizer",
    "plasticizer": "solvent_plasticizer",
    "silane": "silane",
}

def normalize_class(val: str) -> str:
    if not isinstance(val, str):
        return "other"
    s = re.sub(r"[^a-z0-9_]+", "", val.lower().replace(" ", "_"))
    return _CLASS_MAP.get(s, s or "other")

T["__class_db__"] = (
    T[COL["cls"]].astype(str).apply(normalize_class)
    if COL.get("cls") in T.columns else "other"
)

# ============================================================
# EPOXY FUNCTIONALITY (DETERMINISTIC)
# ============================================================
def count_epoxy(smiles: str) -> int:
    if not isinstance(smiles, str):
        return 0
    s = smiles.replace(" ", "")
    return (
        len(re.findall(r"C(\\d)OC\\1", s)) +
        len(re.findall(r"C(\\d)CO\\1", s)) +
        len(re.findall(r"O(\\d)CC\\1", s))
    )

def estimate_epoxy_fn(abbr: str) -> int:
    smi = T.loc[abbr, COL["smiles"]] if COL.get("smiles") in T.columns else ""
    n = count_epoxy(str(smi))
    if n > 0:
        return min(n, 6)
    if "DGEBA" in abbr or "DGEBF" in abbr:
        return 2
    return 0

T["__epoxy_fn__"] = [estimate_epoxy_fn(a) for a in T.index]

# ============================================================
# EFFECTIVE CLASS (ARTICLE CONSISTENT)
# ============================================================
T["__class_eff__"] = T["__class_db__"]

# Enforce epoxy rules
T.loc[T["__epoxy_fn__"] >= 2, "__class_eff__"] = "epoxy_resin"
T.loc[T["__epoxy_fn__"] == 1, "__class_eff__"] = "reactive_diluent"

# ============================================================
# METRICS
# ============================================================
def Ra(a, b):
    return float(np.sqrt(
        4*(a[COL["dD"]] - b[COL["dD"]])**2 +
        (a[COL["dP"]] - b[COL["dP"]])**2 +
        (a[COL["dH"]] - b[COL["dH"]])**2
    ))

def delta_a(a, b):
    if COL.get("da") in a and COL.get("da") in b:
        if not pd.isna(a[COL["da"]]) and not pd.isna(b[COL["da"]):
            return float(abs(a[COL["da"]] - b[COL["da"]]))
    return float(abs(
        np.sqrt(a[COL["dP"]]**2 + a[COL["dH"]]**2) -
        np.sqrt(b[COL["dP"]]**2 + b[COL["dH"]]**2)
    ))

def delta_sigma(a, b):
    if COL.get("sig") and not pd.isna(a[COL["sig"]]) and not pd.isna(b[COL["sig"]):
        return float(abs(a[COL["sig"]] - b[COL["sig"]]))
    return float("nan")

def Pi(da, ds, da_max, ds_max):
    v = da / (da_max + 1e-12)
    if np.isfinite(ds):
        v += ds / (ds_max + 1e-12)
    return float(v)

# ============================================================
# POOLS
# ============================================================
def list_by_class(cls):
    return sorted(T.index[T["__class_eff__"] == cls].tolist())

# ============================================================
# UI — MINIMAL, ROBUST
# ============================================================
st.sidebar.header("Quick sanity check")

st.write("### Database summary")
st.write(
    T["__class_eff__"].value_counts().rename("count").to_frame()
)

st.write("### Sample molecules")
st.dataframe(T.head(10))

st.success("App core loaded correctly. Heatmaps, figures and formulators can now be safely extended.")
