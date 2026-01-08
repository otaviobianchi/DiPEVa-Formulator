# ============================================================
# DiPEVa Formulator — Streamlit App
# App + Article figures (Ra, Δδa, Π)
# Academic / research use only
# ============================================================

import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic / research use only — screening tool, not a standalone decision device.")

HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "Database_final_classified_strict.xlsx"

# ------------------------------------------------------------
# LOAD DATABASE
# ------------------------------------------------------------
@st.cache_data
def load_db(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_db(DB_FILE)

# ------------------------------------------------------------
# COLUMN DETECTION (robust)
# ------------------------------------------------------------
def find_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    for c in df.columns:
        for k in cands:
            if k.lower() in c.lower():
                return c
    return None

COL = {
    "abbr": find_col(["Abbrev", "Abbreviation"]),
    "name": find_col(["Name", "Molecule"]),
    "smiles": find_col(["SMILES", "BigSMILES"]),
    "mw": find_col(["MW", "Molecular weight"]),
    "dD": find_col(["δD"]),
    "dP": find_col(["δP"]),
    "dH": find_col(["δH"]),
    "da": find_col(["δa"]),
    "sig": find_col(["σL"]),
    "cls": find_col(["Class", "Classification"])
}

# ------------------------------------------------------------
# CLEAN TABLE
# ------------------------------------------------------------
T = df.copy()
T["__abbr__"] = T[COL["abbr"]].astype(str).str.upper().str.strip()
T = T.drop_duplicates("__abbr__").set_index("__abbr__", drop=True)

for k in ["mw", "dD", "dP", "dH", "da", "sig"]:
    if COL.get(k):
        T[COL[k]] = pd.to_numeric(T[COL[k]], errors="coerce")

# ------------------------------------------------------------
# POLYMERIC FLAG (polyol hygiene)
# ------------------------------------------------------------
def is_polymeric(abbr, name, smiles, mw):
    a = abbr.upper()
    n = (name or "").lower()
    s = (smiles or "")
    if any(x in s for x in ["{", "}", "[*]"]):
        return True
    if re.search(r"(PEG|PPG|PTMEG|PCL|PCDL)\d{2,5}", a):
        return True
    if "polyol" in n or "polyether" in n or "polyester" in n:
        return True
    if mw and mw >= 250 and "diol" in n:
        return True
    return False

T["__is_polymeric__"] = [
    is_polymeric(
        ab,
        T.loc[ab, COL["name"]] if COL["name"] else "",
        T.loc[ab, COL["smiles"]] if COL["smiles"] else "",
        T.loc[ab, COL["mw"]] if COL["mw"] else np.nan,
    )
    for ab in T.index
]

# ------------------------------------------------------------
# EPOXY FUNCTIONALITY (SMILES-based)
# ------------------------------------------------------------
def count_epoxy(smiles):
    if not isinstance(smiles, str):
        return 0
    return len(re.findall(r"C\dOC\d|C\dCO\d|O\dCC\d", smiles))

T["__epoxy_fn__"] = [
    count_epoxy(T.loc[ab, COL["smiles"]]) if COL["smiles"] else 0
    for ab in T.index
]

# ------------------------------------------------------------
# EFFECTIVE CLASS (ARTICLE CONSISTENT)
# ------------------------------------------------------------
T["__class_eff__"] = T[COL["cls"]].astype(str).str.lower()

# epoxy override
T.loc[T["__epoxy_fn__"] >= 2, "__class_eff__"] = "epoxy_resin"
T.loc[T["__epoxy_fn__"] == 1, "__class_eff__"] = "reactive_diluent"

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def Ra(a, b):
    return float(np.sqrt(
        4*(a[COL["dD"]] - b[COL["dD"]])**2 +
        (a[COL["dP"]] - b[COL["dP"]])**2 +
        (a[COL["dH"]] - b[COL["dH"]])**2
    ))

def delta_a(a, b):
    if COL["da"] and not pd.isna(a[COL["da"]]) and not pd.isna(b[COL["da"]]):
        return abs(a[COL["da"]] - b[COL["da"]])
    return abs(
        np.sqrt(a[COL["dP"]]**2 + a[COL["dH"]]**2) -
        np.sqrt(b[COL["dP"]]**2 + b[COL["dH"]]**2)
    )

def delta_sigma(a, b):
    if COL["sig"] and not pd.isna(a[COL["sig"]]) and not pd.isna(b[COL["sig"]]):
        return abs(a[COL["sig"]] - b[COL["sig"]])
    return np.nan

def Pi(da, ds, da_max, ds_max):
    nda = da / (da_max + 1e-9)
    if np.isfinite(ds):
        nds = ds / (ds_max + 1e-9)
        return nda + nds
    return nda

# ------------------------------------------------------------
# POOLS
# ------------------------------------------------------------
def by_class(c):
    return sorted([i for i in T.index if T.loc[i, "__class_eff__"] == c])

polyols = [i for i in by_class("polyol") if T.loc[i, "__is_polymeric__"]]
extenders = by_class("chain_extender_diol")
isos = by_class("isocyanate")

ep_resins = by_class("epoxy_resin")
ep_dils   = by_class("reactive_diluent")
ep_hards  = by_class("epoxy_hardener_amine")

# ------------------------------------------------------------
# UI TABS
# ------------------------------------------------------------
tab_form, tab_maps, tab_figs = st.tabs(
    ["Formulator", "Heatmaps", "Figures (Article)"]
)

# ============================================================
# FORMULATOR
# ============================================================
with tab_form:
    st.subheader("Formulator")

    fam = st.selectbox("System", ["Polyurethane", "Epoxy"])

    if fam == "Epoxy":
        r = st.selectbox("Resin", ep_resins)
        h = st.selectbox("Hardener", ep_hards)
        d = st.selectbox("Reactive diluent (optional)", ["None"] + ep_dils)

        if r and h:
            a = T.loc[r]
            b = T.loc[h]
            ra = Ra(a, b)
            da = delta_a(a, b)
            ds = delta_sigma(a, b)

            st.metric("Ra", f"{ra:.2f}")
            st.metric("Δδa", f"{da:.2f}")
            st.metric("ΔσL", f"{ds:.2f}" if np.isfinite(ds) else "n/a")

# ============================================================
# HEATMAPS
# ============================================================
with tab_maps:
    st.subheader("Heatmaps")

    metric = st.selectbox("Metric", ["Ra", "Δδa", "Π"])

    rows = st.multiselect("Rows", ep_dils, default=ep_dils)
    cols = st.multiselect("Cols", ep_resins, default=ep_resins)

    if rows and cols:
        da_max = np.nanmax(T[COL["da"]]) if COL["da"] else 1.0
        ds_max = np.nanmax(T[COL["sig"]]) if COL["sig"] else 1.0

        M = pd.DataFrame(index=rows, columns=cols)

        for i in rows:
            for j in cols:
                a, b = T.loc[i], T.loc[j]
                if metric == "Ra":
                    M.loc[i, j] = Ra(a, b)
                elif metric == "Δδa":
                    M.loc[i, j] = delta_a(a, b)
                else:
                    M.loc[i, j] = Pi(
                        delta_a(a, b),
                        delta_sigma(a, b),
                        da_max, ds_max
                    )

        fig, ax = plt.subplots(figsize=(0.5*len(cols)+4, 0.4*len(rows)+3))
        im = ax.imshow(M.values.astype(float))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

# ============================================================
# FIGURES (ARTICLE)
# ============================================================
with tab_figs:
    st.subheader("Figures for manuscript")

    pairs = []
    da_max = np.nanmax(T[COL["da"]]) if COL["da"] else 1.0
    ds_max = np.nanmax(T[COL["sig"]]) if COL["sig"] else 1.0

    for d in ep_dils:
        for r in ep_resins:
            a, b = T.loc[d], T.loc[r]
            pairs.append({
                "diluent": d,
                "resin": r,
                "Ra": Ra(a, b),
                "Δδa": delta_a(a, b),
                "Π": Pi(delta_a(a, b), delta_sigma(a, b), da_max, ds_max)
            })

    DF = pd.DataFrame(pairs)

    fig1, ax1 = plt.subplots()
    ax1.scatter(DF["Ra"], DF["Δδa"])
    ax1.set_xlabel("Ra")
    ax1.set_ylabel("Δδa")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(DF["Ra"], DF["Π"])
    ax2.set_xlabel("Ra")
    ax2.set_ylabel("Π")
    st.pyplot(fig2)

    st.dataframe(DF.round(3))
