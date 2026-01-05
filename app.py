import io
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic / research use only — screening tool, not a standalone decision device.")

DB_FILE = Path(__file__).resolve().parent / "Database_final.xlsx"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_db(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # The user may keep only the columns (Molecule missing in some exports). Handle robustly.
    # Prefer Abbrev as unique ID for formulation work.
    df = df.copy()
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    return df

df_raw = load_db(DB_FILE)

# Detect key columns
COL = {}
def _find_col(cands):
    for c in cands:
        if c in df_raw.columns:
            return c
    # try fuzzy
    for c in df_raw.columns:
        for k in cands:
            if k.lower() in str(c).lower():
                return c
    return None

COL["name"]  = _find_col(["Molecule", "Molecule name", "Name"])
COL["abbr"]  = _find_col(["Abbrev", "Abbreviation", "abbr"])
COL["cas"]   = _find_col(["CAS", "CASRN"])
COL["smiles"]= _find_col(["SMILES/BigSMILES", "SMILES", "BigSMILES"])
COL["dD"]    = _find_col(["δD (MPa1/2)", "deltaD", "dD"])
COL["dP"]    = _find_col(["δP (MPa1/2)", "deltaP", "dP"])
COL["dH"]    = _find_col(["δH (MPa1/2)", "deltaH", "dH"])
COL["sig"]   = _find_col(["σL (mN·m-1)", "sigmaL", "σL"])
COL["da"]    = _find_col(["δa (MPa1/2)", "delta a", "δa"])

missing = [k for k in ["abbr","dD","dP","dH"] if COL.get(k) is None]
if missing:
    st.error(f"Database is missing required columns: {missing}\nFound columns: {list(df_raw.columns)}")
    st.stop()

# Build table indexed by Abbrev
T = df_raw.copy()
T["__abbr__"] = T[COL["abbr"]].astype(str).str.strip()
T = T[T["__abbr__"].ne("")].copy()

# Deduplicate by Abbrev (keep first)
T = T.drop_duplicates(subset="__abbr__", keep="first").set_index("__abbr__", drop=True)

# Coerce numeric
for k in ["dD","dP","dH","sig","da"]:
    c = COL.get(k)
    if c and c in T.columns:
        T[c] = pd.to_numeric(T[c], errors="coerce")

# =========================
# METRICS (as in article usage)
# =========================
def Ra(a: pd.Series, b: pd.Series) -> float:
    dD, dP, dH = COL["dD"], COL["dP"], COL["dH"]
    return float(np.sqrt(4.0*(a[dD]-b[dD])**2 + (a[dP]-b[dP])**2 + (a[dH]-b[dH])**2))

def delta_a(a: pd.Series, b: pd.Series) -> float:
    # Article: use δa column when available (already represents polar/hbond resultant).
    c = COL.get("da")
    if c and (c in a.index) and (c in b.index) and (not pd.isna(a[c])) and (not pd.isna(b[c])):
        return float(abs(a[c] - b[c]))
    # fallback to resultant of δP and δH
    dP, dH = COL["dP"], COL["dH"]
    return float(abs(np.sqrt(a[dP]**2 + a[dH]**2) - np.sqrt(b[dP]**2 + b[dH]**2)))

def delta_sigmaL(a: pd.Series, b: pd.Series) -> float:
    c = COL.get("sig")
    if not c or c not in a.index or c not in b.index:
        return float("nan")
    if pd.isna(a[c]) or pd.isna(b[c]):
        return float("nan")
    return float(abs(a[c] - b[c]))

def Pi(da: float, ds: float, da_max: float, ds_max: float) -> float:
    """Combined penalty descriptor Π as in the article (Eq. 8):
    Π = Δδa/δa,max + ΔσL/σL,max  (no explicit wa/ws weights).
    If σL is unavailable, we fall back to the δa term only.
    """
    nda = da/(da_max + 1e-12)
    if np.isfinite(ds):
        nds = ds/(ds_max + 1e-12)
        return float(nda + nds)
    return float(nda)
def score_percent(ra: float, da: float, pi: float, ra_max: float, da_max: float, pi_max: float) -> float:
    # Higher is better (0..100)
    n_ra = ra/(ra_max + 1e-12)
    n_da = da/(da_max + 1e-12)
    n_pi = pi/(pi_max + 1e-12)
    s = 100.0*(1.0 - (n_ra + n_da + n_pi)/3.0)
    return float(np.clip(s, 0, 100))

def fmt2(x):
    try:
        if pd.isna(x): return ""
        return f"{float(x):.2f}"
    except Exception:
        return ""

# =========================
# CLASSIFICATION (Abbrev-based + name-based)
# =========================
def _name_of(abbr: str) -> str:
    if COL["name"] and COL["name"] in T.columns:
        v = T.loc[abbr, COL["name"]]
        return str(v) if not pd.isna(v) else abbr
    return abbr

def contains_any(text, pats):
    t = str(text).lower()
    return any(p in t for p in pats)

def classify_row(abbr: str, row: pd.Series) -> str:
    name = _name_of(abbr).lower()
    a = abbr.lower()

    # PU
    if "nco" in name or "diisocyanate" in name or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", a):
        return "isocyanate"
    if "butanediol" in name or re.search(r"\b1,4-bdo\b|\b1,3-bdo\b|\bbdo\b", name):
        return "extender"
    if "glycol" in name or "polyol" in name or "diol" in name or re.search(r"\bpeg\d+|\bppg\d+|\bptmeg|\bpcl\d+|\bpcdl", a):
        return "polyol"

    # Polyester acids/anhydrides
    if "acid" in name or "anhydride" in name or a.endswith("a") or re.search(r"\bmah\b|\bmhhpa\b|\bhhpa\b", a):
        return "acid_anhydride"

    # Epoxy resins / hardeners / diluents
    if "epoxy" in name or re.search(r"\bdgeba\b|\bbdge\b", a):
        return "epoxy_resin"
    if "diamin" in name or "amine" in name or re.search(r"\bdeta\b|\bteta\b|\bddm\b|\bdds\b|\bdicy\b", a):
        return "epoxy_hardener"
    if "carbonate" in name or "lactone" in name or "glyme" in name or "ether" in name:
        return "reactive_diluent"

    # Vinyl monomers
    if "styrene" in name or "acrylonitrile" in name or "methacrylate" in name or "vinyl acetate" in name or re.search(r"\ban\b|\bmma\b|\bst\b|\bfn\b|\bα-ms\b", a):
        return "vinyl_monomer"

    # Solvents/plasticizers
    if "alcohol" in name or "phthalate" in name or "benzoate" in name or "adipate" in name or "phosphate" in name:
        return "solvent_plasticizer"

    # Silanes
    if "silane" in name or re.search(r"\bteos\b|\bvtms\b|\baptes\b|\bmptes\b", a):
        return "silane"

    return "other"

T["__class__"] = [classify_row(i, T.loc[i]) for i in T.index]

# Families for heatmaps
def list_by_class(cls):
    return sorted([i for i in T.index if T.loc[i, "__class__"] == cls])

# convenience name labels
def label(abbr: str) -> str:
    nm = _name_of(abbr)
    return f"{abbr} — {nm}"

def select_abbr(title: str, options: list[str], key: str):
    if not options:
        st.warning(f"No options found for: {title}")
        return None
    # show labels but return abbr
    labels = {label(o): o for o in options}
    choice = st.selectbox(title, list(labels.keys()), key=key)
    return labels[choice]

def show_props(abbr: str):
    if not abbr:
        return
    row = T.loc[abbr]
    cols_show = []
    for k in ["name", "cas", "smiles", "dD", "dP", "dH", "da", "sig"]:
        c = COL.get(k)
        if c and c in row.index:
            cols_show.append(c)
    s = row[cols_show].copy()

    # Format solubility parameters to 2 decimals (as requested)
    for k in ["dD", "dP", "dH", "da"]:
        c = COL.get(k)
        if c and c in s.index and pd.api.types.is_number(s[c]):
            s[c] = float(s[c])
    # Build output with per-field formatting
    out = {}
    for k in cols_show:
        v = s[k]
        if isinstance(v, (float, int, np.floating, np.integer)) and not pd.isna(v):
            if k in {COL["dD"], COL["dP"], COL["dH"], COL.get("da")}:
                out[k] = f"{float(v):.2f}"
            elif k == COL.get("sig"):
                out[k] = f"{float(v):.2f}"
            else:
                out[k] = str(v)
        else:
            out[k] = "" if pd.isna(v) else str(v)
    st.write(pd.DataFrame.from_dict(out, orient="index", columns=["value"]))

# =========================
# PLOTS
# =========================
def plot_heatmap(mat: pd.DataFrame, title: str, cbar: str):
    # mat must be numeric
    fig, ax = plt.subplots(figsize=(max(8, 0.45*mat.shape[1]), max(4, 0.35*mat.shape[0])))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar)
    fig.tight_layout()
    return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8,6))
    if hue in df.columns:
        for g in sorted(df[hue].unique()):
            sub = df[df[hue] == g]
            ax.scatter(sub[x], sub[y], label=str(g), alpha=0.75)
        ax.legend()
    else:
        ax.scatter(df[x], df[y], alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

# =========================
# UI TABS
# =========================
tab_form, tab_maps, tab_figs = st.tabs(
    ["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"]
)

# -------------------------
# FORMULATOR
# -------------------------
with tab_form:
    st.subheader("Formulator")

    mode = st.selectbox(
        "Mode",
        ["Pair screening", "PU stoichiometry (NCO/OH)", "Epoxy stoichiometry (EEW/AHEW)", "Ranking (best combinations)"],
        key="mode_select",
    )

    # default sets
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    ep_resins = list_by_class("epoxy_resin")
    ep_hards = list_by_class("epoxy_hardener")
    ep_dils  = list_by_class("reactive_diluent")

    # global maxima for Π normalization
    da_max = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    if mode == "Pair screening":
        st.markdown("### Pair screening")
        c1, c2 = st.columns(2)
        with c1:
            A = select_abbr("Component A", list(T.index), key="pair_A")
        with c2:
            B = select_abbr("Component B", list(T.index), key="pair_B")

        if A and B:
            a = T.loc[A]; b = T.loc[B]
            ra = Ra(a,b)
            da = delta_a(a,b)
            ds = delta_sigmaL(a,b)
            pi = Pi(da, ds, da_max, ds_max)

            st.markdown("#### Results")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Ra (MPa$^{1/2}$)", f"{ra:.2f}")
            r2.metric("Δδa (MPa$^{1/2}$)", f"{da:.2f}")
            r3.metric("ΔσL (mN/m)", f"{ds:.2f}" if np.isfinite(ds) else "n/a")
            r4.metric("Π (0–1)", f"{pi:.3f}")

            st.markdown("#### Selected properties")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**A**")
                show_props(A)
            with c2:
                st.write("**B**")
                show_props(B)

    elif mode == "PU stoichiometry (NCO/OH)":
        st.markdown("### PU stoichiometry (NCO/OH)")
        st.caption("Uses %NCO and OH number; choose a total mass target (default 100 g).")

        c1, c2, c3 = st.columns(3)
        with c1:
            iso = select_abbr("Isocyanate", isos, key="pu_iso")
        with c2:
            pol = select_abbr("Polyol", polyols, key="pu_pol")
        with c3:
            use_ext = st.checkbox("Include chain extender", value=True, key="pu_use_ext")

        ext = None
        if use_ext:
            ext = select_abbr("Chain extender", extenders, key="pu_ext")

        st.markdown("#### Inputs")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            nco = st.number_input("%NCO (wt%)", min_value=0.0, max_value=60.0, value=33.0, step=0.1)
        with cc2:
            oh_pol = st.number_input("OH number polyol (mgKOH/g)", min_value=0.0, max_value=3000.0, value=56.0, step=1.0)
        with cc3:
            oh_ext = st.number_input("OH number extender (mgKOH/g)", min_value=0.0, max_value=3000.0, value=1200.0 if use_ext else 0.0, step=1.0)

        idx = st.number_input("NCO/OH index", min_value=0.5, max_value=2.5, value=1.05, step=0.01)
        mass_mode = st.radio("Mass target mode", ["Total mass (includes isocyanate)", "Polyol-side mass (polyol+extender)"], horizontal=True)
        target = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0)
        ext_frac = st.slider("Extender mass fraction in polyol side", 0.0, 0.8, 0.2, step=0.01) if (use_ext and ext) else 0.0

        # Eq weights:
        EW_OH_pol = 56100.0 / max(oh_pol, 1e-12)
        EW_OH_ext = 56100.0 / max(oh_ext, 1e-12) if (use_ext and ext and oh_ext>0) else np.inf
        EW_NCO    = 4200.0  / max(nco,   1e-12)

        if iso and pol:
            if mass_mode == "Polyol-side mass (polyol+extender)":
                m_pol_side = target
                m_pol = m_pol_side*(1.0-ext_frac)
                m_ext = m_pol_side*ext_frac if (use_ext and ext) else 0.0
                eq_OH = (m_pol/EW_OH_pol) + (m_ext/EW_OH_ext if (use_ext and ext) else 0.0)
                m_iso = idx * eq_OH * EW_NCO
                m_total = m_pol_side + m_iso
            else:
                f = ext_frac if (use_ext and ext) else 0.0
                term = (1.0-f)/EW_OH_pol + (f/EW_OH_ext if (use_ext and ext) else 0.0)
                k = idx * EW_NCO * term
                m_iso = (k/(1.0+k))*target
                m_pol_side = target - m_iso
                m_pol = m_pol_side*(1.0-f)
                m_ext = m_pol_side*f
                m_total = target

            st.markdown("#### Calculated masses")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Isocyanate (g)", f"{m_iso:.2f}")
            r3.metric("Polyol (g)", f"{m_pol:.2f}")
            r4.metric("Extender (g)", f"{m_ext:.2f}" if (use_ext and ext) else "0.00")

            st.markdown("#### Compatibility indices (screening)")
            rows=[]
            for comp_id, tag in [(pol,"polyol")] + ([(ext,"extender")] if (use_ext and ext) else []):
                a = T.loc[comp_id]; b = T.loc[iso]
                ra = Ra(a,b); da = delta_a(a,b); ds = delta_sigmaL(a,b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"pair": f"{tag} × isocyanate", "A": comp_id, "B": iso, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
            dfm = pd.DataFrame(rows)
            if not dfm.empty:
                raM = float(dfm["Ra"].max()); daM = float(dfm["Δδa"].max()); piM = float(dfm["Π"].max())
                dfm["Score_%"] = [score_percent(r,d,p,raM,daM,piM) for r,d,p in zip(dfm["Ra"], dfm["Δδa"], dfm["Π"])]
                st.dataframe(dfm, use_container_width=True, hide_index=True)

    elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.markdown("### Epoxy stoichiometry (EEW/AHEW)")
        st.caption("Total mass targeting (default 100 g). Optionally includes a reactive diluent in the resin-side mass.")

        c1, c2, c3 = st.columns(3)
        with c1:
            resin = select_abbr("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = select_abbr("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

        dil = None
        if use_dil:
            dil = select_abbr("Reactive diluent", ep_dils, key="ep_dil")

        st.markdown("#### Inputs")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            EEW_res = st.number_input("Resin EEW (g/eq)", min_value=50.0, max_value=3000.0, value=185.0, step=1.0)
        with cc2:
            AHEW = st.number_input("Hardener AHEW (g/eq)", min_value=10.0, max_value=2000.0, value=30.0, step=1.0)
        with cc3:
            ratio = st.number_input("Epoxy/Active-H eq ratio", min_value=0.5, max_value=2.0, value=1.0, step=0.01)

        mass_mode = st.radio("Mass target mode", ["Total mass (includes hardener)", "Resin-side mass (resin+diluent)"], horizontal=True)
        target = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key="ep_target")

        dil_frac = 0.0
        EEW_dil = np.inf
        if use_dil and dil:
            dil_frac = st.slider("Diluente mass fraction in resin-side", 0.0, 0.8, 0.2, step=0.01)
            EEW_dil = st.number_input("Diluente EEW (g/eq)", min_value=50.0, max_value=3000.0, value=150.0, step=1.0)

        # equivalent term for resin-side epoxide equivalents per gram
        term = (1.0-dil_frac)/EEW_res + (dil_frac/EEW_dil if (use_dil and dil) else 0.0)
        g = (AHEW/max(ratio,1e-12)) * term  # hardener mass per resin-side mass

        if resin and hard:
            if mass_mode == "Resin-side mass (resin+diluent)":
                m_res_side = target
                m_hard = g*m_res_side
                m_total = m_res_side + m_hard
            else:
                m_hard = (g/(1.0+g))*target
                m_res_side = target - m_hard
                m_total = target

            m_dil = m_res_side*dil_frac if (use_dil and dil) else 0.0
            m_res = m_res_side - m_dil

            st.markdown("#### Calculated masses")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Hardener (g)", f"{m_hard:.2f}")
            r3.metric("Resin (g)", f"{m_res:.2f}")
            r4.metric("Reactive diluent (g)", f"{m_dil:.2f}" if (use_dil and dil) else "0.00")

            st.markdown("#### Compatibility indices (screening)")
            rows=[]
            for comp_id, tag in [(hard,"hardener")] + ([(dil,"diluent")] if (use_dil and dil) else []):
                a = T.loc[comp_id]; b = T.loc[resin]
                ra = Ra(a,b); da = delta_a(a,b); ds = delta_sigmaL(a,b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"pair": f"{tag} × resin", "A": comp_id, "B": resin, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
            dfm = pd.DataFrame(rows)
            if not dfm.empty:
                raM = float(dfm["Ra"].max()); daM = float(dfm["Δδa"].max()); piM = float(dfm["Π"].max())
                dfm["Score_%"] = [score_percent(r,d,p,raM,daM,piM) for r,d,p in zip(dfm["Ra"], dfm["Δδa"], dfm["Π"])]
                st.dataframe(dfm, use_container_width=True, hide_index=True)

    else:
        st.markdown("### Ranking (best combinations)")
        st.caption("Ranks candidates by a compatibility **Score_%** combining Ra + Δδa + Π (equal weighting).")

        family = st.selectbox("Family", ["PU", "Epoxy"], key="rk_family")

        if family == "Epoxy":
            resin = select_abbr("Resin (fixed)", ep_resins, key="rk_ep_resin")
            include_dil = st.checkbox("Include reactive diluents in ranking", value=True, key="rk_ep_dil")
            cand = ep_hards + (ep_dils if include_dil else [])
            if resin and cand:
                rows=[]
                for c in cand:
                    a=T.loc[c]; b=T.loc[resin]
                    ra=Ra(a,b); da=delta_a(a,b); ds=delta_sigmaL(a,b)
                    pi=Pi(da, ds, da_max, ds_max)
                    rows.append({"candidate": c, "type": T.loc[c,"__class__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
                dfc = pd.DataFrame(rows)
                raM=float(dfc["Ra"].max()); daM=float(dfc["Δδa"].max()); piM=float(dfc["Π"].max())
                dfc["Score_%"]=[score_percent(r,d,p,raM,daM,piM) for r,d,p in zip(dfc["Ra"], dfc["Δδa"], dfc["Π"])]
                dfc=dfc.sort_values("Score_%", ascending=False)
                st.dataframe(dfc, use_container_width=True, hide_index=True)
        else:
            iso = select_abbr("Isocyanate (fixed)", isos, key="rk_pu_iso")
            include_ext = st.checkbox("Include extenders in ranking", value=True, key="rk_pu_ext")
            cand = polyols + (extenders if include_ext else [])
            if iso and cand:
                rows=[]
                for c in cand:
                    a=T.loc[c]; b=T.loc[iso]
                    ra=Ra(a,b); da=delta_a(a,b); ds=delta_sigmaL(a,b)
                    pi=Pi(da, ds, da_max, ds_max)
                    rows.append({"candidate": c, "type": T.loc[c,"__class__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
                dfc = pd.DataFrame(rows)
                raM=float(dfc["Ra"].max()); daM=float(dfc["Δδa"].max()); piM=float(dfc["Π"].max())
                dfc["Score_%"]=[score_percent(r,d,p,raM,daM,piM) for r,d,p in zip(dfc["Ra"], dfc["Δδa"], dfc["Π"])]
                dfc=dfc.sort_values("Score_%", ascending=False)
                st.dataframe(dfc, use_container_width=True, hide_index=True)

# -------------------------
# HEATMAPS
# -------------------------
with tab_maps:
    st.subheader("Heatmaps (grouped by purpose)")

    purpose = st.selectbox(
        "Purpose",
        ["Polyesters", "Polyurethanes", "Epoxies", "Vinyls", "Silanes"],
        key="hm_purpose",
    )

    acids = list_by_class("acid_anhydride")
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    ep_resins = list_by_class("epoxy_resin")
    ep_hards = list_by_class("epoxy_hardener")
    ep_dils  = list_by_class("reactive_diluent")
    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    sil = list_by_class("silane")

    def build_matrix(rows, cols):
        mat = pd.DataFrame(index=rows, columns=cols, dtype=float)
        for r in rows:
            a = T.loc[r]
            for c in cols:
                b = T.loc[c]
                mat.loc[r,c] = Ra(a,b)
        return mat

    if purpose == "Polyesters":
        st.caption("Acids/anhydrides × polyols (Ra).")
        left = st.multiselect("Polyols (rows)", polyols, default=polyols[:12], format_func=label)
        right = st.multiselect("Acids/anhydrides (cols)", acids, default=acids[:12], format_func=label)
        if left and right:
            mat = build_matrix(left, right)
            fig = plot_heatmap(mat, "POLYESTERS — Polyols × Acids/Anhydrides (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig)

    elif purpose == "Polyurethanes":
        st.caption("Isocyanates × polyols and extenders (Ra).")
        c1, c2 = st.columns(2)
        with c1:
            isos_sel = st.multiselect("Isocyanates (cols)", isos, default=isos[:10], format_func=label)
        with c2:
            show_ext = st.checkbox("Include extenders heatmap", value=True)

        pol_sel = st.multiselect("Polyols (rows)", polyols, default=polyols[:12], format_func=label)
        if pol_sel and isos_sel:
            mat = build_matrix(pol_sel, isos_sel)
            fig = plot_heatmap(mat, "PU — Polyols × Isocyanates (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig)

        if show_ext:
            ext_sel = st.multiselect("Extenders (rows)", extenders, default=extenders[:10], format_func=label)
            if ext_sel and isos_sel:
                mat2 = build_matrix(ext_sel, isos_sel)
                fig2 = plot_heatmap(mat2, "PU — Extenders × Isocyanates (Ra)", "Ra (MPa$^{1/2}$)")
                st.pyplot(fig2)

    elif purpose == "Epoxies":
        st.caption("Epoxy resins × hardeners and reactive diluents (Ra).")
        res_sel = st.multiselect("Resins (cols)", ep_resins, default=ep_resins[:6], format_func=label)
        hard_sel = st.multiselect("Hardeners (rows)", ep_hards, default=ep_hards[:10], format_func=label)
        if hard_sel and res_sel:
            mat = build_matrix(hard_sel, res_sel)
            fig = plot_heatmap(mat, "EPOXY — Hardeners × Resins (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig)

        dil_sel = st.multiselect("Reactive diluents (rows)", ep_dils, default=ep_dils[:10], format_func=label)
        if dil_sel and res_sel:
            mat2 = build_matrix(dil_sel, res_sel)
            fig2 = plot_heatmap(mat2, "EPOXY — Reactive diluents × Resins (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig2)

    elif purpose == "Vinyls":
        st.caption("Vinyl monomers × solvents/plasticizers (Ra).")
        mon_sel = st.multiselect("Monomers (rows)", mons, default=mons[:10], format_func=label)
        sol_sel = st.multiselect("Solvents/plasticizers (cols)", solv, default=solv[:12], format_func=label)
        if mon_sel and sol_sel:
            mat = build_matrix(mon_sel, sol_sel)
            fig = plot_heatmap(mat, "VINYLS — Monomers × Solvents/Plasticizers (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig)

    else:
        st.caption("Silane coupling agents × selected targets (Ra).")
        sil_sel = st.multiselect("Silanes (rows)", sil, default=sil[:10], format_func=label)
        # targets: water/alcohols + vinyl monomers + selected polyols
        target_pool = []
        # water/alcohols by name heuristics
        for ab in T.index:
            nm = _name_of(ab).lower()
            if "water" in nm or "alcohol" in nm:
                target_pool.append(ab)
        target_pool = sorted(set(target_pool + mons[:10] + polyols[:10]))
        tar_sel = st.multiselect("Targets (cols)", target_pool, default=target_pool[:12], format_func=label)
        if sil_sel and tar_sel:
            mat = build_matrix(sil_sel, tar_sel)
            fig = plot_heatmap(mat, "SILANES — Silanes × Targets (Ra)", "Ra (MPa$^{1/2}$)")
            st.pyplot(fig)

# -------------------------
# FIGURES
# -------------------------
with tab_figs:
    st.subheader("FIGURES — Ra vs Δδa and Ra vs Π")
    st.caption("Axes follow the manuscript: **Ra** on x-axis; **Δδa** or **Π** on y-axis.")

    purpose = st.selectbox("Family", ["Polyesters", "Polyurethanes", "Epoxies", "Vinyls", "Silanes"], key="fig_family")

    # Build pair set per family
    da_max = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    def make_pairs(A_list, B_list, tagA="A", tagB="B", hue_name="type"):
        rows=[]
        for a_id in A_list:
            a = T.loc[a_id]
            for b_id in B_list:
                b = T.loc[b_id]
                ra = Ra(a,b)
                da = delta_a(a,b)
                ds = delta_sigmaL(a,b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"A": a_id, "B": b_id, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi, hue_name: f"{tagA}×{tagB}"})
        return pd.DataFrame(rows)

    if purpose == "Polyesters":
        acids = list_by_class("acid_anhydride")
        polyols = list_by_class("polyol")
        A = st.multiselect("Polyols", polyols, default=polyols[:10], format_func=label)
        B = st.multiselect("Acids/anhydrides", acids, default=acids[:10], format_func=label)
        dfp = make_pairs(A,B,tagA="polyol",tagB="acid",hue_name="pair")
    elif purpose == "Polyurethanes":
        isos = list_by_class("isocyanate")
        polyols = list_by_class("polyol")
        ext = list_by_class("extender")
        iso_sel = st.multiselect("Isocyanates", isos, default=isos[:8], format_func=label)
        pol_sel = st.multiselect("Polyols", polyols, default=polyols[:10], format_func=label)
        ext_sel = st.multiselect("Extenders", ext, default=ext[:6], format_func=label)
        df1 = make_pairs(pol_sel, iso_sel, tagA="polyol", tagB="iso", hue_name="pair")
        df2 = make_pairs(ext_sel, iso_sel, tagA="ext", tagB="iso", hue_name="pair")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()
    elif purpose == "Epoxies":
        res = list_by_class("epoxy_resin")
        hard = list_by_class("epoxy_hardener")
        dil = list_by_class("reactive_diluent")
        res_sel = st.multiselect("Resins", res, default=res[:5], format_func=label)
        hard_sel = st.multiselect("Hardeners", hard, default=hard[:10], format_func=label)
        dil_sel = st.multiselect("Reactive diluents", dil, default=dil[:8], format_func=label)
        df1 = make_pairs(hard_sel, res_sel, tagA="hard", tagB="resin", hue_name="pair")
        df2 = make_pairs(dil_sel,  res_sel, tagA="dil",  tagB="resin", hue_name="pair")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()
    elif purpose == "Vinyls":
        mon = list_by_class("vinyl_monomer")
        sol = list_by_class("solvent_plasticizer")
        mon_sel = st.multiselect("Monomers", mon, default=mon[:10], format_func=label)
        sol_sel = st.multiselect("Solvents/plasticizers", sol, default=sol[:12], format_func=label)
        dfp = make_pairs(mon_sel, sol_sel, tagA="mon", tagB="solv", hue_name="pair")
    else:
        sil = list_by_class("silane")
        # targets as in heatmap
        target_pool=[]
        for ab in T.index:
            nm=_name_of(ab).lower()
            if "water" in nm or "alcohol" in nm:
                target_pool.append(ab)
        mons = list_by_class("vinyl_monomer")
        polyols = list_by_class("polyol")
        target_pool = sorted(set(target_pool + mons[:10] + polyols[:10]))
        sil_sel = st.multiselect("Silanes", sil, default=sil[:10], format_func=label)
        tar_sel = st.multiselect("Targets", target_pool, default=target_pool[:12], format_func=label)
        dfp = make_pairs(sil_sel, tar_sel, tagA="sil", tagB="target", hue_name="pair")

    if dfp.empty:
        st.warning("No pairs available (check selections).")
        st.stop()

    # Figures
    fig1 = plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)")
    st.pyplot(fig1)

    fig2 = plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)")
    st.pyplot(fig2)

    st.markdown("#### Data used in figures")
    st.dataframe(dfp.sort_values("Ra").round({"Ra":2,"Δδa":2,"ΔσL":2,"Π":3}), use_container_width=True, hide_index=True)
