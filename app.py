
import re
import os
import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting libs
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

st.set_page_config(page_title="DiPEVa Formulator", layout="wide")

# ============================================================
# DiPEVa Formulator — Streamlit App (Integrated)
# - Uses your Database_final.xlsx structure:
#   CAS, Abbrev, SMILES/BigSMILES, Vm, δD, δP, δH, δT, σL, δa
# - Integrated deliverables:
#   (A) Formulator (pair + ranking + stoichiometry helpers)
#   (B) Hotmaps by purpose (Polyester / PU / Epoxy / Vinylics / Silanes)
#   (C) Figures 7–8 (Ra vs Δδa, Ra vs Π) by purpose + export tables
# ============================================================

st.title("DiPEVa Formulator")
st.caption("Research tool — compatibility screening using Hansen-based metrics (Ra, Δδa, ΔσL, Π).")

# ---------------------------
# Column constants (your DB)
# ---------------------------
COL_CAS   = "CAS"
COL_ABBR  = "Abbrev"
COL_SMILES= "SMILES/BigSMILES"
COL_Vm    = "Vm (cm3·mol⁻1)"
COL_dD    = "δD (MPa1/2)"
COL_dP    = "δP (MPa1/2)"
COL_dH    = "δH (MPa1/2)"
COL_dT    = "δT (MPa1/2)"
COL_sig   = "σL (mN·m-1)"
COL_da    = "δa (MPa1/2)"

REQ_COLS = [COL_ABBR, COL_dD, COL_dP, COL_dH]

# ---------------------------
# Sidebar: load DB
# ---------------------------
with st.sidebar:
    st.header("Database")
    up = st.file_uploader("Upload your DiPEVa database (.xlsx)", type=["xlsx"])
    st.write("If you deployed this repo with `Database_final.xlsx` included, you can also load it automatically.")
    use_local = st.checkbox("Load local Database_final.xlsx (if present in repo)", value=True)

db_path = None
if use_local and os.path.exists("Database_final.xlsx"):
    db_path = "Database_final.xlsx"
elif up is not None:
    db_path = up

if db_path is None:
    st.info("Upload the database to start, or add `Database_final.xlsx` to the repo root.")
    st.stop()

sheets = pd.ExcelFile(db_path).sheet_names
sheet = sheets[0] if len(sheets) == 1 else st.sidebar.selectbox("Sheet", sheets)
raw = pd.read_excel(db_path, sheet_name=sheet)

missing = [c for c in REQ_COLS if c not in raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}\nFound: {list(raw.columns)}")
    st.stop()

df = raw.copy()

# numeric coercion
for c in [COL_Vm, COL_dD, COL_dP, COL_dH, COL_dT, COL_sig, COL_da]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# compute δa if absent or mostly empty: δa = sqrt(δP^2 + δH^2)
if COL_da not in df.columns or df[COL_da].isna().mean() > 0.95:
    df[COL_da] = np.sqrt(df[COL_dP]**2 + df[COL_dH]**2)

# clean identifier
df[COL_ABBR] = df[COL_ABBR].astype(str).str.strip()
df = df[df[COL_ABBR].str.len() > 0].copy()

# aggregate duplicates by mean (numeric) + first (strings)
num_cols = [c for c in [COL_Vm, COL_dD, COL_dP, COL_dH, COL_dT, COL_sig, COL_da] if c in df.columns]
agg = {c: "mean" for c in num_cols}
for c in [COL_CAS, COL_SMILES]:
    if c in df.columns:
        agg[c] = "first"
dup_n = df.duplicated(subset=[COL_ABBR]).sum()
if dup_n > 0:
    df = df.groupby(COL_ABBR, as_index=False).agg(agg)

df = df.set_index(COL_ABBR)

st.sidebar.success(f"Loaded {len(df)} unique abbreviations from sheet '{sheet}'.")

# global maxima for Π (article style; no weights)
DA_MAX  = float(np.nanmax(df[COL_da].values))
SIG_MAX = float(np.nanmax(df[COL_sig].values)) if (COL_sig in df.columns and df[COL_sig].notna().any()) else np.nan

# ---------------------------
# Core metrics (article-consistent)
# ---------------------------
def Ra(a, b):
    return float(np.sqrt(
        4.0*(a[COL_dD]-b[COL_dD])**2 +
        (a[COL_dP]-b[COL_dP])**2 +
        (a[COL_dH]-b[COL_dH])**2
    ))

def Delta_a(a, b):
    # Δδa = |δa_A − δa_B|, δa = sqrt(δP^2 + δH^2)
    return float(abs(a[COL_da] - b[COL_da]))

def Delta_sigma(a, b):
    if COL_sig not in df.columns or np.isnan(SIG_MAX):
        return np.nan
    av = a.get(COL_sig, np.nan)
    bv = b.get(COL_sig, np.nan)
    if np.isnan(av) or np.isnan(bv):
        return np.nan
    return float(abs(av - bv))

def Pi(da, ds):
    # Π = Δδa/δa,max + ΔσL/σL,max  (no weights)
    if np.isnan(ds) or np.isnan(SIG_MAX):
        return float(da/(DA_MAX+1e-12))
    return float(da/(DA_MAX+1e-12) + ds/(SIG_MAX+1e-12))

def comp_score(ra, pi, ra_ref, pi_ref):
    # Screening score: higher is better.
    # Normalize by robust percentiles from the current database reference pool.
    ra_n = min(ra/(ra_ref+1e-12), 1.0)
    pi_n = min(pi/(pi_ref+1e-12), 1.0)
    s = 100.0*(1.0 - 0.5*ra_n - 0.5*pi_n)
    return float(max(0.0, min(100.0, s)))

# robust refs for scoring
# sample random pairs once per session
@st.cache_data(show_spinner=False)
def compute_refs(labels, sample_n=8000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < 3:
        return 1.0, 1.0
    m = min(sample_n, n*(n-1)//2)
    ia = rng.integers(0, n, size=m)
    ib = rng.integers(0, n, size=m)
    ras, pis = [], []
    for i,j in zip(ia, ib):
        if i == j:
            continue
        a = df.loc[labels[i]]
        b = df.loc[labels[j]]
        r = Ra(a,b)
        da = Delta_a(a,b)
        ds = Delta_sigma(a,b)
        p = Pi(da, ds)
        ras.append(r); pis.append(p)
    ra_ref = float(np.nanpercentile(ras, 95)) if ras else 1.0
    pi_ref = float(np.nanpercentile(pis, 95)) if pis else 1.0
    return ra_ref, pi_ref

labels = list(df.index)
ra_ref, pi_ref = compute_refs(labels)

# ---------------------------
# Purpose-based sets (no proxies, no "small polyol")
# NOTE: app uses your Abbrev codes exactly.
# ---------------------------
SETS = {
    "Polyester": {
        "pairs": [
            ("Polyols", ["EG","DEG","TEG","PG","DPG","MPG","1,3-BDO","1,4-BDO","1,5-PeDO","1,6-HDO","NPG","GLY","TMP","PER","SOR",
                        "PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PCDL500","PCDL1000","PCL1000","PPG500","PTMEG1000"],
             "Acids/Anhydrides", ["SA","AA","GA","PA","AzA","SuA","SeA","FA","IPA","TPA","PhA","MAH"])
        ]
    },
    "PU": {
        "pairs": [
            ("Polyols", ["PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PPG500","PTMEG1000","PCDL500","PCDL1000","PCL1000",
                        "PG","DPG","MPG","TEG","EG","1,4-BDO","1,6-HDO"],
             "Isocyanates", ["HDI","IPDI","H12MDI","MDI 2,4′","MDI 4,4′","TDI 2,4","TDI 2,6","TDI 80/20","pMDI (polymeric MDI)"]),
            ("Chain extenders", ["1,4-BDO","EDA","HMDA","MEA","DEA","TEA"],
             "Isocyanates", ["HDI","IPDI","H12MDI","MDI 2,4′","MDI 4,4′","TDI 2,4","TDI 2,6","TDI 80/20","pMDI (polymeric MDI)"])
        ]
    },
    "Epoxy": {
        "pairs": [
            ("Hardeners", ["DDM","DDS","DETA","TETA","DICY","EDA","HMDA","MEA","DEA","TEA"],
             "Resins", ["DGEBA","BDGE (aliphatic epoxy)"]),
            ("Reactive diluents", ["ECH","ECA","PCA","GBL"],
             "Resins", ["DGEBA","BDGE (aliphatic epoxy)"])
        ]
    },
    "Vinylics": {
        "pairs": [
            ("Monomers", ["St","α-MS","AN","MMA","FN","VAc","MAA"],
             "Solvents/Plasticizers", ["DMP","DEP","DBP","DEHP","TBP","TCP","DMC","DEC","MB","EB","MS","TAC","DMA","DMM","DMO","DEHA","IPM","IPP","BS",
                                      "MeOH","EtOH","n-PrOH","i-PrOH","2-EH","BnOH","CyOH","EGME","Diglyme"])
        ]
    },
    "Silanes": {
        "pairs": [
            ("Silanes/Siloxanes", ["VTMS","TEOS","APTES","MPTES","BTESPD","BTMPSD","TMS-OH","EMNS"],
             "Targets", ["St","α-MS","AN","MMA","VAc","MAA","MeOH","EtOH","EGME","Diglyme","MB","EB","DMP","DEP","DBP","DEHP"])
        ]
    },
}

def present_in_db(lst):
    out = []
    seen = set()
    for x in lst:
        if x in df.index and x not in seen:
            out.append(x); seen.add(x)
    return out

# ---------------------------
# Plot helpers (Streamlit)
# ---------------------------
def plot_heatmap(mat, title, cbar_label):
    if mat is None or mat.empty:
        st.warning(f"{title}: no entries found in your database for these sets.")
        return
    mat2 = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if mat2.shape[0] < 2 or mat2.shape[1] < 2:
        st.warning(f"{title}: matrix too small after filtering.")
        return

    fig, ax = plt.subplots(figsize=(max(9, 0.55*mat2.shape[1]), max(5, 0.35*mat2.shape[0])))
    if _HAS_SNS:
        sns.heatmap(mat2.astype(float), cmap="viridis", cbar_kws={"label": cbar_label}, ax=ax)
    else:
        im = ax.imshow(mat2.astype(float).values, aspect="auto")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(mat2.shape[1])); ax.set_xticklabels(mat2.columns, rotation=90)
        ax.set_yticks(range(mat2.shape[0])); ax.set_yticklabels(mat2.index)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def plot_scatter(df_sc, x, y, title, color_by=None):
    if df_sc is None or df_sc.empty:
        st.warning(f"{title}: empty.")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    if color_by and color_by in df_sc.columns:
        groups = list(df_sc[color_by].unique())
        for g in groups:
            sub = df_sc[df_sc[color_by] == g]
            ax.scatter(sub[x], sub[y], label=str(g), alpha=0.75, s=20)
        if len(groups) <= 12:
            ax.legend()
    else:
        ax.scatter(df_sc[x], df_sc[y], alpha=0.75, s=20)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

def build_matrix(rows, cols, metric="Ra"):
    rows_ok = present_in_db(rows)
    cols_ok = present_in_db(cols)
    if not rows_ok or not cols_ok:
        return pd.DataFrame()
    M = pd.DataFrame(index=rows_ok, columns=cols_ok, dtype=float)
    for r in rows_ok:
        a = df.loc[r]
        for c in cols_ok:
            b = df.loc[c]
            if metric == "Ra":
                M.loc[r,c] = Ra(a,b)
            elif metric == "Delta_a":
                M.loc[r,c] = Delta_a(a,b)
            elif metric == "Pi":
                da = Delta_a(a,b)
                ds = Delta_sigma(a,b)
                M.loc[r,c] = Pi(da, ds)
            else:
                raise ValueError("Unknown metric")
    return M

def build_pairs(A, B, family_label):
    A_ok = present_in_db(A)
    B_ok = present_in_db(B)
    rows = []
    for a_lbl in A_ok:
        a = df.loc[a_lbl]
        for b_lbl in B_ok:
            b = df.loc[b_lbl]
            r = Ra(a,b)
            da = Delta_a(a,b)
            ds = Delta_sigma(a,b)
            p = Pi(da, ds)
            sc = comp_score(r, p, ra_ref, pi_ref)
            rows.append({
                "family": family_label,
                "A": a_lbl,
                "B": b_lbl,
                "Ra": r,
                "Delta_a": da,
                "Delta_sigmaL": ds,
                "Pi": p,
                "Score_%": sc
            })
    return pd.DataFrame(rows)

# ---------------------------
# Tabs: integrated
# ---------------------------
tab_form, tab_hot, tab_figs = st.tabs(["Formulator", "Hotmaps", "Figures 7–8"])

# ============================================================
# TAB 1 — FORMULATOR
# ============================================================
with tab_form:
    st.subheader("1) Pair evaluation")
    left, right = st.columns(2)

    with left:
        A = st.selectbox("Component A (Abbrev)", labels, index=0)
        st.write("**A properties**")
        st.dataframe(df.loc[[A]].reset_index(), use_container_width=True)
    with right:
        B = st.selectbox("Component B (Abbrev)", labels, index=min(1, len(labels)-1))
        st.write("**B properties**")
        st.dataframe(df.loc[[B]].reset_index(), use_container_width=True)

    a = df.loc[A]; b = df.loc[B]
    ra = Ra(a,b)
    da = Delta_a(a,b)
    ds = Delta_sigma(a,b)
    pi = Pi(da, ds)
    sc = comp_score(ra, pi, ra_ref, pi_ref)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Ra", f"{ra:.2f}")
    m2.metric("Δδa", f"{da:.2f}")
    m3.metric("ΔσL", "—" if np.isnan(ds) else f"{ds:.2f}")
    m4.metric("Π", f"{pi:.3f}")
    m5.metric("Score %", f"{sc:.1f}")

    st.divider()
    st.subheader("2) Rank B candidates against a chosen reference A")
    st.caption("This helps a formulator quickly scan best candidates within a target set.")

    purpose = st.selectbox("Purpose set", list(SETS.keys()) + ["Explore-all"])
    topN = st.slider("Top N", 5, 100, 20)

    if purpose == "Explore-all":
        poolA = labels
        poolB = labels
    else:
        # choose which pair-set to rank within this purpose
        pair_names = [f"{p[0]} × {p[2]}" for p in SETS[purpose]["pairs"]]
        pick_pair = st.selectbox("Which set?", pair_names)
        idx = pair_names.index(pick_pair)
        A_name, A_set, B_name, B_set = SETS[purpose]["pairs"][idx]
        poolA = present_in_db(A_set)
        poolB = present_in_db(B_set)

    refA = st.selectbox("Reference A for ranking", poolA if poolA else labels)

    ref = df.loc[refA]
    rows = []
    for b_lbl in poolB:
        if b_lbl == refA:
            continue
        bb = df.loc[b_lbl]
        r = Ra(ref, bb)
        d = Delta_a(ref, bb)
        s = Delta_sigma(ref, bb)
        p = Pi(d,s)
        sc = comp_score(r,p,ra_ref,pi_ref)
        rows.append((b_lbl, sc, r, d, s, p))

    rank = pd.DataFrame(rows, columns=["B", "Score_%", "Ra", "Delta_a", "Delta_sigmaL", "Pi"])\
            .sort_values("Score_%", ascending=False).head(topN)

    st.dataframe(rank, use_container_width=True)

    st.download_button(
        "Download ranking CSV",
        data=rank.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_{purpose}_{refA}.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("3) Stoichiometry helpers (optional)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**PU: NCO/OH index** (requires %NCO and OH# inputs)")
        pct_nco = st.number_input("%NCO (isocyanate)", min_value=0.0, max_value=60.0, value=31.5, step=0.1)
        oh_num  = st.number_input("OH number (mgKOH/g) — formulation blend", min_value=0.0, max_value=800.0, value=200.0, step=1.0)
        # equivalents: 42 g/eq for NCO; OH eq = 56100/OH# (g/eq)
        nco_eq_wt = 42.0
        oh_eq_wt = 56100.0/(oh_num+1e-12)
        st.write(f"OH equivalent weight ≈ {oh_eq_wt:.1f} g/eq")
        # for 100 g iso: eq NCO = (pct_nco/100)*100 / 42
        eq_nco_per100 = (pct_nco/100.0)*100.0 / nco_eq_wt
        # for 100 g polyol blend: eq OH = 100 / (OH eq wt)
        eq_oh_per100  = 100.0 / oh_eq_wt
        idx_nco_oh = eq_nco_per100 / (eq_oh_per100 + 1e-12)
        st.metric("Estimated NCO/OH index (per 100g vs 100g)", f"{idx_nco_oh:.2f}")

    with c2:
        st.markdown("**Epoxy: stoichiometry** (requires EEW + AHEW)")
        eew = st.number_input("EEW (g/eq epoxy)", min_value=50.0, max_value=1000.0, value=190.0, step=1.0)
        ahew = st.number_input("AHEW (g/eq amine-H)", min_value=10.0, max_value=500.0, value=50.0, step=1.0)
        # for 100 g resin: epoxy eq = 100/EEW
        epoxy_eq = 100.0/(eew+1e-12)
        # grams hardener needed for stoich = epoxy_eq * AHEW
        g_hard = epoxy_eq * ahew
        st.metric("Hardener mass for 100 g resin (stoichiometric)", f"{g_hard:.1f} g")

# ============================================================
# TAB 2 — HOTMAPS
# ============================================================
with tab_hot:
    st.subheader("Hotmaps (organized by purpose)")
    st.caption("Heatmaps are generated only with compounds present in your database. No proxies, no artificial labels.")

    purpose_h = st.selectbox("Purpose", list(SETS.keys()))
    metric = st.radio("Metric", ["Ra", "Delta_a", "Pi"], horizontal=True)

    for (A_name, A_set, B_name, B_set) in SETS[purpose_h]["pairs"]:
        A_ok = present_in_db(A_set)
        B_ok = present_in_db(B_set)

        st.markdown(f"### {purpose_h} — {A_name} × {B_name}")
        st.write(f"Rows used: {len(A_ok)} | Columns used: {len(B_ok)}")

        mat = build_matrix(A_ok, B_ok, metric=metric)
        cbar = "Ra (MPa$^{1/2}$)" if metric == "Ra" else ("Δδa (MPa$^{1/2}$)" if metric == "Delta_a" else "Π (normalized)")
        plot_heatmap(mat, f"{purpose_h} — {A_name} × {B_name} ({metric})", cbar)

        st.download_button(
            f"Download matrix CSV — {A_name}×{B_name} ({metric})",
            data=mat.to_csv(index=True).encode("utf-8"),
            file_name=f"HM_{purpose_h}_{A_name}_x_{B_name}_{metric}.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 3 — FIGURES 7–8
# ============================================================
with tab_figs:
    st.subheader("Figures 7–8 (article metrics)")
    st.caption("Scatter plots for screening compatibility: Fig 7 (Ra vs Δδa) and Fig 8 (Ra vs Π).")

    purpose_f = st.selectbox("Purpose", list(SETS.keys()))
    # pick pair set within purpose
    pair_names = [f"{p[0]} × {p[2]}" for p in SETS[purpose_f]["pairs"]]
    pick_pair = st.selectbox("Which set?", pair_names)
    idx = pair_names.index(pick_pair)
    A_name, A_set, B_name, B_set = SETS[purpose_f]["pairs"][idx]

    df_pairs = build_pairs(A_set, B_set, family_label=f"{purpose_f}:{A_name}×{B_name}")

    if df_pairs.empty:
        st.warning("No valid pairs (check if your abbreviations exist in the current DB).")
    else:
        # axis controls
        yopt = st.radio("Y-axis", ["Delta_a (Fig 7)", "Pi (Fig 8)"], horizontal=True)
        ycol = "Delta_a" if yopt.startswith("Delta_a") else "Pi"

        color_by = st.selectbox("Color by", ["none", "A", "B"], index=0)
        cby = None if color_by == "none" else color_by

        plot_scatter(df_pairs, "Ra", ycol, f"{'Figure 7' if ycol=='Delta_a' else 'Figure 8'} — {purpose_f} ({A_name}×{B_name})",
                     color_by=cby)

        st.markdown("### Pair table")
        st.dataframe(df_pairs.sort_values("Score_%", ascending=False), use_container_width=True)

        st.download_button(
            "Download pair table CSV",
            data=df_pairs.to_csv(index=False).encode("utf-8"),
            file_name=f"pairs_{purpose_f}_{A_name}_x_{B_name}.csv",
            mime="text/csv"
        )

        # quick top list (for discussion)
        st.markdown("### Top pairs (highest Score%)")
        topK = st.slider("Top K pairs", 10, 200, 30)
        st.dataframe(df_pairs.sort_values("Score_%", ascending=False).head(topK), use_container_width=True)
