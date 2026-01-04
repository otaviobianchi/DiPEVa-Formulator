# app.py — DiPEVa (Streamlit)
# Database is bundled with the app (Database_final.xlsx in repo root)

import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

st.set_page_config(page_title="DiPEVa — Formulator & Maps", layout="wide")

# -----------------------------
# Utilities: robust column detection
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def find_col(df: pd.DataFrame, keywords):
    cols = list(df.columns)
    ncols = [_norm(c) for c in cols]
    for k in keywords:
        kn = _norm(k)
        for c, nc in zip(cols, ncols):
            if kn in nc:
                return c
    return None

def detect_columns(df: pd.DataFrame):
    col_abbr = find_col(df, ["abbrev", "abbr", "abrev"])
    col_name = find_col(df, ["molecule", "name", "compound", "substance"])
    col_cas  = find_col(df, ["cas"])
    col_smiles = find_col(df, ["smiles"])
    col_vm = find_col(df, ["vm", "cm3", "cm3·mol", "molar volume"])
    col_group = find_col(df, ["group", "family", "class", "classe", "categoria"])

    col_dD = find_col(df, ["δd", "deltad", "delta d"])
    col_dP = find_col(df, ["δp", "deltap", "delta p"])
    col_dH = find_col(df, ["δh", "deltah", "delta h"])
    col_dT = find_col(df, ["δt", "deltat", "delta t"])
    col_sig = find_col(df, ["σl", "sigma", "surface tension", "mn·m-1", "mN·m-1"])
    col_da  = find_col(df, ["δa", "delta a", "associative"])

    meta = dict(
        abbr=col_abbr, name=col_name, cas=col_cas, smiles=col_smiles, vm=col_vm,
        group=col_group, dD=col_dD, dP=col_dP, dH=col_dH, dT=col_dT, sig=col_sig, da=col_da
    )
    return meta

def prep_table(df_raw: pd.DataFrame):
    meta = detect_columns(df_raw)

    # required
    if not all([meta["dD"], meta["dP"], meta["dH"]]):
        raise KeyError(f"Could not detect δD/δP/δH columns. Found columns: {list(df_raw.columns)}")

    key = meta["abbr"] or meta["name"]
    if key is None:
        raise KeyError("Could not detect identifier column (Abbrev or Molecule/Name).")

    keep = [c for c in [
        meta["abbr"], meta["name"], meta["cas"], meta["smiles"], meta["vm"], meta["group"],
        meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"]
    ] if c is not None and c in df_raw.columns]

    T = df_raw[keep].copy()

    # numeric coercion
    for c in [meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"], meta["vm"]]:
        if c is not None and c in T.columns:
            T[c] = pd.to_numeric(T[c], errors="coerce")

    # compute δa if missing/empty: δa = sqrt(δP^2 + δH^2)
    if meta["da"] is None or (meta["da"] in T.columns and T[meta["da"]].isna().all()):
        meta["da"] = "delta_a_calc"
        T[meta["da"]] = np.sqrt(T[meta["dP"]]**2 + T[meta["dH"]]**2)

    # clean key
    T[key] = T[key].astype(str).str.strip()
    T = T[T[key].str.len() > 0].copy()

    # keep valid HSP rows
    T = T.dropna(subset=[meta["dD"], meta["dP"], meta["dH"]]).copy()

    # aggregate duplicate identifiers (mean for numerics, first for strings)
    num_cols = [c for c in [meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"], meta["vm"]] if c is not None and c in T.columns]
    agg = {c: "mean" for c in num_cols}
    for c in [meta["name"], meta["cas"], meta["smiles"], meta["group"]]:
        if c is not None and c in T.columns:
            agg[c] = "first"

    T = T.groupby(key, as_index=False).agg(agg).set_index(key)

    meta["key"] = key
    return T, meta

# -----------------------------
# Metrics (as in manuscript)
# -----------------------------
def Ra(a, b, dD, dP, dH):
    return float(np.sqrt(
        4.0*(a[dD]-b[dD])**2 +
        (a[dP]-b[dP])**2 +
        (a[dH]-b[dH])**2
    ))

def delta_a(a, b, da):
    # Δδa = |δa_A - δa_B| ; δa = sqrt(δP^2+δH^2)
    return float(abs(a[da]-b[da]))

def delta_sigmaL(a, b, sig):
    if sig is None:
        return np.nan
    if sig not in a.index or sig not in b.index:
        return np.nan
    return float(abs(a[sig]-b[sig]))

def Pi(da_val, ds_val, da_max, sig_max):
    # Π = Δδa/δa,max + ΔσL/σL,max  (no weights)
    if np.isnan(ds_val) or np.isnan(sig_max):
        return float(da_val/(da_max+1e-12))
    return float(da_val/(da_max+1e-12) + ds_val/(sig_max+1e-12))

# -----------------------------
# Plot helpers
# -----------------------------
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plot_heatmap(mat, title, cbar_label):
    fig, ax = plt.subplots(figsize=(max(8, 0.35*mat.shape[1]), max(4, 0.35*mat.shape[0])))
    if _HAS_SNS:
        sns.heatmap(mat, cmap="viridis", cbar_kws={"label": cbar_label}, ax=ax)
    else:
        im = ax.imshow(mat.values, aspect="auto")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_scatter(df_sc, x, y, hue, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue in df_sc.columns:
        for g in sorted(df_sc[hue].unique()):
            sub = df_sc[df_sc[hue] == g]
            ax.scatter(sub[x], sub[y], label=str(g), alpha=0.75, s=22)
        if len(df_sc[hue].unique()) <= 12:
            ax.legend()
    else:
        ax.scatter(df_sc[x], df_sc[y], alpha=0.75, s=22)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

# -----------------------------
# Purpose sets (Abbrev labels expected)
# -----------------------------
SETS = {
    "Polyesters": {
        "A_name": "Polyols",
        "B_name": "Acids/Anhydrides",
        "A": ["EG","DEG","TEG","PG","DPG","1,3-BDO","1,4-BDO","1,5-PeDO","1,6-HDO","NPG","GLY","TMP","PER","SOR",
              "PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PCDL500","PCDL1000","PCL1000","PPG500","PTMEG1000"],
        "B": ["SA","AA","GA","PA","SuA","SeA","AzA","FA","IPA","TPA","PhA","MAH"],
    },
    "Polyurethanes (PU)": {
        "A_name": "Polyols / Chain extenders",
        "B_name": "Isocyanates",
        "A": ["PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PPG500","PTMEG1000","PCDL500","PCDL1000","PCL1000",
              "PG","DPG","MPG","TEG","EG","1,4-BDO","1,6-HDO","EDA","HMDA","MEA","DEA","TEA"],
        "B": ["HDI","IPDI","H12MDI","MDI 2,4′","MDI 4,4′","TDI 2,4","TDI 2,6","TDI 80/20","pMDI (polymeric MDI)"],
        "subgroups": {
            "PU_polyols": ["PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PPG500","PTMEG1000","PCDL500","PCDL1000","PCL1000","PG","DPG","MPG","TEG","EG","1,4-BDO","1,6-HDO"],
            "PU_extenders": ["1,4-BDO","EDA","HMDA","MEA","DEA","TEA"]
        }
    },
    "Epoxies": {
        "A_name": "Hardeners / Reactive diluents",
        "B_name": "Resins",
        "A": ["DDM","DDS","DETA","TETA","DICY","EDA","HMDA","MEA","DEA","TEA","ECH","ECA","PCA","GBL"],
        "B": ["DGEBA", "BDGE (aliphatic epoxy)"],
        "subgroups": {
            "Epoxy_hardeners": ["DDM","DDS","DETA","TETA","DICY","EDA","HMDA","MEA","DEA","TEA"],
            "Epoxy_diluents": ["ECH","ECA","PCA","GBL"]
        }
    },
    "Vinylics": {
        "A_name": "Monomers",
        "B_name": "Solvents / Plasticizers",
        "A": ["St","α-MS","AN","MMA","FN","VAc","MAA"],
        "B": ["DMP","DEP","DBP","DEHP","TBP","TCP","DMC","DEC","MB","EB","MS","TAC","DMA","DMM","DMO","DEHA","IPM","IPP","BS",
              "MeOH","EtOH","n-PrOH","i-PrOH","2-EH","BnOH","CyOH","EGME","Diglyme"]
    },
    "Silanes": {
        "A_name": "Silanes / Siloxanes",
        "B_name": "Monomers / Solvents",
        "A": ["VTMS","TEOS","APTES","MPTES","BTESPD","BTMPSD","TMS-OH","EMNS"],
        "B": ["St","α-MS","AN","MMA","FN","VAc","MAA","MeOH","EtOH","EGME","Diglyme","MB","EB","DMP","DEP","DBP","DEHP"]
    }
}

# -----------------------------
# Load internal database
# -----------------------------
@st.cache_data
def load_db():
    fname = "Database_final.xlsx"
    if not os.path.exists(fname):
        # fallback: allow running from other cwd
        raise FileNotFoundError("Database_final.xlsx not found in repo root.")
    db = pd.read_excel(fname, sheet_name=None)
    return db

db = load_db()
sheet_default = None
for s in ["compounds_primary", "compounds", "database", "data", "Sheet1", "sheet1"]:
    if s in db:
        sheet_default = s
        break
if sheet_default is None:
    sheet_default = list(db.keys())[0]

# Sidebar controls
st.sidebar.header("Controls")
sheet = st.sidebar.selectbox("Database sheet", list(db.keys()), index=list(db.keys()).index(sheet_default), key="sb_sheet")

try:
    T, meta = prep_table(db[sheet])
except Exception as e:
    st.error(f"Failed to parse sheet '{sheet}': {e}")
    st.stop()

labels = list(T.index)
da_max = float(np.nanmax(T[meta["da"]].values))
sig_max = float(np.nanmax(T[meta["sig"]].values)) if (meta["sig"] and T[meta["sig"]].notna().any()) else np.nan

# -----------------------------
# Header + didactic overview
# -----------------------------
st.title("DiPEVa — Formulator & Compatibility Maps")
st.caption("Internal database bundled with the app (no upload). Use: Ra, Δδₐ and Π for compatibility screening.")

with st.expander("Equations used (didactic)", expanded=False):
    st.markdown(r"""
**Hansen distance (Ra)**  
\[
R_a = \sqrt{4(\delta_D^A-\delta_D^B)^2+(\delta_P^A-\delta_P^B)^2+(\delta_H^A-\delta_H^B)^2}
\]

**Associative parameter**  
\[
\delta_a = \sqrt{\delta_P^2+\delta_H^2}
\]
\[
\Delta \delta_a = |\delta_a^A-\delta_a^B|
\]

**Surface tension mismatch**  
\[
\Delta \sigma_L = |\sigma_L^A-\sigma_L^B|
\]

**Interfacial affinity index (Π) — as in the manuscript (no weights)**  
\[
\Pi=\frac{\Delta\delta_a}{\delta_{a,\max}}+\frac{\Delta\sigma_L}{\sigma_{L,\max}}
\]
""")

# -----------------------------
# Layout tabs
# -----------------------------
tab_overview, tab_formulator, tab_heatmaps, tab_figures = st.tabs(
    ["Overview", "Formulator", "Heatmaps", "FIGURES: Ra vs Δδₐ and Ra vs Π"]
)

# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("How to read the outputs")
    st.write(
        "Use **Ra** for a first-pass thermodynamic similarity. Use **Δδₐ** and **Π** to capture "
        "specific interactions and interfacial mismatch. Lower values generally indicate higher compatibility."
    )
    st.markdown("**Database summary**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entries", f"{len(T)}")
    c2.metric("δa,max", f"{da_max:.2f}")
    if np.isnan(sig_max):
        c3.metric("σL,max", "missing")
    else:
        c3.metric("σL,max", f"{sig_max:.2f}")

    st.markdown("**Detected columns**")
    st.json({k: v for k, v in meta.items() if k != "key"})

# -----------------------------
# Formulator tab
# -----------------------------
def show_props(lbl, key_prefix):
    row = T.loc[lbl]
    props = {}
    for k in ["name","cas","smiles","vm","dD","dP","dH","dT","sig","da"]:
        col = meta.get(k)
        if col is None:
            continue
        val = row[col]
        if pd.isna(val):
            props[k] = None
        else:
            props[k] = float(val) if isinstance(val, (int,float,np.floating)) else str(val)
    st.markdown(f"**Selected:** `{lbl}`")
    st.json(props)

with tab_formulator:
    st.subheader("Pair evaluation (didactic)")
    purpose_f = st.selectbox("Purpose", list(SETS.keys()), key="purpose_formulator")

    A_list = [x for x in SETS[purpose_f]["A"] if x in T.index]
    B_list = [x for x in SETS[purpose_f]["B"] if x in T.index]
    if not A_list or not B_list:
        st.warning("No matching labels for this purpose were found in the database. Check abbreviations.")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        A = st.selectbox(f"{SETS[purpose_f]['A_name']} (A)", A_list, key="form_A")
        show_props(A, "A")
    with colB:
        B = st.selectbox(f"{SETS[purpose_f]['B_name']} (B)", B_list, key="form_B")
        show_props(B, "B")

    a = T.loc[A]; b = T.loc[B]
    ra = Ra(a,b, meta["dD"], meta["dP"], meta["dH"])
    da = delta_a(a,b, meta["da"])
    ds = delta_sigmaL(a,b, meta["sig"])
    pi = Pi(da, ds, da_max, sig_max)

    st.markdown("### Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ra (MPa$^{1/2}$)", f"{ra:.2f}")
    m2.metric("Δδₐ (MPa$^{1/2}$)", f"{da:.2f}")
    m3.metric("ΔσL (mN/m)", "NaN" if np.isnan(ds) else f"{ds:.2f}")
    m4.metric("Π (normalized)", f"{pi:.3f}")

    st.caption("Interpretation: lower Ra, lower Δδₐ and lower Π suggest higher compatibility (screening).")

# -----------------------------
# Heatmaps tab
# -----------------------------
def build_matrix(rows, cols, metric):
    rows_ok = [r for r in rows if r in T.index]
    cols_ok = [c for c in cols if c in T.index]
    if not rows_ok or not cols_ok:
        return pd.DataFrame()

    M = pd.DataFrame(index=rows_ok, columns=cols_ok, dtype=float)
    for r in rows_ok:
        ar = T.loc[r]
        for c in cols_ok:
            bc = T.loc[c]
            if metric == "Ra":
                M.loc[r, c] = Ra(ar, bc, meta["dD"], meta["dP"], meta["dH"])
            elif metric == "Δδa":
                M.loc[r, c] = delta_a(ar, bc, meta["da"])
            elif metric == "Π":
                da_ = delta_a(ar, bc, meta["da"])
                ds_ = delta_sigmaL(ar, bc, meta["sig"])
                M.loc[r, c] = Pi(da_, ds_, da_max, sig_max)
            else:
                raise ValueError("unknown metric")
    return M

with tab_heatmaps:
    st.subheader("Heatmaps (by purpose)")
    purpose_h = st.selectbox("Purpose", list(SETS.keys()), key="purpose_heatmaps")

    metric = st.radio("Metric", ["Ra", "Δδa", "Π"], horizontal=True, key="hm_metric")

    rows = SETS[purpose_h]["A"]
    cols = SETS[purpose_h]["B"]

    # limit sizes for UI (still can export full if needed later)
    max_rows = st.slider("Max rows shown", 10, 80, 40, key="hm_max_rows")
    max_cols = st.slider("Max columns shown", 10, 80, 30, key="hm_max_cols")

    rows_ok = [r for r in rows if r in T.index][:max_rows]
    cols_ok = [c for c in cols if c in T.index][:max_cols]

    M = build_matrix(rows_ok, cols_ok, metric)
    if M.empty or M.shape[0] < 2 or M.shape[1] < 2:
        st.warning("Heatmap too small (not enough matching labels).")
    else:
        cbar = "Ra (MPa$^{1/2}$)" if metric == "Ra" else ("Δδₐ (MPa$^{1/2}$)" if metric == "Δδa" else "Π (normalized)")
        fig = plot_heatmap(M, f"{purpose_h} — {metric} heatmap", cbar)
        st.pyplot(fig, clear_figure=True)
        st.download_button(
            "Download PNG",
            data=fig_to_png_bytes(fig),
            file_name=f"HM_{purpose_h}_{metric}.png",
            mime="image/png",
            key=f"dl_hm_{purpose_h}_{metric}"
        )

# -----------------------------
# FIGURES tab (Scatter)
# -----------------------------
def build_pairs(purpose):
    spec = SETS[purpose]
    subgroups = spec.get("subgroups", None)

    rows = []
    if subgroups:
        # Pair each subgroup against B list, mark subgroup
        B_ok = [b for b in spec["B"] if b in T.index]
        for gname, A_list in subgroups.items():
            A_ok = [a for a in A_list if a in T.index]
            for a_id in A_ok:
                a = T.loc[a_id]
                for b_id in B_ok:
                    b = T.loc[b_id]
                    ra = Ra(a,b, meta["dD"], meta["dP"], meta["dH"])
                    da = delta_a(a,b, meta["da"])
                    ds = delta_sigmaL(a,b, meta["sig"])
                    pi = Pi(da, ds, da_max, sig_max)
                    rows.append({"group": gname, "A": a_id, "B": b_id, "Ra": ra, "delta_a": da, "Pi": pi})
    else:
        A_ok = [a for a in spec["A"] if a in T.index]
        B_ok = [b for b in spec["B"] if b in T.index]
        for a_id in A_ok:
            a = T.loc[a_id]
            for b_id in B_ok:
                b = T.loc[b_id]
                ra = Ra(a,b, meta["dD"], meta["dP"], meta["dH"])
                da = delta_a(a,b, meta["da"])
                ds = delta_sigmaL(a,b, meta["sig"])
                pi = Pi(da, ds, da_max, sig_max)
                rows.append({"group": purpose, "A": a_id, "B": b_id, "Ra": ra, "delta_a": da, "Pi": pi})
    return pd.DataFrame(rows)

with tab_figures:
    st.subheader("FIGURES: Ra vs Δδₐ and Ra vs Π")
    purpose_p = st.selectbox("Purpose", list(SETS.keys()), key="purpose_figures")

    df_pairs = build_pairs(purpose_p)
    if df_pairs.empty:
        st.warning("No valid pairs for this purpose (check abbreviations in the database).")
        st.stop()

    st.caption("These plots replicate the manuscript logic, with purpose-based grouping.")
    # Figure 1: Ra vs Δδa
    fig1 = plot_scatter(
        df_pairs, "Ra", "delta_a", "group",
        title=f"{purpose_p} — Ra vs Δδₐ",
        xlabel="Ra (MPa$^{1/2}$)",
        ylabel="Δδₐ (MPa$^{1/2}$)"
    )
    st.pyplot(fig1, clear_figure=True)
    st.download_button(
        "Download PNG (Ra vs Δδₐ)",
        data=fig_to_png_bytes(fig1),
        file_name=f"FIG_Ra_vs_deltaa_{purpose_p}.png",
        mime="image/png",
        key=f"dl_fig1_{purpose_p}"
    )

    # Figure 2: Ra vs Π
    fig2 = plot_scatter(
        df_pairs, "Ra", "Pi", "group",
        title=f"{purpose_p} — Ra vs Π",
        xlabel="Ra (MPa$^{1/2}$)",
        ylabel="Π (normalized)"
    )
    st.pyplot(fig2, clear_figure=True)
    st.download_button(
        "Download PNG (Ra vs Π)",
        data=fig_to_png_bytes(fig2),
        file_name=f"FIG_Ra_vs_Pi_{purpose_p}.png",
        mime="image/png",
        key=f"dl_fig2_{purpose_p}"
    )

    with st.expander("Show pair table (copy/export)"):
        st.dataframe(df_pairs, use_container_width=True)
        st.download_button(
            "Download CSV (pairs)",
            data=df_pairs.to_csv(index=False).encode("utf-8"),
            file_name=f"pairs_{purpose_p}.csv",
            mime="text/csv",
            key=f"dl_pairs_{purpose_p}"
        )
