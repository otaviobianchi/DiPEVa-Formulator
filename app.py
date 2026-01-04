import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

st.set_page_config(page_title="DiPEVa — Formulator", layout="wide")
st.title("DiPEVa — Formulator & Compatibility Maps")
st.caption("Database is internal to the app (Database_final.xlsx).")

# -----------------------------
# Column detection / table prep
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
    meta = {}
    meta["abbr"] = find_col(df, ["abbrev", "abbr", "abrev"])
    meta["name"] = find_col(df, ["molecule", "name", "compound", "substance"])
    meta["cas"]  = find_col(df, ["cas"])
    meta["smiles"] = find_col(df, ["smiles"])
    meta["vm"]   = find_col(df, ["vm", "cm3", "cm3·mol", "molar volume"])
    meta["group"]= find_col(df, ["group", "family", "class", "classe", "categoria"])

    meta["dD"]   = find_col(df, ["δd", "deltad", "delta d"])
    meta["dP"]   = find_col(df, ["δp", "deltap", "delta p"])
    meta["dH"]   = find_col(df, ["δh", "deltah", "delta h"])
    meta["dT"]   = find_col(df, ["δt", "deltat", "delta t"])
    meta["sig"]  = find_col(df, ["σl", "sigma", "surface tension", "mn·m-1", "mN·m-1"])
    meta["da"]   = find_col(df, ["δa", "delta a", "associative"])
    return meta

def prep_table(df_raw: pd.DataFrame):
    meta = detect_columns(df_raw)
    if not all([meta["dD"], meta["dP"], meta["dH"]]):
        raise KeyError("Could not detect δD/δP/δH columns in the selected sheet.")
    key = meta["abbr"] or meta["name"]
    if key is None:
        raise KeyError("Could not detect identifier column (Abbrev or Molecule).")
    keep = [c for c in [
        meta["abbr"], meta["name"], meta["cas"], meta["smiles"], meta["vm"], meta["group"],
        meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"]
    ] if c is not None and c in df_raw.columns]
    T = df_raw[keep].copy()

    for c in [meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"], meta["vm"]]:
        if c is not None and c in T.columns:
            T[c] = pd.to_numeric(T[c], errors="coerce")

    if meta["da"] is None or (meta["da"] in T.columns and T[meta["da"]].isna().all()):
        meta["da"] = "delta_a_calc"
        T[meta["da"]] = np.sqrt(T[meta["dP"]]**2 + T[meta["dH"]]**2)

    T[key] = T[key].astype(str).str.strip()
    T = T[T[key].str.len() > 0].dropna(subset=[meta["dD"], meta["dP"], meta["dH"]]).copy()

    # aggregate duplicates
    num_cols = [c for c in [meta["dD"], meta["dP"], meta["dH"], meta["dT"], meta["sig"], meta["da"], meta["vm"]] if c is not None and c in T.columns]
    agg = {c: "mean" for c in num_cols}
    for c in [meta["name"], meta["cas"], meta["smiles"], meta["group"]]:
        if c is not None and c in T.columns:
            agg[c] = "first"
    T = T.groupby(key, as_index=False).agg(agg).set_index(key)
    meta["key"] = key
    return T, meta

# -----------------------------
# Metrics (manuscript-consistent)
# -----------------------------
def Ra(a, b, dD, dP, dH):
    return float(np.sqrt(4.0*(a[dD]-b[dD])**2 + (a[dP]-b[dP])**2 + (a[dH]-b[dH])**2))

def delta_a(a, b, da):
    return float(abs(a[da]-b[da]))

def delta_sigmaL(a, b, sig):
    if sig is None:
        return np.nan
    if sig not in a.index or sig not in b.index:
        return np.nan
    return float(abs(a[sig]-b[sig]))

def Pi(da_val, ds_val, da_max, sig_max):
    # Π = Δδa/δa,max + ΔσL/σL,max   (no weights)
    if np.isnan(ds_val) or np.isnan(sig_max) or sig_max <= 0:
        return float(da_val/(da_max+1e-12))
    return float(da_val/(da_max+1e-12) + ds_val/(sig_max+1e-12))

def score_percent(ra, pi, da, ra_max, pi_max, da_max):
    # Compatibility score in % (higher is better), fixed equal weighting:
    # score = 100 * (1 - mean(normalized distances))
    n_ra = ra/(ra_max+1e-12)
    n_pi = pi/(pi_max+1e-12)
    n_da = da/(da_max+1e-12)
    s = 100.0*(1.0 - (n_ra + n_pi + n_da)/3.0)
    return float(np.clip(s, 0, 100))

# -----------------------------
# Plot helpers
# -----------------------------
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def heatmap(mat, title, cbar_label):
    fig, ax = plt.subplots(figsize=(max(8, 0.35*mat.shape[1]), max(4, 0.35*mat.shape[0])))
    if _HAS_SNS:
        sns.heatmap(mat, cmap="viridis", cbar_kws={"label": cbar_label}, ax=ax)
    else:
        im = ax.imshow(mat.values, aspect="auto")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def scatter(df_sc, x, y, hue, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8,6))
    if hue in df_sc.columns:
        for g in sorted(df_sc[hue].unique()):
            sub = df_sc[df_sc[hue]==g]
            ax.scatter(sub[x], sub[y], label=str(g), alpha=0.75, s=22)
        if len(df_sc[hue].unique()) <= 12:
            ax.legend()
    else:
        ax.scatter(df_sc[x], df_sc[y], alpha=0.75, s=22)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    return fig

# -----------------------------
# Purpose sets (Abbrev labels)
# -----------------------------
SETS = {
    "Epoxies": {
        "A_name": "Hardeners / Reactive diluents",
        "B_name": "Resins",
        "A": ["DDM","DDS","DETA","TETA","DICY","EDA","HMDA","MEA","DEA","TEA","ECH","ECA","PCA","GBL"],
        "B": ["DGEBA", "BDGE (aliphatic epoxy)", "BDGE"],
        "subgroups": {
            "Hardeners": ["DDM","DDS","DETA","TETA","DICY","EDA","HMDA","MEA","DEA","TEA"],
            "Reactive diluents": ["ECH","ECA","PCA","GBL"]
        }
    },
    "Polyurethanes (PU)": {
        "A_name": "Polyols / Chain extenders",
        "B_name": "Isocyanates",
        "A": ["PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PPG500","PTMEG1000","PCDL500","PCDL1000","PCL1000",
              "PG","DPG","MPG","TEG","EG","1,4-BDO","1,6-HDO","EDA","HMDA","MEA","DEA","TEA"],
        "B": ["HDI","IPDI","H12MDI","MDI 2,4′","MDI 4,4′","TDI 2,4","TDI 2,6","TDI 80/20","pMDI (polymeric MDI)"],
        "subgroups": {
            "Polyols": ["PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PPG500","PTMEG1000","PCDL500","PCDL1000","PCL1000","PG","DPG","MPG","TEG","EG","1,4-BDO","1,6-HDO"],
            "Chain extenders": ["1,4-BDO","EDA","HMDA","MEA","DEA","TEA"]
        }
    },
    "Polyesters": {
        "A_name": "Polyols",
        "B_name": "Acids/Anhydrides",
        "A": ["EG","DEG","TEG","PG","DPG","1,3-BDO","1,4-BDO","1,5-PeDO","1,6-HDO","NPG","GLY","TMP","PER","SOR",
              "PEG200","PEG300","PEG400","PEG600","PEG900","PEG1000","PCDL500","PCDL1000","PCL1000","PPG500","PTMEG1000"],
        "B": ["SA","AA","GA","PA","SuA","SeA","AzA","FA","IPA","TPA","PhA","MAH"],
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
# Load internal DB
# -----------------------------
@st.cache_data
def load_db():
    fname = "Database_final.xlsx"
    if not os.path.exists(fname):
        raise FileNotFoundError("Database_final.xlsx not found in repo root.")
    return pd.read_excel(fname, sheet_name=None)

@st.cache_data
def load_equivalents():
    fname = "equivalents_library.csv"
    if os.path.exists(fname):
        return pd.read_csv(fname)
    # fallback empty
    return pd.DataFrame(columns=["Abbrev","Kind","NCO_wt_percent","OH_mgKOH_g","EEW_g_eq","AHEW_g_eq","Notes"])

db = load_db()
eq0 = load_equivalents()

default_sheet = list(db.keys())[0]
for s in ["compounds_primary", "compounds", "database", "data", "Sheet1", "sheet1"]:
    if s in db:
        default_sheet = s
        break

st.sidebar.header("Database")
sheet = st.sidebar.selectbox("Sheet", list(db.keys()), index=list(db.keys()).index(default_sheet), key="sb_sheet")

T, meta = prep_table(db[sheet])

# -----------------------------
# Label resolver (robust to aliases/spacing)
# -----------------------------
def _simplify_label(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("′","'").replace("’","'").replace("“","\"").replace("”","\"")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9\-\(\)\/_\+\. ]+", "", x)
    return x.strip()

# Common aliases (extend as needed)
ALIASES = {
    "bdge": ["BDGE", "BDGE (aliphatic epoxy)", "aliphatic epoxy", "aliphatic epoxy resin"],
    "dgeba": ["DGEBA", "aromatic epoxy", "aromatic epoxy resin"],
    "h2o": ["H2O", "H₂O", "Water"],
    "mdi 4,4'": ["MDI 4,4′", "MDI 4,4'", "4,4-MDI", "4,4 MDI"],
    "mdi 2,4'": ["MDI 2,4′", "MDI 2,4'", "2,4-MDI", "2,4 MDI"],
    "tdi 2,4": ["TDI 2,4", "2,4-TDI", "2.4-TDI"],
    "tdi 2,6": ["TDI 2,6", "2,6-TDI", "2.6-TDI"],
    "tdi 80/20": ["TDI 80/20", "TDI80/20", "80/20 TDI"],
}

def resolve_label(wanted: str):
    if wanted is None:
        return None
    idx = list(T.index.astype(str))
    simp_map = { _simplify_label(v): v for v in idx }
    w = _simplify_label(wanted)

    if w in simp_map:
        return simp_map[w]

    for _, vals in ALIASES.items():
        if w in [_simplify_label(v) for v in vals]:
            for v in vals:
                sv = _simplify_label(v)
                if sv in simp_map:
                    return simp_map[sv]

    for s, orig in simp_map.items():
        if w and (w in s or s in w):
            return orig
    return None

def resolve_list(cands):
    out=[]
    for c in cands:
        r = resolve_label(c)
        if r is not None:
            out.append(r)
    seen=set()
    uniq=[]
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

da_max = float(np.nanmax(T[meta["da"]].values))
sig_max = float(np.nanmax(T[meta["sig"]].values)) if (meta["sig"] and T[meta["sig"]].notna().any()) else np.nan

# -----------------------------
# Equivalents library (editable)
# -----------------------------
st.sidebar.header("Equivalents library")
if "eq_lib" not in st.session_state:
    st.session_state.eq_lib = eq0.copy()

eq_lib = st.session_state.eq_lib

with st.sidebar.expander("Edit / Save equivalents", expanded=False):
    st.write("Edit values and download the updated CSV to keep it in your repo.")
    eq_edit = st.data_editor(eq_lib, num_rows="dynamic", use_container_width=True, key="eq_editor")
    # enforce 1-decimal storage for constants
    for _c in ["NCO_wt_percent","OH_mgKOH_g","EEW_g_eq","AHEW_g_eq"]:
        if _c in eq_edit.columns:
            eq_edit[_c] = pd.to_numeric(eq_edit[_c], errors="coerce")
    st.session_state.eq_lib = eq_edit
    st.download_button("Download equivalents_library.csv",
                       eq_edit.to_csv(index=False).encode("utf-8"),
                       file_name="equivalents_library.csv",
                       mime="text/csv",
                       key="dl_eq_csv")

def eq_lookup(abbrev: str):
    df = st.session_state.eq_lib
    if df.empty:
        return {}
    m = df[df["Abbrev"].astype(str) == str(abbrev)]
    if m.empty:
        return {}
    r = m.iloc[0].to_dict()
    return r

# Quick search helper
def label_select(label, options, key):
    query = st.text_input(f"Search {label}", value="", key=f"{key}_q")
    opts = options
    if query.strip():
        q = query.strip().lower()
        opts = [o for o in options if q in str(o).lower()]
    return st.selectbox(label, opts, key=key)

def selected_card(lbl):
    r = T.loc[lbl]
    rows = []

    def add_num(title, val, fmt):
        if val is None or (isinstance(val, (float, int, np.floating)) and np.isnan(val)):
            return
        rows.append((title, fmt.format(float(val))))

    def add_txt(title, val):
        if isinstance(val, str) and val.strip():
            rows.append((title, val.strip()))

    def get(colkey):
        c = meta.get(colkey)
        if c is None:
            return None
        return r.get(c, np.nan)

    # Text fields
    add_txt("Molecule", get("name"))
    add_txt("CAS", get("cas"))
    add_txt("SMILES/BigSMILES", get("smiles"))

    # Numeric fields (formatting: solubility params with 2 decimals)
    add_num("Vm (cm³/mol)", get("vm"), "{:.1f}")
    add_num("δD", get("dD"), "{:.2f}")
    add_num("δP", get("dP"), "{:.2f}")
    add_num("δH", get("dH"), "{:.2f}")
    add_num("δT", get("dT"), "{:.2f}")
    add_num("σL (mN/m)", get("sig"), "{:.2f}")
    add_num("δa", get("da"), "{:.2f}")

    df = pd.DataFrame(rows, columns=["Property","Value"])
    st.markdown(f"**Selected:** `{lbl}`")
    st.dataframe(df, hide_index=True, use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab_form, tab_maps, tab_figs = st.tabs(["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"])

# -----------------------------
# Helpers for ranking
# -----------------------------
def build_candidates(purpose, A_list, B_list, group_name):
    rows=[]
    A_ok=[a for a in A_list if a in T.index]
    B_ok=[b for b in B_list if b in T.index]
    for a_id in A_ok:
        a = T.loc[a_id]
        for b_id in B_ok:
            b = T.loc[b_id]
            ra = Ra(a,b,meta["dD"],meta["dP"],meta["dH"])
            da = delta_a(a,b,meta["da"])
            ds = delta_sigmaL(a,b,meta["sig"])
            pi = Pi(da, ds, da_max, sig_max)
            rows.append({"group": group_name, "A": a_id, "B": b_id, "Ra": ra, "Δδa": da, "Π": pi})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ra_max = float(df["Ra"].max())
    pi_max = float(df["Π"].max())
    daM = float(df["Δδa"].max())
    df["Score_%"] = [score_percent(r, p, d, ra_max, pi_max, daM) for r,p,d in zip(df["Ra"], df["Π"], df["Δδa"])]
    return df.sort_values("Score_%", ascending=False)

# -----------------------------
# Formulator (improved)
# -----------------------------
with tab_form:
    st.subheader("Formulator")
    purpose = st.selectbox("Purpose", list(SETS.keys()), key="form_purpose")
    spec = SETS[purpose]
    # Mode options depend on purpose (avoid mixing PU/Epoxy widgets)
modes = ["Pair screening", "Ranking"]
if purpose == "Polyurethanes (PU)":
    modes.insert(1, "PU stoichiometry (NCO/OH)")
if purpose == "Epoxies":
    modes.insert(1, "Epoxy stoichiometry (EEW/AHEW)")
mode = st.radio("Mode", modes, horizontal=True, key=f"form_mode_{purpose}")

A_all = resolve_list(spec["A"])
B_all = resolve_list(spec["B"])

if mode == "Pair screening":
        c1, c2 = st.columns(2)
        with c1:
            A = label_select(f"{spec['A_name']} (A)", A_all, key=f"pair_A_{purpose}")
            selected_card(A)
        with c2:
            B = label_select(f"{spec['B_name']} (B)", B_all, key=f"pair_B_{purpose}")
            selected_card(B)

        a = T.loc[A]; b = T.loc[B]
        ra = Ra(a,b, meta["dD"], meta["dP"], meta["dH"])
        da = delta_a(a,b, meta["da"])
        ds = delta_sigmaL(a,b, meta["sig"])
        pi = Pi(da, ds, da_max, sig_max)

        # Score % (within this single pair, normalize by dataset ranges)
        ra_max = float(np.nanmax(np.sqrt(4*(T[meta["dD"]]-T[meta["dD"]].median())**2 + (T[meta["dP"]]-T[meta["dP"]].median())**2 + (T[meta["dH"]]-T[meta["dH"]].median())**2)))
        pi_max = float((da_max/(da_max+1e-12)) + (0 if np.isnan(sig_max) else sig_max/(sig_max+1e-12)))
        sc = score_percent(ra, pi, da, ra_max, pi_max, da_max)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Ra", f"{ra:.2f}")
        m2.metric("Δδa", f"{da:.2f}")
        m3.metric("ΔσL", "NaN" if np.isnan(ds) else f"{ds:.2f}")
        m4.metric("Π", f"{pi:.3f}")
        m5.metric("Score (%)", f"{sc:.1f}")

elif mode == "PU stoichiometry (NCO/OH)":
        st.markdown("### PU stoichiometry")
        st.caption("Uses **total batch mass target** OR polyol-side mass. Inputs can be auto-filled from the equivalents library.")

        isos = resolve_list(spec["B"])
        polyols = resolve_list(spec.get("subgroups", {}).get("Polyols", spec["A"]))
        extenders = resolve_list(spec.get("subgroups", {}).get("Chain extenders", []))

        left, right = st.columns(2)
        with left:
            iso = label_select("Isocyanate", isos, key=f"pu_iso_{purpose}")
            selected_card(iso)
        with right:
            poly = label_select("Main polyol", polyols, key=f"pu_poly_{purpose}")
            selected_card(poly)

        use_ext = st.checkbox("Add chain extender", value=True, key=f"pu_use_ext_{purpose}")
        ext = None
        if use_ext and extenders:
            ext = label_select("Chain extender", extenders, key=f"pu_ext_{purpose}")
            selected_card(ext)

        # Autofill
        iso_eq = eq_lookup(iso)
        poly_eq = eq_lookup(poly)
        ext_eq = eq_lookup(ext) if ext else {}

        auto = st.checkbox("Auto-fill from equivalents library (recommended)", value=True, key=f"pu_auto_{purpose}")

        def _num(val, fallback):
            try:
                if pd.isna(val): return fallback
                return float(val)
            except Exception:
                return fallback

        nco_default = _num(iso_eq.get("NCO_wt_percent", np.nan), 33.0)
        oh_poly_default = _num(poly_eq.get("OH_mgKOH_g", np.nan), 56.0)
        oh_ext_default = _num(ext_eq.get("OH_mgKOH_g", np.nan), 1200.0)

        st.markdown("#### Stoichiometry inputs")
        c1,c2,c3 = st.columns(3)
        with c1:
            nco_percent = st.number_input("%NCO of isocyanate (wt%)", min_value=0.0, max_value=60.0,
                                          value=nco_default if auto else 33.0, step=0.1, key=f"pu_nco_{purpose}")
        with c2:
            oh_poly = st.number_input("OH number of polyol (mgKOH/g)", min_value=0.0, max_value=2000.0,
                                      value=oh_poly_default if auto else 56.0, step=1.0, key=f"pu_oh_poly_{purpose}")
        with c3:
            oh_ext = st.number_input("OH number of extender (mgKOH/g)", min_value=0.0, max_value=2000.0,
                                     value=oh_ext_default if (auto and use_ext and ext) else (1200.0 if use_ext else 0.0),
                                     step=1.0, key=f"pu_oh_ext_{purpose}")

        idx = st.number_input("NCO/OH index (ratio of equivalents)", min_value=0.5, max_value=2.5,
                              value=1.05, step=0.01, key=f"pu_index_{purpose}")

        mass_mode = st.radio("Mass target mode", ["Total mass (includes isocyanate)", "Polyol-side mass (polyol+extender)"], horizontal=True, key=f"pu_mass_mode_{purpose}")
        total_mass = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key=f"pu_mass_target_{purpose}")
        ext_frac = 0.0
        if use_ext and ext:
            ext_frac = st.slider("Extender mass fraction in polyol side", 0.0, 0.8, 0.2, step=0.01, key=f"pu_ext_frac_{purpose}")

        # Eq weights:
        EW_OH_poly = 56100.0 / max(oh_poly, 1e-12)
        EW_OH_ext  = 56100.0 / max(oh_ext,  1e-12) if (use_ext and ext) else np.inf
        EW_NCO     = 4200.0  / max(nco_percent, 1e-12)

        if mass_mode == "Polyol-side mass (polyol+extender)":
            m_poly_side = total_mass
            m_poly = m_poly_side*(1.0-ext_frac)
            m_ext  = m_poly_side*ext_frac if (use_ext and ext) else 0.0
            eq_OH = (m_poly/EW_OH_poly) + (m_ext/EW_OH_ext if (use_ext and ext) else 0.0)
            m_iso = idx * eq_OH * EW_NCO
            m_total = m_poly_side + m_iso
        else:
            # Solve for m_iso such that total mass = total_mass
            f = ext_frac if (use_ext and ext) else 0.0
            term = (1.0-f)/EW_OH_poly + (f/EW_OH_ext if (use_ext and ext) else 0.0)
            k = idx * EW_NCO * term
            m_iso = (k/(1.0+k)) * total_mass
            m_poly_side = total_mass - m_iso
            m_poly = m_poly_side*(1.0-f)
            m_ext  = m_poly_side*f
            m_total = total_mass

        st.markdown("#### Calculated masses")
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Total (g)", f"{m_total:.2f}")
        r2.metric("Isocyanate (g)", f"{m_iso:.2f}")
        r3.metric("Polyol (g)", f"{m_poly:.2f}")
        r4.metric("Extender (g)", f"{m_ext:.2f}" if (use_ext and ext) else "0.00")

        st.markdown("#### Compatibility indices (screening)")
        rows=[]
        for comp_id, tag in [(poly,"polyol")] + ([(ext,"extender")] if (use_ext and ext) else []):
            a = T.loc[comp_id]; b = T.loc[iso]
            ra = Ra(a,b,meta["dD"],meta["dP"],meta["dH"])
            da = delta_a(a,b,meta["da"])
            ds = delta_sigmaL(a,b,meta["sig"])
            pi = Pi(da, ds, da_max, sig_max)
            rows.append({"pair": f"{tag} × isocyanate", "A": comp_id, "B": iso, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
        df = pd.DataFrame(rows)
        if not df.empty:
            # Score % within these rows
            raM = float(df["Ra"].max()); piM = float(df["Π"].max()); daM = float(df["Δδa"].max())
            df["Score_%"] = [score_percent(r,p,d,raM,piM,daM) for r,p,d in zip(df["Ra"],df["Π"],df["Δδa"])]
            st.dataframe(df, use_container_width=True, hide_index=True)

elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.markdown("### Epoxy stoichiometry")
        st.caption("Uses **total batch mass target** OR resin-side mass. Inputs can be auto-filled from the equivalents library.")

        resins = resolve_list(spec["B"])
        hardeners = resolve_list(spec.get("subgroups", {}).get("Hardeners", spec["A"]))
        diluents = resolve_list(spec.get("subgroups", {}).get("Reactive diluents", []))

        c1,c2 = st.columns(2)
        with c1:
            resin = label_select("Resin", resins, key=f"ep_resin_{purpose}")
            selected_card(resin)
        with c2:
            hard = label_select("Hardener", hardeners, key=f"ep_hard_{purpose}")
            selected_card(hard)

        use_dil = st.checkbox("Add reactive diluent", value=False, key=f"ep_use_dil_{purpose}")
        dil = None
        if use_dil and diluents:
            dil = label_select("Reactive diluent", diluents, key=f"ep_dil_{purpose}")
            selected_card(dil)

        resin_eq = eq_lookup(resin)
        hard_eq = eq_lookup(hard)
        dil_eq = eq_lookup(dil) if dil else {}

        auto = st.checkbox("Auto-fill from equivalents library (recommended)", value=True, key=f"ep_auto_{purpose}")

        def _num(val, fallback):
            try:
                if pd.isna(val): return fallback
                return float(val)
            except Exception:
                return fallback

        EEW_res_default = _num(resin_eq.get("EEW_g_eq", np.nan), 185.0)
        AHEW_h_default  = _num(hard_eq.get("AHEW_g_eq", np.nan), 30.0)
        EEW_dil_default = _num(dil_eq.get("EEW_g_eq", np.nan), 150.0)

        st.markdown("#### Stoichiometry inputs")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            EEW_resin = st.number_input("Resin EEW (g/eq epoxide)", min_value=50.0, max_value=2000.0,
                                        value=EEW_res_default if auto else 185.0, step=1.0, key=f"ep_EEW_{purpose}")
        with cc2:
            AHEW_hard = st.number_input("Hardener AHEW (g/eq active H)", min_value=10.0, max_value=1000.0,
                                        value=AHEW_h_default if auto else 30.0, step=1.0, key=f"ep_AHEW_{purpose}")
        with cc3:
            ratio = st.number_input("Epoxy/Active-H equivalents ratio", min_value=0.5, max_value=2.0,
                                    value=1.00, step=0.01, key=f"ep_ratio_{purpose}")

        mass_mode = st.radio("Mass target mode", ["Total mass (includes hardener)", "Resin-side mass (resin+diluent)"], horizontal=True, key=f"ep_mass_mode_{purpose}")
        target_mass = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key=f"ep_mass_target_{purpose}")

        dil_frac = 0.0
        EEW_dil = None
        if use_dil and dil:
            dil_frac = st.slider("Diluente mass fraction in resin-side", 0.0, 0.8, 0.2, step=0.01, key=f"ep_dil_frac_{purpose}")
            EEW_dil = st.number_input("Diluente EEW (g/eq epoxide)", min_value=50.0, max_value=2000.0,
                                      value=EEW_dil_default if (auto) else 150.0, step=1.0, key=f"ep_EEW_dil_{purpose}")

        if not (use_dil and dil):
            dil_frac = 0.0
            EEW_dil = np.inf

        term = (1.0-dil_frac)/EEW_resin + (dil_frac/EEW_dil if (use_dil and dil) else 0.0)
        g = (AHEW_hard/max(ratio,1e-12)) * term

        if mass_mode == "Resin-side mass (resin+diluent)":
            m_res_side = target_mass
            m_hard = g * m_res_side
            m_total = m_res_side + m_hard
        else:
            # solve hardener mass given total mass
            m_hard = (g/(1.0+g)) * target_mass
            m_res_side = target_mass - m_hard
            m_total = target_mass

        m_dil = m_res_side * dil_frac if (use_dil and dil) else 0.0
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
            ra = Ra(a,b,meta["dD"],meta["dP"],meta["dH"])
            da = delta_a(a,b,meta["da"])
            ds = delta_sigmaL(a,b,meta["sig"])
            pi = Pi(da, ds, da_max, sig_max)
            rows.append({"pair": f"{tag} × resin", "A": comp_id, "B": resin, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
        df = pd.DataFrame(rows)
        if not df.empty:
            raM = float(df["Ra"].max()); piM = float(df["Π"].max()); daM = float(df["Δδa"].max())
            df["Score_%"] = [score_percent(r,p,d,raM,piM,daM) for r,p,d in zip(df["Ra"],df["Π"],df["Δδa"])]
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.markdown("### Ranking (best combinations)")
        st.caption("Ranks candidates by a **compatibility % score** using Ra + Δδa + Π (fixed equal weighting).")

        if purpose == "Epoxies":
            resins = resolve_list(spec["B"])
            resin = label_select("Resin (fixed)", resins, key=f"rk_ep_resin_{purpose}")
            selected_card(resin)
            include_dil = st.checkbox("Rank reactive diluents too", value=True, key=f"rk_ep_inc_dil_{purpose}")
            hardeners = [x for x in spec["subgroups"]["Hardeners"] if x in T.index]
            diluents = [x for x in spec["subgroups"]["Reactive diluents"] if x in T.index]
            cand = hardeners + (diluents if include_dil else [])
            df = build_candidates("Epoxies", cand, [resin], "candidates")
            if df.empty:
                st.warning("No candidates found in database.")
            else:
                topn = st.slider("Top N", 5, min(100, len(df)), min(20, len(df)), key=f"rk_ep_topn_{purpose}")
                st.dataframe(df.head(topn), use_container_width=True, hide_index=True)
                st.download_button("Download ranking CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name="ranking_epoxy.csv", mime="text/csv", key="dl_rk_ep")
        elif purpose == "Polyurethanes (PU)":
            isos = resolve_list(spec["B"])
            iso = label_select("Isocyanate (fixed)", isos, key=f"rk_pu_iso_{purpose}")
            selected_card(iso)
            polyols = [x for x in spec["subgroups"]["Polyols"] if x in T.index]
            extenders = [x for x in spec["subgroups"]["Chain extenders"] if x in T.index]
            cand = polyols + extenders
            df = build_candidates("PU", cand, [iso], "candidates")
            if df.empty:
                st.warning("No candidates found in database.")
            else:
                topn = st.slider("Top N", 5, min(150, len(df)), min(30, len(df)), key=f"rk_pu_topn_{purpose}")
                st.dataframe(df.head(topn), use_container_width=True, hide_index=True)
                st.download_button("Download ranking CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name="ranking_pu.csv", mime="text/csv", key="dl_rk_pu")
        else:
            st.info("Ranking is implemented for Epoxies and PU (most common formulation decisions).")

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
                M.loc[r,c] = Ra(ar, bc, meta["dD"], meta["dP"], meta["dH"])
            elif metric == "Δδa":
                M.loc[r,c] = delta_a(ar, bc, meta["da"])
            elif metric == "Π":
                da_ = delta_a(ar, bc, meta["da"])
                ds_ = delta_sigmaL(ar, bc, meta["sig"])
                M.loc[r,c] = Pi(da_, ds_, da_max, sig_max)
    return M

with tab_maps:
    st.subheader("Heatmaps")
    purpose_h = st.selectbox("Purpose", list(SETS.keys()), key="hm_purpose")
    metric = st.radio("Metric", ["Ra","Δδa","Π"], horizontal=True, key="hm_metric")

    rows = [x for x in SETS[purpose_h]["A"] if x in T.index]
    cols = [x for x in SETS[purpose_h]["B"] if x in T.index]

    max_rows = st.slider("Max rows shown", 10, 200, 60, key="hm_max_rows")
    max_cols = st.slider("Max columns shown", 10, 200, 40, key="hm_max_cols")

    M = build_matrix(rows[:max_rows], cols[:max_cols], metric)
    if M.empty or M.shape[0] < 2 or M.shape[1] < 2:
        st.warning("Not enough matching labels for a meaningful heatmap.")
    else:
        cbar = "Ra (MPa$^{1/2}$)" if metric=="Ra" else ("Δδa (MPa$^{1/2}$)" if metric=="Δδa" else "Π (normalized)")
        fig = heatmap(M, f"{purpose_h} — {metric} heatmap", cbar)
        st.pyplot(fig, clear_figure=True)
        st.download_button("Download PNG", fig_to_png_bytes(fig),
                           file_name=f"HM_{purpose_h}_{metric}.png", mime="image/png",
                           key=f"dl_hm_{purpose_h}_{metric}")

# -----------------------------
# FIGURES tab
# -----------------------------
def build_pairs(purpose):
    spec = SETS[purpose]
    subgroups = spec.get("subgroups", None)
    rows=[]
    B_ok = [b for b in spec["B"] if b in T.index]
    if subgroups:
        for gname, A_list in subgroups.items():
            A_ok = [a for a in A_list if a in T.index]
            for a_id in A_ok:
                a = T.loc[a_id]
                for b_id in B_ok:
                    b = T.loc[b_id]
                    ra = Ra(a,b,meta["dD"],meta["dP"],meta["dH"])
                    da = delta_a(a,b,meta["da"])
                    ds = delta_sigmaL(a,b,meta["sig"])
                    pi = Pi(da, ds, da_max, sig_max)
                    rows.append({"group": gname, "A": a_id, "B": b_id, "Ra": ra, "delta_a": da, "Pi": pi})
    else:
        A_ok = [a for a in spec["A"] if a in T.index]
        for a_id in A_ok:
            a = T.loc[a_id]
            for b_id in B_ok:
                b = T.loc[b_id]
                ra = Ra(a,b,meta["dD"],meta["dP"],meta["dH"])
                da = delta_a(a,b,meta["da"])
                ds = delta_sigmaL(a,b,meta["sig"])
                pi = Pi(da, ds, da_max, sig_max)
                rows.append({"group": purpose, "A": a_id, "B": b_id, "Ra": ra, "delta_a": da, "Pi": pi})
    return pd.DataFrame(rows)

with tab_figs:
    st.subheader("FIGURES: Ra vs Δδa and Ra vs Π")
    purpose_p = st.selectbox("Purpose", list(SETS.keys()), key="fig_purpose")
    df = build_pairs(purpose_p)
    if df.empty:
        st.warning("No valid pairs for this purpose.")
    else:
        fig1 = scatter(df, "Ra", "delta_a", "group",
                       title=f"{purpose_p} — Ra vs Δδa",
                       xlabel="Ra (MPa$^{1/2}$)", ylabel="Δδa (MPa$^{1/2}$)")
        st.pyplot(fig1, clear_figure=True)
        st.download_button("Download PNG (Ra vs Δδa)", fig_to_png_bytes(fig1),
                           file_name=f"FIG_Ra_vs_deltaa_{purpose_p}.png", mime="image/png",
                           key=f"dl_fig_da_{purpose_p}")

        fig2 = scatter(df, "Ra", "Pi", "group",
                       title=f"{purpose_p} — Ra vs Π",
                       xlabel="Ra (MPa$^{1/2}$)", ylabel="Π (normalized)")
        st.pyplot(fig2, clear_figure=True)
        st.download_button("Download PNG (Ra vs Π)", fig_to_png_bytes(fig2),
                           file_name=f"FIG_Ra_vs_Pi_{purpose_p}.png", mime="image/png",
                           key=f"dl_fig_pi_{purpose_p}")
