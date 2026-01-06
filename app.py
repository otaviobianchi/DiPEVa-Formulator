
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# DiPEVa Formulator — app.py
# (internal DB; no upload; formulation-oriented UI)
# ============================================================

st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic / research use only — screening tool, not a standalone decision device.")

DB_FILE = Path(__file__).resolve().parent / "Database_final.xlsx"


@st.cache_data(show_spinner=False)
def load_db(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

df_raw = load_db(DB_FILE)


def _find_col(cands):
    for c in cands:
        if c in df_raw.columns:
            return c
    for c in df_raw.columns:
        cl = str(c).lower()
        for k in cands:
            if str(k).lower() in cl:
                return c
    return None


COL = {}
COL["name"]   = _find_col(["Molecule", "Molecule name", "Name"])
COL["abbr"]   = _find_col(["Abbrev", "Abbreviation", "abbr"])
COL["cas"]    = _find_col(["CAS", "CASRN"])
COL["smiles"] = _find_col(["SMILES/BigSMILES", "SMILES", "BigSMILES"])
COL["dD"]     = _find_col(["δD (MPa1/2)", "deltaD", "dD", "δD"])
COL["dP"]     = _find_col(["δP (MPa1/2)", "deltaP", "dP", "δP"])
COL["dH"]     = _find_col(["δH (MPa1/2)", "deltaH", "dH", "δH"])
COL["sig"]    = _find_col(["σL (mN·m-1)", "sigmaL", "σL", "σ"])
COL["da"]     = _find_col(["δa (MPa1/2)", "delta a", "δa"])

missing = [k for k in ["abbr", "dD", "dP", "dH"] if COL.get(k) is None]
if missing:
    st.error(
        f"Database is missing required columns: {missing}\n"
        f"Found columns: {list(df_raw.columns)}"
    )
    st.stop()


# -----------------------------
# Normalize / build internal table (DO NOT collapse duplicates)
# -----------------------------
T = df_raw.copy()

T["__abbr__"] = (
    T[COL["abbr"]]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.upper()
)
T = T[T["__abbr__"].ne("")].copy()

cas_series = (
    T[COL["cas"]].astype(str).str.strip()
    if COL.get("cas") and COL["cas"] in T.columns
    else pd.Series([""] * len(T), index=T.index)
)

T["__id__"] = T["__abbr__"]
dup = T["__id__"].duplicated(keep=False)
if dup.any():
    for i in T.index[dup]:
        cas = cas_series.loc[i]
        if cas and cas.lower() != "nan":
            T.at[i, "__id__"] = f'{T.at[i, "__abbr__"]} [{cas}]'
    seen = {}
    new_ids = []
    for s in T["__id__"].tolist():
        if s not in seen:
            seen[s] = 1
            new_ids.append(s)
        else:
            seen[s] += 1
            new_ids.append(f"{s} #{seen[s]}")
    T["__id__"] = new_ids

for k in ["dD", "dP", "dH", "sig", "da"]:
    c = COL.get(k)
    if c and c in T.columns:
        T[c] = pd.to_numeric(T[c], errors="coerce")

T = T.set_index("__id__", drop=False)


def _nm(row) -> str:
    if COL.get("name") and COL["name"] in row.index:
        v = row[COL["name"]]
        return "" if pd.isna(v) else str(v)
    return ""


def label(_id: str) -> str:
    row = T.loc[_id]
    ab = str(row["__abbr__"]).strip()
    nm = _nm(row).strip()
    if nm == "" or nm.lower() in ("nan", "none") or nm.upper() == ab.upper():
        return ab
    return f"{ab} — {nm}"


def classify_row(row: pd.Series) -> str:
    """Heuristic family classifier.
    Note: 'extender' is intended for *low-molecular-weight* multi-functional alcohols
    (diols/triols) used as chain extenders/crosslinkers in PU formulations.
    """
    ab = str(row["__abbr__"]).strip().lower()
    nm = str(row.get("__name__", "")).strip().lower()

    # -----------------
    # ISOCYANATES
    # -----------------
    if (
        ("isocyan" in nm)
        or ("nco" in nm)
        or re.search(r"\b(mdi|tdi|hdi|ipdi|pmdi|h12mdi)\b", ab)
        or re.search(r"\b(mdi|tdi|hdi|ipdi|pmdi|h12mdi)\b", nm)
    ):
        return "isocyanate"

    # -----------------
    # PU EXTENDERS / CROSSLINKERS (multi-functional alcohols, non-polymeric)
    # -----------------
    # Typical diols
    diol_ab_pat = r"(?:\b(eg|meg|pg|mpg|npg|deg|teg|dpg|bdg|bdo|hdo|pdo|pe?do)\b|\b\d[,\.]\d\s*-?\s*(bdo|pdo|hdo|pe?do)\b)"
    # Names / keywords for low-MW diols
    diol_nm_keys = [
        "ethylene glycol", "etileno glicol", "ethane-1,2-diol",
        "propylene glycol", "propanediol", "propane-1,2-diol", "propane-1,3-diol",
        "butanediol", "butane-1,4-diol", "butane-1,3-diol", "butane-2,3-diol",
        "pentanediol", "hexanediol", "1,6-hexanediol", "1,5-pentanediol",
        "neopentyl glycol", "npg",
        "diethylene glycol", "triethylene glycol", "tetraethylene glycol",
        "cyclohexanedimethanol", "chdm",
    ]
    # Triols / crosslinkers (keep separate class so UI can optionally include)
    triol_ab_pat = r"\b(tmp|gly|glycerol|tea)\b"
    triol_nm_keys = [
        "glycerol", "glicerol",
        "trimethylolpropane", "tmp",
        "triethanolamine", "tea",
        "triol",
    ]

    # extenders: diols (and optionally triols)
    if (
        ("diol" in nm and "poly" not in nm and "polyol" not in nm)
        or ("glycol" in nm and "poly" not in nm)
        or re.search(diol_ab_pat, ab)
        or any(k in nm for k in diol_nm_keys)
        or re.search(r"\b(1,6-hdo|1,6hdo|1,5-pedo|1,5pedo|1,4-bdo|1,4bdo|1,3-bdo|1,3bdo|2,3-bdo|2,3bdo)\b", nm)
    ):
        return "extender"

    if (
        re.search(triol_ab_pat, ab)
        or any(k in nm for k in triol_nm_keys)
        or ("trimethylol" in nm)
    ):
        return "crosslinker"

    # -----------------
    # POLYOLS (polymeric or higher functionality, incl. polyethers/polyesters)
    # -----------------
    if (
        ("polyol" in nm)
        or re.search(r"\b(peg\d+|ppg\d+|ptmeg\d+|pcl\d+|pcdl\d+|pcldiol|pcdl)\b", ab)
        or ("polyethylene glycol" in nm)
        or ("polypropylene glycol" in nm)
        or ("polytetramethylene" in nm)
        or ("caprolactone" in nm and "diol" in nm)
        or ("polyether" in nm)
    ):
        return "polyol"

    # -----------------
    # POLYESTER ACIDS / ANHYDRIDES
    # -----------------
    if (
        ("anhydride" in nm)
        or ("acid" in nm)
        or re.search(r"\b(mah|hhpa|mhhpa)\b", ab)
        or re.search(r"\b(anhydride|acid)\b", nm)
    ):
        return "acid_anhydride"

    # -----------------
    # EPOXY SYSTEMS
    # -----------------
    if (
        ("epoxy" in nm)
        or re.search(r"\b(dgeba|bdge|dgbf|tgddm|novolac)\b", ab)
        or ("diglycidyl" in nm)
        or ("glycidyl" in nm and "amine" not in nm)
    ):
        return "epoxy_resin"

    if (
        ("amine" in nm)
        or ("diamine" in nm)
        or re.search(r"\b(deta|teta|ddm|dds|dicy|ipda|eda)\b", ab)
    ):
        return "epoxy_hardener"

    # Reactive diluents (epoxy side)
    if (
        ("reactive diluent" in nm)
        or ("glycidyl ether" in nm)
        or ("benzyl alcohol" in nm)
        or ("glyme" in nm)
        or ("ether" in nm and "poly" not in nm)
    ):
        return "reactive_diluent"

    # -----------------
    # VINYLICS
    # -----------------
    if (
        ("styrene" in nm)
        or ("acrylonitrile" in nm)
        or ("methacrylate" in nm)
        or ("vinyl acetate" in nm)
        or re.search(r"\b(an|mma|st|fn|α-ms|a-ms)\b", ab)
    ):
        return "vinyl_monomer"

    if (
        ("phthalate" in nm)
        or ("adipate" in nm)
        or ("benzoate" in nm)
        or ("phosphate" in nm)
        or ("solvent" in nm)
        or ("plasticizer" in nm)
        or ("alcohol" in nm and "poly" not in nm)
    ):
        return "solvent_plasticizer"

    # -----------------
    # SILANES
    # -----------------
    if (
        ("silane" in nm)
        or re.search(r"\b(teos|vtms|aptes|mptes|vinylsilane|vinyl silane)\b", ab)
        or ("silane" in nm)
    ):
        return "silane"

    return "other"




T["__class__"] = [classify_row(T.loc[i]) for i in T.index]


def list_by_class(cls: str) -> list[str]:
    return [i for i in T.index if T.loc[i, "__class__"] == cls]


# -----------------------------
# Metrics (article-consistent)
# -----------------------------
def Ra(rowA: pd.Series, rowB: pd.Series) -> float:
    dD, dP, dH = COL["dD"], COL["dP"], COL["dH"]
    return float(np.sqrt(4.0 * (rowA[dD] - rowB[dD]) ** 2 + (rowA[dP] - rowB[dP]) ** 2 + (rowA[dH] - rowB[dH]) ** 2))

def delta_a(rowA: pd.Series, rowB: pd.Series) -> float:
    c = COL.get("da")
    if c and (c in rowA.index) and (c in rowB.index) and np.isfinite(rowA[c]) and np.isfinite(rowB[c]):
        return float(abs(rowA[c] - rowB[c]))
    dP, dH = COL["dP"], COL["dH"]
    return float(abs(np.sqrt(rowA[dP] ** 2 + rowA[dH] ** 2) - np.sqrt(rowB[dP] ** 2 + rowB[dH] ** 2)))

def delta_sigmaL(rowA: pd.Series, rowB: pd.Series) -> float:
    c = COL.get("sig")
    if not c or c not in rowA.index or c not in rowB.index:
        return float("nan")
    if pd.isna(rowA[c]) or pd.isna(rowB[c]):
        return float("nan")
    return float(abs(rowA[c] - rowB[c]))

def Pi(da: float, ds: float, da_max: float, ds_max: float) -> float:
    nda = da / (da_max + 1e-12)
    if np.isfinite(ds):
        nds = ds / (ds_max + 1e-12)
        return float(nda + nds)
    return float(nda)

def score_percent(ra: float, da: float, pi: float, ra_max: float, da_max: float, pi_max: float) -> float:
    n_ra = ra / (ra_max + 1e-12)
    n_da = da / (da_max + 1e-12)
    n_pi = pi / (pi_max + 1e-12)
    s = 100.0 * (1.0 - (n_ra + n_da + n_pi) / 3.0)
    return float(np.clip(s, 0, 100))


_da_max = float(np.nanmax(T[COL["da"]])) if COL.get("da") and COL["da"] in T.columns and np.isfinite(np.nanmax(T[COL["da"]])) else 1.0
_ds_max = float(np.nanmax(T[COL["sig"]])) if COL.get("sig") and COL["sig"] in T.columns and np.isfinite(np.nanmax(T[COL["sig"]])) else 1.0


# -----------------------------
# Library of equivalents (editable)
# -----------------------------
DEFAULT_EQUIV = [
    {"Abbrev": "2,4-TDI", "Role": "Isocyanate", "NCO_%": 48.3, "OH_mgKOH_g": np.nan, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "HDI",     "Role": "Isocyanate", "NCO_%": 50.0, "OH_mgKOH_g": np.nan, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "IPDI",    "Role": "Isocyanate", "NCO_%": 37.8, "OH_mgKOH_g": np.nan, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "H12MDI",  "Role": "Isocyanate", "NCO_%": 32.0, "OH_mgKOH_g": np.nan, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "1,4-BDO", "Role": "Extender",   "NCO_%": np.nan, "OH_mgKOH_g": 1246.0, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "1,3-BDO", "Role": "Extender",   "NCO_%": np.nan, "OH_mgKOH_g": 1246.0, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "EG",      "Role": "Extender",   "NCO_%": np.nan, "OH_mgKOH_g": 1806.0, "EEW_g_eq": np.nan, "AHEW_g_eq": np.nan},
    {"Abbrev": "DGEBA",   "Role": "Epoxy resin","NCO_%": np.nan, "OH_mgKOH_g": np.nan, "EEW_g_eq": 185.0, "AHEW_g_eq": np.nan},
    {"Abbrev": "BDGE",    "Role": "Epoxy resin","NCO_%": np.nan, "OH_mgKOH_g": np.nan, "EEW_g_eq": 185.0, "AHEW_g_eq": np.nan},
    {"Abbrev": "DDM",     "Role": "Hardener",   "NCO_%": np.nan, "OH_mgKOH_g": np.nan, "EEW_g_eq": np.nan, "AHEW_g_eq": 30.0},
]

if "equiv_lib" not in st.session_state:
    st.session_state["equiv_lib"] = pd.DataFrame(DEFAULT_EQUIV)

def equiv_lookup(abbrev: str, field: str):
    lib = st.session_state["equiv_lib"]
    if lib.empty:
        return None
    mask = lib["Abbrev"].astype(str).str.strip().str.upper().eq(str(abbrev).strip().upper())
    if not mask.any():
        return None
    v = lib.loc[mask, field].iloc[0]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None

def base_abbrev(_id: str) -> str:
    return str(T.loc[_id, "__abbr__"]).strip()

def choose(title: str, options: list[str], key: str):
    if not options:
        st.warning(f"No options available for: {title}")
        return None
    options = list(dict.fromkeys(options))
    labels = [label(i) for i in options]
    seen = {}
    final = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            final.append(lab)
        else:
            seen[lab] += 1
            final.append(f"{lab} ({seen[lab]})")
    mapping = dict(zip(final, options))
    choice = st.selectbox(title, final, key=key)
    return mapping[choice]

def show_props(_id: str):
    row = T.loc[_id]
    show = {}
    if COL.get("name"):   show["Name"] = _nm(row)
    if COL.get("cas") and COL["cas"] in row.index: show["CAS"] = "" if pd.isna(row[COL["cas"]]) else str(row[COL["cas"]])
    if COL.get("smiles") and COL["smiles"] in row.index: show["SMILES/BigSMILES"] = "" if pd.isna(row[COL["smiles"]]) else str(row[COL["smiles"]])
    for k in ["dD","dP","dH","da","sig"]:
        c = COL.get(k)
        if c and c in row.index and np.isfinite(row[c]):
            show[c] = f"{float(row[c]):.2f}"
    st.dataframe(pd.DataFrame.from_dict(show, orient="index", columns=["Value"]), use_container_width=True)

# -----------------------------
# Plots
# -----------------------------
def plot_heatmap(mat: pd.DataFrame, title: str, cbar_label: str):
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * mat.shape[1]), max(4, 0.35 * mat.shape[0])))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in sorted(df[hue].unique()):
        sub = df[df[hue] == g]
        ax.scatter(sub[x], sub[y], label=str(g), alpha=0.75)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

# -----------------------------
# Tabs
# -----------------------------
tab_form, tab_maps, tab_figs = st.tabs(["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"])

with tab_form:
    st.subheader("Formulator")

    with st.expander("📚 Library of equivalents (editable)", expanded=False):
        st.write("Matching is done by **Abbrev** (case-insensitive).")
        st.session_state["equiv_lib"] = st.data_editor(
            st.session_state["equiv_lib"],
            num_rows="dynamic",
            use_container_width=True,
            key="equiv_editor",
        )

    mode = st.selectbox(
        "Mode",
        ["Pair screening", "PU stoichiometry (NCO/OH)", "Epoxy stoichiometry (EEW/AHEW)", "Ranking (best combinations)"],
        key="mode_select",
    )

    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender") + list_by_class("crosslinker")
    ep_resins = list_by_class("epoxy_resin")
    ep_hards = list_by_class("epoxy_hardener")
    ep_dils  = list_by_class("reactive_diluent")

    if mode == "Pair screening":
        c1, c2 = st.columns(2)
        with c1:
            A = choose("Component A", list(T.index), key="pair_A")
        with c2:
            B = choose("Component B", list(T.index), key="pair_B")

        if A and B:
            a = T.loc[A]; b = T.loc[B]
            ra = Ra(a,b); da = delta_a(a,b); ds = delta_sigmaL(a,b)
            pi = Pi(da, ds, _da_max, _ds_max)
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Ra (MPa$^{1/2}$)", f"{ra:.2f}")
            r2.metric("Δδa (MPa$^{1/2}$)", f"{da:.2f}")
            r3.metric("ΔσL (mN/m)", f"{ds:.2f}" if np.isfinite(ds) else "n/a")
            r4.metric("Π (dimensionless)", f"{pi:.3f}")
            c1,c2 = st.columns(2)
            with c1:
                st.write("**A**"); show_props(A)
            with c2:
                st.write("**B**"); show_props(B)

    elif mode == "PU stoichiometry (NCO/OH)":
        st.caption("Auto-fills %NCO and OH numbers from the equivalents library (you can override).")
        c1,c2,c3 = st.columns(3)
        with c1:
            iso = choose("Isocyanate", isos, key="pu_iso")
        with c2:
            pol = choose("Polyol", polyols, key="pu_pol")
        with c3:
            use_ext = st.checkbox("Include chain extender", value=True, key="pu_use_ext")
        ext = choose("Chain extender", extenders, key="pu_ext") if use_ext else None

        iso_ab = base_abbrev(iso) if iso else ""
        pol_ab = base_abbrev(pol) if pol else ""
        ext_ab = base_abbrev(ext) if ext else ""

        nco_auto = equiv_lookup(iso_ab, "NCO_%") if iso else None
        oh_pol_auto = equiv_lookup(pol_ab, "OH_mgKOH_g") if pol else None
        oh_ext_auto = equiv_lookup(ext_ab, "OH_mgKOH_g") if (use_ext and ext) else None

        lock = st.checkbox("Lock inputs to library values", value=True, key="pu_lock")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            nco = st.number_input("%NCO (wt%)", 0.0, 60.0, value=float(nco_auto) if (lock and nco_auto is not None) else 33.0, step=0.1)
        with cc2:
            oh_pol = st.number_input("OH number polyol (mgKOH/g)", 0.0, 3000.0, value=float(oh_pol_auto) if (lock and oh_pol_auto is not None) else 56.0, step=1.0)
        with cc3:
            oh_ext = st.number_input("OH number extender (mgKOH/g)", 0.0, 3000.0, value=float(oh_ext_auto) if (lock and oh_ext_auto is not None) else (1200.0 if use_ext else 0.0), step=1.0)

        idx = st.number_input("NCO/OH index", 0.5, 2.5, 1.05, 0.01)
        mass_mode = st.radio("Mass target mode", ["Total mass (includes isocyanate)", "Polyol-side mass (polyol+extender)"], horizontal=True)
        target = st.number_input("Target mass (g)", 1.0, value=100.0, step=1.0)
        ext_frac = st.slider("Extender mass fraction in polyol side", 0.0, 0.8, 0.2, 0.01) if (use_ext and ext) else 0.0

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

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Isocyanate (g)", f"{m_iso:.2f}")
            r3.metric("Polyol (g)", f"{m_pol:.2f}")
            r4.metric("Extender (g)", f"{m_ext:.2f}" if (use_ext and ext) else "0.00")

    elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.caption("Auto-fills EEW/AHEW from the equivalents library (you can override).")
        c1,c2,c3 = st.columns(3)
        with c1:
            resin = choose("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = choose("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")
        dil = choose("Reactive diluent", ep_dils, key="ep_dil") if use_dil else None

        res_ab = base_abbrev(resin) if resin else ""
        hard_ab = base_abbrev(hard) if hard else ""
        dil_ab = base_abbrev(dil) if dil else ""

        EEW_auto = equiv_lookup(res_ab, "EEW_g_eq") if resin else None
        AHEW_auto = equiv_lookup(hard_ab, "AHEW_g_eq") if hard else None
        EEW_dil_auto = equiv_lookup(dil_ab, "EEW_g_eq") if (use_dil and dil) else None

        lock = st.checkbox("Lock inputs to library values", value=True, key="ep_lock")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            EEW_res = st.number_input("Resin EEW (g/eq)", 50.0, 3000.0, value=float(EEW_auto) if (lock and EEW_auto is not None) else 185.0, step=1.0)
        with cc2:
            AHEW = st.number_input("Hardener AHEW (g/eq)", 10.0, 2000.0, value=float(AHEW_auto) if (lock and AHEW_auto is not None) else 30.0, step=1.0)
        with cc3:
            ratio = st.number_input("Epoxy/Active-H eq ratio", 0.5, 2.0, 1.0, 0.01)

        mass_mode = st.radio("Mass target mode", ["Total mass (includes hardener)", "Resin-side mass (resin+diluent)"], horizontal=True)
        target = st.number_input("Target mass (g)", 1.0, value=100.0, step=1.0)

        dil_frac = 0.0
        EEW_dil = np.inf
        if use_dil and dil:
            dil_frac = st.slider("Diluent mass fraction (in resin-side)", 0.0, 0.8, 0.2, 0.01)
            EEW_dil = st.number_input("Diluent EEW (g/eq)", 50.0, 3000.0, value=float(EEW_dil_auto) if (lock and EEW_dil_auto is not None) else 150.0, step=1.0)

        term = (1.0-dil_frac)/EEW_res + (dil_frac/EEW_dil if (use_dil and dil) else 0.0)
        g = (AHEW/max(ratio,1e-12)) * term

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

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Hardener (g)", f"{m_hard:.2f}")
            r3.metric("Resin (g)", f"{m_res:.2f}")
            r4.metric("Reactive diluent (g)", f"{m_dil:.2f}" if (use_dil and dil) else "0.00")

    else:
        st.info("Ranking tab kept in v14 improved; use full version if needed.")

with tab_maps:
    st.subheader("Heatmaps")
    st.write("Use the v14 improved build for full heatmaps/figures module.")
with tab_figs:
    st.subheader("FIGURES")
    st.write("Use the v14 improved build for full heatmaps/figures module.")
