import re
from pathlib import Path
import math

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
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

df_raw = load_db(DB_FILE)

# Detect key columns (robust to header variants)
COL = {}
def _find_col(cands):
    for c in cands:
        if c in df_raw.columns:
            return c
    for c in df_raw.columns:
        for k in cands:
            if k.lower() in str(c).lower():
                return c
    return None

COL["name"]   = _find_col(["Molecule", "Molecule name", "Name"])
COL["abbr"]   = _find_col(["Abbrev", "Abbreviation", "abbr"])
COL["cas"]    = _find_col(["CAS", "CASRN"])
COL["smiles"] = _find_col(["SMILES/BigSMILES", "SMILES", "BigSMILES"])
COL["dD"]     = _find_col(["δD (MPa1/2)", "deltaD", "dD", "δD"])
COL["dP"]     = _find_col(["δP (MPa1/2)", "deltaP", "dP", "δP"])
COL["dH"]     = _find_col(["δH (MPa1/2)", "deltaH", "dH", "δH"])
COL["sig"]    = _find_col(["σL (mN·m-1)", "sigmaL", "σL", "surface tension"])
COL["da"]     = _find_col(["δa (MPa1/2)", "delta a", "δa", "delta_a"])

missing = [k for k in ["abbr", "dD", "dP", "dH"] if COL.get(k) is None]
if missing:
    st.error(
        "Database is missing required columns: "
        f"{missing}\nFound columns: {list(df_raw.columns)}"
    )
    st.stop()

# Build clean table indexed by Abbrev (as chemistry key)
T = df_raw.copy()
T["__abbr__"] = (
    T[COL["abbr"]].astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
T = T[T["__abbr__"].ne("")].copy()
T["__abbr__"] = T["__abbr__"].str.upper()

# keep FIRST occurrence per Abbrev
T = T.drop_duplicates(subset="__abbr__", keep="first").set_index("__abbr__", drop=True)

# numeric coercion
for k in ["dD", "dP", "dH", "sig", "da"]:
    c = COL.get(k)
    if c and c in T.columns:
        T[c] = pd.to_numeric(T[c], errors="coerce")

# =========================
# DEFAULT EQUIVALENTS LIBRARY
# =========================
DEFAULT_EQUIV = [
    # Isocyanates (NCO %)
    {"Abbrev": "2,4-TDI", "Role": "Isocyanate", "NCO_%": 48.3, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "2,6-TDI", "Role": "Isocyanate", "NCO_%": 48.3, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TDI 80/20", "Role": "Isocyanate", "NCO_%": 48.3, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "HDI", "Role": "Isocyanate", "NCO_%": 50.0, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "IPDI", "Role": "Isocyanate", "NCO_%": 37.8, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "H12MDI", "Role": "Isocyanate", "NCO_%": 32.0, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "POLYMERIC MDI", "Role": "Isocyanate", "NCO_%": 31.5, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Difunctional alcohols (Extenders) - include common glycols/diols
    {"Abbrev": "EG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1806.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "DEG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1058.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TEG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 748.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,3-BDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1246.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,4-BDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1246.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "2,3-BDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1246.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,6-HDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 967.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "MPG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1475.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "NPG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1197.0, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Triols / Crosslinkers (block separate)
    {"Abbrev": "GLY", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1828.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TMP", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1253.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "SOR", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1217.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "PER", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1760.0, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Epoxy resins / hardeners (EEW/AHEW)
    {"Abbrev": "DGEBA", "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 185.0, "AHEW_g_eq": None},
    {"Abbrev": "BDGE",  "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 185.0, "AHEW_g_eq": None},
    {"Abbrev": "DDM",   "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 30.0},
    {"Abbrev": "DDS",   "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 62.0},
    {"Abbrev": "DICY",  "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 42.0},
]

def _norm_abbr(s: str) -> str:
    return str(s).strip().upper()

if "equiv_lib" not in st.session_state:
    st.session_state["equiv_lib"] = pd.DataFrame(DEFAULT_EQUIV)

def get_equiv_df() -> pd.DataFrame:
    df = st.session_state["equiv_lib"].copy()
    if "Abbrev" in df.columns:
        df["Abbrev"] = df["Abbrev"].map(_norm_abbr)
    if "Role" in df.columns:
        df["Role"] = df["Role"].astype(str).str.strip()
    return df

def equiv_lookup(abbr: str) -> dict:
    ab = _norm_abbr(abbr)
    df = get_equiv_df()
    if "Abbrev" not in df.columns:
        return {}
    hit = df[df["Abbrev"] == ab]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()

# =========================
# METRICS (article-consistent)
# =========================
def Ra(a: pd.Series, b: pd.Series) -> float:
    dD, dP, dH = COL["dD"], COL["dP"], COL["dH"]
    return float(np.sqrt(4.0*(a[dD]-b[dD])**2 + (a[dP]-b[dP])**2 + (a[dH]-b[dH])**2))

def delta_a(a: pd.Series, b: pd.Series) -> float:
    c = COL.get("da")
    if c and (c in a.index) and (c in b.index) and (not pd.isna(a[c])) and (not pd.isna(b[c])):
        return float(abs(a[c] - b[c]))
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
    # Article: no wa/ws weights: Π = Δδa/δa,max + ΔσL/σL,max
    nda = da/(da_max + 1e-12)
    if np.isfinite(ds):
        nds = ds/(ds_max + 1e-12)
        return float(nda + nds)
    return float(nda)

def score_percent(ra: float, da: float, pi: float, ra_max: float, da_max: float, pi_max: float) -> float:
    # Higher is better (0..100): 1 - mean(normalized penalties)
    n_ra = ra/(ra_max + 1e-12)
    n_da = da/(da_max + 1e-12)
    n_pi = pi/(pi_max + 1e-12)
    s = 100.0*(1.0 - (n_ra + n_da + n_pi)/3.0)
    return float(np.clip(s, 0, 100))

# =========================
# LABELS / CLASSIFICATION
# =========================
def _name_of(abbr: str) -> str:
    if COL["name"] and COL["name"] in T.columns:
        v = T.loc[abbr, COL["name"]]
        return str(v) if not pd.isna(v) else abbr
    return abbr

def _smiles_of(abbr: str) -> str:
    if COL.get("smiles") and COL["smiles"] in T.columns:
        v = T.loc[abbr, COL["smiles"]]
        if pd.isna(v):
            return ""
        return str(v).strip()
    return ""

def label(abbr: str) -> str:
    nm = _name_of(abbr)
    base = str(abbr).strip()
    nm_s = str(nm).strip()
    if nm_s.lower() in ("", "nan", "none") or nm_s.upper() == base.upper():
        return base
    return f"{base} — {nm_s}"

def classify_row(abbr: str, row: pd.Series) -> str:
    """Heuristic classification for UI grouping (heatmaps/formulator).
    The editable equivalents library can override roles via explicit filtering later.
    """
    name = _name_of(abbr).lower()
    a = str(abbr).lower()

    # PU: isocyanates
    if (
        "isocyanate" in name
        or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", a)
        or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", name)
        or "nco" in name
    ):
        return "isocyanate"

    # Triols / multifunctional alcohols
    if (
        "triol" in name
        or "glycer" in name
        or "sorbit" in name
        or "pentaerythrit" in name
        or re.search(r"\btmp\b|\bgly\b|\bper\b|\bsor\b", a)
    ):
        return "crosslinker"

    # Extenders (diols / glycols)
    if (
        "butanediol" in name
        or "hexanediol" in name
        or "propanediol" in name
        or re.search(r"\beg\b|\bdeg\b|\bteg\b", a)
        or re.search(r"\b1,?3-?bdo\b|\b1,?4-?bdo\b|\b2,?3-?bdo\b|\b1,?6-?hdo\b|\bhexd?anediol\b", a)
        or re.search(r"\bmpg\b|\bpg\b|\bdpg\b|\bnpg\b", a)
    ):
        return "extender"

    # Polyols (oligomeric / polymeric diols)
    if (
        "polyol" in name
        or "polyether" in name
        or "polyester" in name
        or "polycarbonate" in name
        or "diol" in name
        or re.search(r"\bpeg\d+|\bppg\d+|\bptmeg\d*|\bpcl\d+|\bpcdl\d+|\bpbd\d*", a)
    ):
        return "polyol"

    # Polyester acids / anhydrides (include diacids)
    acid_abbr = {"AA","SA","PA","GA","MA","LA","OA","FA","IPA","TPA","SUA","SEA","MAH","HHPA","MHHPA"}
    if (
        "acid" in name
        or "diacid" in name
        or "dicarbox" in name
        or "anhydride" in name
        or abbr.upper() in acid_abbr
        or re.search(r"\b(phthalic|isophthalic|terephthalic|succinic|sebacic|adipic|glutaric|malonic|maleic|fumaric)\b", name)
        or re.search(r"\bmah\b|\bhhpa\b|\bmhhpa\b", a)
    ):
        return "acid_anhydride"

    # Epoxy family
    if ("epoxy" in name) or ("oxirane" in name) or re.search(r"\b(epoxide|glycidyl|oxiran)\b", name):
        # Resin vs diluent (best effort)
        if ("bisphenol" in name) or ("novolac" in name) or ("resin" in name) or re.search(r"\bdgeba\b|\bdgebf\b", a):
            return "epoxy_resin"
        if ("glycidyl" in name) or ("diluent" in name):
            return "reactive_diluent"
        return "epoxy_resin"

    if (
        "amine" in name
        or "diamin" in name
        or "polyamine" in name
        or "hardener" in name
        or "curing" in name
        or re.search(r"\bdeta\b|\bteta\b|\bddm\b|\bdds\b|\bdicy\b|\bipda\b|\bjeffamine\b", a)
    ):
        return "epoxy_hardener"

    # Vinyl family
    if (
        "styrene" in name
        or "acrylonitrile" in name
        or "methacrylate" in name
        or "acrylate" in name
        or "vinyl acetate" in name
        or "alpha-methylstyrene" in name
        or re.search(r"\bst\b|\ban\b|\bmma\b|\bva\b|\bvac\b|\bfn\b|\bα-ms\b|\ba-ms\b", a)
    ):
        return "vinyl_monomer"

    # Solvents / plasticizers
    if (
        "alcohol" in name
        or "solvent" in name
        or "plasticizer" in name
        or "phthalate" in name
        or "phosphate" in name
        or "adipate" in name
        or "benzoate" in name
        or re.search(r"\bethanol\b|\bmethanol\b|\bpropanol\b|\bbutanol\b", name)
    ):
        return "solvent_plasticizer"

    # Silanes
    if ("silane" in name) or re.search(r"\bteos\b|\bvtms\b|\baptes\b|\bmptes\b", a):
        return "silane"

    return "other"

T["__class__"] = [classify_row(i, T.loc[i]) for i in T.index]

def list_by_class(cls: str) -> list[str]:
    return [i for i in T.index if T.loc[i, "__class__"] == cls]

def select_abbr(title: str, options: list[str], key: str):
    if not options:
        st.warning(f"No options found for: {title}")
        return None
    options = list(dict.fromkeys([o for o in options if o in T.index]))
    labels = [label(o) for o in options]
    # disambiguate identical labels
    seen = {}
    final_labels = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            final_labels.append(lab)
        else:
            seen[lab] += 1
            final_labels.append(f"{lab} (dup {seen[lab]})")
    mapping = dict(zip(final_labels, options))
    choice = st.selectbox(title, final_labels, key=key)
    return mapping[choice]

def show_props(abbr: str):
    if not abbr:
        return
    row = T.loc[abbr]
    cols_show = []
    for k in ["name", "cas", "smiles", "dD", "dP", "dH", "da", "sig"]:
        c = COL.get(k)
        if c and c in row.index:
            cols_show.append(c)

    out = {}
    for k in cols_show:
        v = row[k]
        if isinstance(v, (float, int, np.floating, np.integer)) and not pd.isna(v):
            if k in {COL["dD"], COL["dP"], COL["dH"], COL.get("da"), COL.get("sig")}:
                out[k] = f"{float(v):.2f}"
            else:
                out[k] = str(v)
        else:
            out[k] = "" if pd.isna(v) else str(v)
    st.write(pd.DataFrame.from_dict(out, orient="index", columns=["value"]))

# =========================
# Epoxy functionality estimation (monofunctional filter)
# =========================
_OXIRANE_PATTERNS = [
    r"C1OC1", r"O1CC1", r"C1CO1",   # common oxirane encodings
    r"c1oc1", r"o1cc1", r"c1co1",
]
_GLYCIDYL_KEYWORDS = ["glycidyl", "oxirane", "epoxide", "epoxy"]

def estimate_epoxy_functionality(abbr: str) -> int | None:
    """
    Estimate number of epoxide (oxirane) rings from SMILES/BigSMILES if available.
    If SMILES is missing, fallback to name heuristics (lower confidence).

    Returns:
      - integer >= 0 when estimated
      - None when cannot estimate
    """
    smi = _smiles_of(abbr)
    if smi:
        # Remove whitespace and try to count explicit oxirane ring patterns
        s = re.sub(r"\s+", "", smi)
        cnt = 0
        for pat in _OXIRANE_PATTERNS:
            cnt += len(re.findall(pat, s))
        # Additional heuristic: count 'O1CC1' variants where ring numbering may differ (rare)
        # Keep conservative to avoid false positives.
        if cnt > 0:
            return int(cnt)
        # If SMILES includes explicit "oxirane" in BigSMILES annotations
        if "oxirane" in s.lower() or "epoxide" in s.lower():
            return 1  # conservative default
        return 0

    # Fallback to name-based estimate
    name = _name_of(abbr).lower()
    if any(k in name for k in _GLYCIDYL_KEYWORDS):
        # Try to catch "di-" / "tri-" / "tetra-" glycidyl wording in names
        if re.search(r"\b(di|bis)\b", name):
            return 2
        if re.search(r"\btri\b", name):
            return 3
        if re.search(r"\btetra\b", name):
            return 4
        return 1

    # Abbrev hints (very conservative)
    a = abbr.lower()
    if "gma" in a or "glycer" in a:
        return 1

    return None

# Precompute (fast enough for typical DB sizes; cached by Streamlit reruns)
@st.cache_data
def build_epoxy_func_table(index_list: list[str]) -> pd.Series:
    d = {}
    for ab in index_list:
        d[ab] = estimate_epoxy_functionality(ab)
    return pd.Series(d)

EP_FUNC = build_epoxy_func_table(list(T.index))

def is_monofunctional_epoxy(abbr: str) -> bool:
    v = EP_FUNC.get(abbr, None)
    return (v is not None) and (int(v) == 1)

def looks_like_epoxy_candidate(abbr: str) -> bool:
    # Used to broaden candidates when DB classification doesn't explicitly mark diluents
    nm = _name_of(abbr).lower()
    smi = _smiles_of(abbr).lower()
    if any(k in nm for k in _GLYCIDYL_KEYWORDS):
        return True
    if any(k in smi for k in ["oxirane", "epoxide"]) and (EP_FUNC.get(abbr, 0) not in (0, None)):
        return True
    # If the SMILES has at least one oxirane ring
    v = EP_FUNC.get(abbr, None)
    if v is not None and v >= 1:
        return True
    return False

# =========================
# PLOTS
# =========================
def plot_heatmap(mat: pd.DataFrame, title: str, cbar: str):
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
    fig, ax = plt.subplots(figsize=(8, 6))
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
tab_form, tab_maps, tab_figs = st.tabs(["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"])

# =========================
# EQUIV LIB EDITOR (top of Formulator)
# =========================
def equiv_editor():
    with st.expander("📚 Library of equivalents (editable)", expanded=False):
        st.caption("Matching is done by **Abbrev** (case-insensitive). You can edit values and they will be used to auto-fill the stoichiometry inputs.")
        df = get_equiv_df()
        df_edit = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="equiv_editor",
        )
        # normalize + save
        df_edit = df_edit.copy()
        if "Abbrev" in df_edit.columns:
            df_edit["Abbrev"] = df_edit["Abbrev"].map(_norm_abbr)
        st.session_state["equiv_lib"] = df_edit

def sync_from_library(selection_key: str, value_key_map: dict, enabled: bool):
    """
    If enabled, reads the selected Abbrev from st.session_state[selection_key]
    and updates st.session_state[...] numeric inputs according to value_key_map,
    e.g. {"NCO_%": "pu_nco", "OH_mgKOH_g": "pu_oh_pol"}.
    """
    if not enabled:
        return
    abbr = st.session_state.get(selection_key)
    if not abbr:
        return
    eq = equiv_lookup(abbr)
    if not eq:
        return
    for col_name, state_key in value_key_map.items():
        v = eq.get(col_name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        st.session_state[state_key] = float(v)

# =========================
# Build lists (with epoxy diluent monofunctional expansion)
# =========================
def build_role_lists():
    eqdf = get_equiv_df()
    roles = {}
    if ("Abbrev" in eqdf.columns) and ("Role" in eqdf.columns):
        r = eqdf.set_index("Abbrev")["Role"].astype(str).str.lower()
        roles = r.to_dict()

    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = sorted(set(list_by_class("extender") + [a for a,v in roles.items() if "extender" in v]))
    crosslinkers = sorted(set(list_by_class("crosslinker") + [a for a,v in roles.items() if ("crosslinker" in v or "triol" in v)]))

    acids = sorted(set(list_by_class("acid_anhydride") + [a for a,v in roles.items() if ("acid" in v or "anhydride" in v or "diacid" in v)]))

    ep_resins = sorted(set(list_by_class("epoxy_resin") + [a for a,v in roles.items() if v == "epoxy resin"]))
    ep_hards  = sorted(set(list_by_class("epoxy_hardener") + [a for a,v in roles.items() if ("hardener" in v or "amine" in v)]))
    ep_dils   = sorted(set(list_by_class("reactive_diluent") + [a for a,v in roles.items() if "diluent" in v]))

    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    sil  = list_by_class("silane")

    # Filter to those present in DB
    def _keep(lst): return [x for x in lst if x in T.index]
    return {
        "isos": _keep(isos),
        "polyols": _keep(polyols),
        "extenders": _keep(extenders),
        "crosslinkers": _keep(crosslinkers),
        "acids": _keep(acids),
        "ep_resins": _keep(ep_resins),
        "ep_hards": _keep(ep_hards),
        "ep_dils": _keep(ep_dils),
        "mons": _keep(mons),
        "solv": _keep(solv),
        "sil": _keep(sil),
        "roles": roles,
    }

# -------------------------
# FORMULATOR
# -------------------------
with tab_form:
    st.subheader("Formulator")

    equiv_editor()

    mode = st.selectbox(
        "Mode",
        ["Pair screening", "PU stoichiometry (NCO/OH)", "Epoxy stoichiometry (EEW/AHEW)", "Ranking (best combinations)"],
        key="mode_select",
    )

    L = build_role_lists()
    isos = L["isos"]
    polyols = L["polyols"]
    extenders = L["extenders"]
    crosslinkers = L["crosslinkers"]
    ep_resins = L["ep_resins"]
    ep_hards = L["ep_hards"]
    ep_dils_base = L["ep_dils"]

    # global maxima for Π normalization
    da_max = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    lock = st.checkbox("🔒 Auto-fill stoichiometry inputs from the equivalents library", value=True, key="lock_to_lib")

    # Epoxy monofunctional expansion controls (formulator-scoped)
    with st.expander("🧩 Epoxy reactive diluent selection (article-aligned)", expanded=False):
        st.caption(
            "When the database does not explicitly label reactive diluents, the app can **estimate epoxy functionality** "
            "from SMILES/BigSMILES and include **monofunctional epoxides (f≈1)** as reactive diluent candidates."
        )
        include_estimated_mono = st.checkbox(
            "Include estimated monofunctional epoxides (f≈1) as reactive diluents",
            value=True,
            key="ep_include_est_mono",
        )
        strict_mono_only = st.checkbox(
            "Restrict diluent list to monofunctional only (f=1)",
            value=True,
            key="ep_mono_only",
        )
        show_epoxy_func_table = st.checkbox(
            "Show epoxy functionality table (debug)",
            value=False,
            key="ep_show_func",
        )

    # Build epoxy diluent list: base + estimated candidates
    ep_dils = list(ep_dils_base)
    if include_estimated_mono:
        est_mono = [ab for ab in T.index if is_monofunctional_epoxy(ab)]
        # optionally broaden candidates beyond monofunctional-only list
        ep_dils = sorted(set(ep_dils + est_mono))

    # If strict monofunctional: keep only those with f=1 (when estimate exists), but keep explicit diluents that lack SMILES estimate
    if strict_mono_only:
        _kept = []
        for ab in ep_dils:
            v = EP_FUNC.get(ab, None)
            if v is None:
                # keep explicit reactive diluents from DB/library even if function cannot be estimated
                # (important when SMILES missing)
                if (ab in ep_dils_base) or (L["roles"].get(ab, "").lower().find("diluent") >= 0):
                    _kept.append(ab)
            else:
                if int(v) == 1:
                    _kept.append(ab)
        ep_dils = sorted(set(_kept))

    # Final safety fallback: if still empty, include any "epoxy candidates" with f>=1 and prefer f=1 if strict_mono_only
    if not ep_dils:
        cand = [ab for ab in T.index if looks_like_epoxy_candidate(ab)]
        if strict_mono_only:
            cand = [ab for ab in cand if is_monofunctional_epoxy(ab)]
        ep_dils = sorted(set(cand))

    if show_epoxy_func_table:
        df_func = pd.DataFrame({
            "Abbrev": list(T.index),
            "Name": [_name_of(a) for a in T.index],
            "Epoxy_func_est": [EP_FUNC.get(a, None) for a in T.index],
            "Class": [T.loc[a, "__class__"] for a in T.index],
        })
        df_func = df_func[df_func["Epoxy_func_est"].notna()].sort_values(["Epoxy_func_est", "Abbrev"])
        st.dataframe(df_func, use_container_width=True, hide_index=True)

    if mode == "Pair screening":
        st.markdown("### Pair screening")
        c1, c2 = st.columns(2)
        with c1:
            A = select_abbr("Component A", list(T.index), key="pair_A")
        with c2:
            B = select_abbr("Component B", list(T.index), key="pair_B")

        if A and B:
            a = T.loc[A]; b = T.loc[B]
            ra = Ra(a, b)
            da = delta_a(a, b)
            ds = delta_sigmaL(a, b)
            pi = Pi(da, ds, da_max, ds_max)

            st.markdown("#### Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Ra (MPa$^{1/2}$)", f"{ra:.2f}")
            r2.metric("Δδa (MPa$^{1/2}$)", f"{da:.2f}")
            r3.metric("ΔσL (mN/m)", f"{ds:.2f}" if np.isfinite(ds) else "n/a")
            r4.metric("Π (dimensionless)", f"{pi:.3f}")

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
        st.caption("Uses %NCO and OH number. Inputs can be auto-filled from the internal equivalents library.")

        c1, c2, c3 = st.columns(3)
        with c1:
            iso = select_abbr("Isocyanate", isos, key="pu_iso")
        with c2:
            pol = select_abbr("Polyol", polyols, key="pu_pol")
        with c3:
            use_ext = st.checkbox("Include chain extender (diol)", value=True, key="pu_use_ext")

        ext = None
        if use_ext:
            ext = select_abbr("Chain extender (diol)", extenders, key="pu_ext")

        use_xl = st.checkbox("Include Crosslinker/triol", value=False, key="pu_use_xl")
        xl = None
        if use_xl:
            xl = select_abbr("Crosslinker/triol", crosslinkers, key="pu_xl")

        # Callbacks: when selection changes, update numeric inputs
        sync_from_library("pu_iso", {"NCO_%": "pu_nco"}, lock)
        sync_from_library("pu_pol", {"OH_mgKOH_g": "pu_oh_pol"}, lock)
        if use_ext and ext:
            sync_from_library("pu_ext", {"OH_mgKOH_g": "pu_oh_ext"}, lock)
        if use_xl and xl:
            sync_from_library("pu_xl", {"OH_mgKOH_g": "pu_oh_xl"}, lock)

        st.markdown("#### Inputs")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            nco = st.number_input("%NCO (wt%)", min_value=0.0, max_value=60.0,
                                  value=float(st.session_state.get("pu_nco", 33.0)),
                                  step=0.1, key="pu_nco")
        with cc2:
            oh_pol = st.number_input("OH number polyol (mgKOH/g)", min_value=0.0, max_value=3000.0,
                                     value=float(st.session_state.get("pu_oh_pol", 56.0)),
                                     step=1.0, key="pu_oh_pol")
        with cc3:
            oh_ext_default = float(st.session_state.get("pu_oh_ext", 1200.0 if use_ext else 0.0))
            oh_ext = st.number_input("OH number extender (mgKOH/g)", min_value=0.0, max_value=3000.0,
                                     value=oh_ext_default,
                                     step=1.0, key="pu_oh_ext", disabled=not (use_ext and ext))

        oh_xl_default = float(st.session_state.get("pu_oh_xl", 1000.0))
        oh_xl = st.number_input("OH number crosslinker (mgKOH/g)", min_value=0.0, max_value=4000.0,
                                value=oh_xl_default,
                                step=1.0, key="pu_oh_xl", disabled=not (use_xl and xl))

        idx = st.number_input("NCO/OH index", min_value=0.5, max_value=2.5, value=1.05, step=0.01, key="pu_index")
        mass_mode = st.radio(
            "Mass target mode",
            ["Total mass (includes isocyanate)", "Polyol-side mass (polyol+extender+crosslinker)"],
            horizontal=True, key="pu_mass_mode"
        )
        target = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key="pu_target")

        # Fractions on polyol side
        ext_frac = 0.0
        xl_frac = 0.0
        if use_ext and ext:
            ext_frac = st.slider("Extender mass fraction (in polyol-side)", 0.0, 0.8, 0.2, step=0.01, key="pu_ext_frac")
        if use_xl and xl:
            max_xl = max(0.0, 0.9 - ext_frac)
            xl_frac = st.slider("Crosslinker/triol mass fraction (in polyol-side)", 0.0, float(max_xl), min(0.1, float(max_xl)), step=0.01, key="pu_xl_frac")
        pol_frac = max(0.0, 1.0 - ext_frac - xl_frac)

        # Eq weights:
        EW_OH_pol = 56100.0 / max(oh_pol, 1e-12)
        EW_OH_ext = 56100.0 / max(oh_ext, 1e-12) if (use_ext and ext and oh_ext > 0) else np.inf
        EW_OH_xl  = 56100.0 / max(oh_xl, 1e-12) if (use_xl and xl and oh_xl > 0) else np.inf
        EW_NCO    = 4200.0  / max(nco,   1e-12)

        if iso and pol:
            if mass_mode == "Polyol-side mass (polyol+extender+crosslinker)":
                m_pol_side = target
                m_pol = m_pol_side * pol_frac
                m_ext = m_pol_side * ext_frac if (use_ext and ext) else 0.0
                m_xl  = m_pol_side * xl_frac if (use_xl and xl) else 0.0
                eq_OH = (m_pol / EW_OH_pol) + (m_ext / EW_OH_ext if (use_ext and ext) else 0.0) + (m_xl / EW_OH_xl if (use_xl and xl) else 0.0)
                m_iso = idx * eq_OH * EW_NCO
                m_total = m_pol_side + m_iso
            else:
                term = (pol_frac / EW_OH_pol) \
                       + (ext_frac / EW_OH_ext if (use_ext and ext) else 0.0) \
                       + (xl_frac  / EW_OH_xl if (use_xl and xl) else 0.0)
                k = idx * EW_NCO * term
                m_iso = (k / (1.0 + k)) * target
                m_pol_side = target - m_iso
                m_pol = m_pol_side * pol_frac
                m_ext = m_pol_side * ext_frac
                m_xl  = m_pol_side * xl_frac
                m_total = target

            st.markdown("#### Calculated masses")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Isocyanate (g)", f"{m_iso:.2f}")
            r3.metric("Polyol (g)", f"{m_pol:.2f}")
            r4.metric("Extender (g)", f"{m_ext:.2f}" if (use_ext and ext) else "0.00")
            r5.metric("Crosslinker (g)", f"{m_xl:.2f}" if (use_xl and xl) else "0.00")

            st.markdown("#### Compatibility indices (screening)")
            rows = []
            for comp_id, tag in [(pol, "polyol")] + ([(ext, "extender")] if (use_ext and ext) else []) + ([(xl, "crosslinker")] if (use_xl and xl) else []):
                a = T.loc[comp_id]; b = T.loc[iso]
                ra = Ra(a, b); da = delta_a(a, b); ds = delta_sigmaL(a, b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"pair": f"{tag} × isocyanate", "A": comp_id, "B": iso, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
            dfm = pd.DataFrame(rows)
            if not dfm.empty:
                raM = float(dfm["Ra"].max()); daM = float(dfm["Δδa"].max()); piM = float(dfm["Π"].max())
                dfm["Score_%"] = [score_percent(r, d, p, raM, daM, piM) for r, d, p in zip(dfm["Ra"], dfm["Δδa"], dfm["Π"])]
                st.dataframe(dfm.round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3, "Score_%": 1}), use_container_width=True, hide_index=True)

    elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.markdown("### Epoxy stoichiometry (EEW/AHEW)")
        st.caption("Total mass targeting. Inputs can be auto-filled from the internal equivalents library.")

        c1, c2, c3 = st.columns(3)
        with c1:
            resin = select_abbr("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = select_abbr("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

        dil = None
        if use_dil:
            # This list is now article-aligned: any estimated monofunctional epoxy can appear here.
            dil = select_abbr("Reactive diluent (monofunctional epoxy, f≈1)", ep_dils, key="ep_dil")

        # auto-fill from library
        sync_from_library("ep_resin", {"EEW_g_eq": "ep_eew"}, lock)
        sync_from_library("ep_hard", {"AHEW_g_eq": "ep_ahew"}, lock)
        if use_dil and dil:
            sync_from_library("ep_dil", {"EEW_g_eq": "ep_dil_eew"}, lock)

        st.markdown("#### Inputs")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            EEW_res = st.number_input("Resin EEW (g/eq)", min_value=50.0, max_value=3000.0,
                                      value=float(st.session_state.get("ep_eew", 185.0)),
                                      step=1.0, key="ep_eew")
        with cc2:
            AHEW = st.number_input("Hardener AHEW (g/eq)", min_value=10.0, max_value=2000.0,
                                   value=float(st.session_state.get("ep_ahew", 30.0)),
                                   step=1.0, key="ep_ahew")
        with cc3:
            ratio = st.number_input("Epoxy/Active-H eq ratio", min_value=0.5, max_value=2.0, value=1.0, step=0.01, key="ep_ratio")

        # Optional: show estimated epoxy functionality of selected diluent
        if use_dil and dil:
            f_est = EP_FUNC.get(dil, None)
            if f_est is None:
                st.info("Epoxy functionality estimate for selected diluent: **n/a** (missing SMILES/name cues).")
            else:
                st.info(f"Epoxy functionality estimate for selected diluent: **f = {int(f_est)}** (from SMILES/name heuristics).")

        mass_mode = st.radio("Mass target mode", ["Total mass (includes hardener)", "Resin-side mass (resin+diluent)"], horizontal=True, key="ep_mass_mode")
        target = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key="ep_target")

        dil_frac = 0.0
        EEW_dil = np.inf
        if use_dil and dil:
            dil_frac = st.slider("Diluente mass fraction (in resin-side)", 0.0, 0.8, 0.2, step=0.01, key="ep_dil_frac")
            EEW_dil = st.number_input("Diluente EEW (g/eq)", min_value=50.0, max_value=3000.0,
                                      value=float(st.session_state.get("ep_dil_eew", 150.0)),
                                      step=1.0, key="ep_dil_eew")

        term = (1.0 - dil_frac) / EEW_res + (dil_frac / EEW_dil if (use_dil and dil) else 0.0)
        g = (AHEW / max(ratio, 1e-12)) * term  # hardener mass per resin-side mass

        if resin and hard:
            if mass_mode == "Resin-side mass (resin+diluent)":
                m_res_side = target
                m_hard = g * m_res_side
                m_total = m_res_side + m_hard
            else:
                m_hard = (g / (1.0 + g)) * target
                m_res_side = target - m_hard
                m_total = target

            m_dil = m_res_side * dil_frac if (use_dil and dil) else 0.0
            m_res = m_res_side - m_dil

            st.markdown("#### Calculated masses")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total (g)", f"{m_total:.2f}")
            r2.metric("Hardener (g)", f"{m_hard:.2f}")
            r3.metric("Resin (g)", f"{m_res:.2f}")
            r4.metric("Reactive diluent (g)", f"{m_dil:.2f}" if (use_dil and dil) else "0.00")

            st.markdown("#### Compatibility indices (screening)")
            rows = []
            for comp_id, tag in [(hard, "hardener")] + ([(dil, "diluent")] if (use_dil and dil) else []):
                a = T.loc[comp_id]; b = T.loc[resin]
                ra = Ra(a, b); da = delta_a(a, b); ds = delta_sigmaL(a, b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"pair": f"{tag} × resin", "A": comp_id, "B": resin, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
            dfm = pd.DataFrame(rows)
            if not dfm.empty:
                raM = float(dfm["Ra"].max()); daM = float(dfm["Δδa"].max()); piM = float(dfm["Π"].max())
                dfm["Score_%"] = [score_percent(r, d, p, raM, daM, piM) for r, d, p in zip(dfm["Ra"], dfm["Δδa"], dfm["Π"])]
                st.dataframe(dfm.round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3, "Score_%": 1}), use_container_width=True, hide_index=True)

    else:
        st.markdownжин

    # NOTE:
    # You asked to keep the Heatmaps/Figures modules as in your provided code.
    # They are included below unchanged in structure; only the epoxy diluent list used there is expanded similarly.

# -------------------------
# HEATMAPS
# -------------------------
with tab_maps:
    st.subheader("Heatmaps (grouped by purpose)")

    # Metric selector (matches the manuscript descriptors)
    metric = st.radio("Heatmap metric", ["Ra", "Δδa", "Π"], horizontal=True, key="hm_metric")

    purpose = st.selectbox(
        "Purpose",
        ["Polyesters", "Polyurethanes", "Epoxies", "Vinyls", "Silanes"],
        key="hm_purpose",
    )

    use_all_defaults = st.checkbox(
        "Use full lists by default (recommended)",
        value=True,
        help="If disabled, the app preselects only the first items to keep figures small.",
        key="hm_use_all_defaults",
    )

    L = build_role_lists()
    acids = L["acids"]
    isos = L["isos"]
    polyols = L["polyols"]
    extenders = L["extenders"]
    crosslinkers = L["crosslinkers"]
    ep_resins = L["ep_resins"]
    ep_hards = L["ep_hards"]
    ep_dils_base = L["ep_dils"]
    mons = L["mons"]
    solv = L["solv"]
    sil = L["sil"]

    # Expand epoxy diluents here too (article-aligned)
    ep_dils = list(ep_dils_base)
    if st.session_state.get("ep_include_est_mono", True):
        ep_dils = sorted(set(ep_dils + [ab for ab in T.index if is_monofunctional_epoxy(ab)]))
    if st.session_state.get("ep_mono_only", True):
        _kept = []
        for ab in ep_dils:
            v = EP_FUNC.get(ab, None)
            if v is None:
                if (ab in ep_dils_base) or (L["roles"].get(ab, "").lower().find("diluent") >= 0):
                    _kept.append(ab)
            else:
                if int(v) == 1:
                    _kept.append(ab)
        ep_dils = sorted(set(_kept))
    if not ep_dils:
        cand = [ab for ab in T.index if looks_like_epoxy_candidate(ab)]
        if st.session_state.get("ep_mono_only", True):
            cand = [ab for ab in cand if is_monofunctional_epoxy(ab)]
        ep_dils = sorted(set(cand))

    # Normalization constants for Π (computed from the loaded database)
    da_max_hm = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max_hm = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    def _default(opts, n=12):
        return opts if use_all_defaults else opts[:min(n, len(opts))]

    def build_matrix(rows, cols):
        mat = pd.DataFrame(index=rows, columns=cols, dtype=float)
        for r in rows:
            a = T.loc[r]
            for c in cols:
                b = T.loc[c]
                da = delta_a(a, b)
                ds = delta_sigmaL(a, b)
                if metric == "Ra":
                    mat.loc[r, c] = Ra(a, b)
                elif metric == "Δδa":
                    mat.loc[r, c] = da
                else:  # Π
                    mat.loc[r, c] = Pi(da, ds, da_max_hm, ds_max_hm)
        return mat

    cbar_label = (
        "Ra (MPa$^{1/2}$)" if metric == "Ra" else
        "Δδa (MPa$^{1/2}$)" if metric == "Δδa" else
        "Π (dimensionless)"
    )

    if purpose == "Polyesters":
        st.caption("Polyols/diols/triols × diacids/anhydrides (heatmap).")
        left = st.multiselect(
            "Polyols (rows)",
            sorted(set(polyols + extenders + crosslinkers)),
            default=_default(sorted(set(polyols + extenders + crosslinkers)), 18),
            format_func=label,
            key="hm_polyesters_rows",
        )
        right = st.multiselect(
            "Acids/anhydrides (cols)",
            acids,
            default=_default(acids, 18),
            format_func=label,
            key="hm_polyesters_cols",
        )
        if left and right:
            mat = build_matrix(left, right)
            fig = plot_heatmap(mat, f"POLYESTERS — Polyols/Diols/Triols × Diacids/Anhydrides ({metric})", cbar_label)
            st.pyplot(fig)

    elif purpose == "Polyurethanes":
        st.caption("Isocyanates × polyols/extenders/crosslinkers (heatmap).")
        isos_sel = st.multiselect(
            "Isocyanates (cols)",
            isos,
            default=_default(isos, 10),
            format_func=label,
            key="hm_pu_isos",
        )

        show_ext = st.checkbox("Include extenders", value=True, key="hm_pu_show_ext")
        show_xl  = st.checkbox("Include crosslinkers/triols", value=False, key="hm_pu_show_xl")

        pol_sel = st.multiselect(
            "Polyols (rows)",
            sorted(set(polyols + extenders + crosslinkers)),
            default=_default(sorted(set(polyols + extenders + crosslinkers)), 18),
            format_func=label,
            key="hm_pu_polyols",
        )
        if pol_sel and isos_sel:
            mat = build_matrix(pol_sel, isos_sel)
            fig = plot_heatmap(mat, f"PU — Polyols × Isocyanates ({metric})", cbar_label)
            st.pyplot(fig)

        if show_ext:
            ext_sel = st.multiselect(
                "Extenders (rows)",
                extenders,
                default=_default(extenders, 18),
                format_func=label,
                key="hm_pu_extenders",
            )
            if ext_sel and isos_sel:
                mat2 = build_matrix(ext_sel, isos_sel)
                fig2 = plot_heatmap(mat2, f"PU — Extenders × Isocyanates ({metric})", cbar_label)
                st.pyplot(fig2)

        if show_xl and crosslinkers:
            xl_sel = st.multiselect(
                "Crosslinkers/triols (rows)",
                crosslinkers,
                default=_default(crosslinkers, 12),
                format_func=label,
                key="hm_pu_crosslinkers",
            )
            if xl_sel and isos_sel:
                mat3 = build_matrix(xl_sel, isos_sel)
                fig3 = plot_heatmap(mat3, f"PU — Crosslinkers/triols × Isocyanates ({metric})", cbar_label)
                st.pyplot(fig3)

    elif purpose == "Epoxies":
        st.caption("Epoxy resins × hardeners and reactive diluents (heatmap).")
        res_sel = st.multiselect(
            "Resins (cols)",
            ep_resins,
            default=_default(ep_resins, 12),
            format_func=label,
            key="hm_ep_resins",
        )
        hard_sel = st.multiselect(
            "Hardeners (rows)",
            ep_hards,
            default=_default(ep_hards, 18),
            format_func=label,
            key="hm_ep_hards",
        )
        if hard_sel and res_sel:
            mat = build_matrix(hard_sel, res_sel)
            fig = plot_heatmap(mat, f"EPOXY — Hardeners × Resins ({metric})", cbar_label)
            st.pyplot(fig)

        dil_sel = st.multiselect(
            "Reactive diluents (rows)",
            ep_dils,
            default=_default(ep_dils, 18),
            format_func=label,
            key="hm_ep_dils",
        )
        if dil_sel and res_sel:
            mat2 = build_matrix(dil_sel, res_sel)
            fig2 = plot_heatmap(mat2, f"EPOXY — Reactive diluents × Resins ({metric})", cbar_label)
            st.pyplot(fig2)

    elif purpose == "Vinyls":
        st.caption("Vinyl monomers × solvents/plasticizers (heatmap).")
        mon_sel = st.multiselect(
            "Monomers (rows)",
            mons,
            default=_default(mons, 18),
            format_func=label,
            key="hm_vinyl_mons",
        )
        sol_sel = st.multiselect(
            "Solvents/plasticizers (cols)",
            solv,
            default=solv,
            format_func=label,
            key="hm_vinyl_solv",
        )
        if mon_sel and sol_sel:
            mat = build_matrix(mon_sel, sol_sel)
            fig = plot_heatmap(mat, f"VINYLS — Monomers × Solvents/Plasticizers ({metric})", cbar_label)
            st.pyplot(fig)

    else:
        st.caption("Silane coupling agents × selected targets (heatmap).")
        sil_sel = st.multiselect(
            "Silanes (rows)",
            sil,
            default=_default(sil, 18),
            format_func=label,
            key="hm_silanes",
        )

        # Targets: alcohols/water (by name) + vinyl monomers + selected polyols
        target_pool = []
        for ab in T.index:
            nm = _name_of(ab).lower()
            if ("water" in nm) or ("alcohol" in nm) or ("ethanol" in nm) or ("methanol" in nm) or ("propan" in nm):
                target_pool.append(ab)
        target_pool = sorted(set(target_pool + mons + polyols))

        tar_sel = st.multiselect(
            "Targets (cols)",
            target_pool,
            default=_default(target_pool, 18),
            format_func=label,
            key="hm_sil_targets",
        )
        if sil_sel and tar_sel:
            mat = build_matrix(sil_sel, tar_sel)
            fig = plot_heatmap(mat, f"SILANES — Silanes × Targets ({metric})", cbar_label)
            st.pyplot(fig)

# -------------------------
# FIGURES
# -------------------------
with tab_figs:
    st.subheader("FIGURES — Ra vs Δδa and Ra vs Π")
    st.caption("Axes follow the manuscript: **Ra** on x-axis; **Δδa** or **Π** on y-axis.")

    fam = st.selectbox("Family", ["Polyesters", "Polyurethanes", "Epoxies", "Vinyls", "Silanes"], key="fig_family")

    use_all_fig = st.checkbox("Use full lists (may be slower)", value=True, key="fig_all")
    def _def(opts, n):
        return opts if use_all_fig else opts[:n]

    da_max = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    def make_pairs(A_list, B_list, tagA="A", tagB="B", hue_name="pair"):
        rows = []
        for a_id in A_list:
            a = T.loc[a_id]
            for b_id in B_list:
                if a_id == b_id:
                    continue
                b = T.loc[b_id]
                ra = Ra(a, b)
                da = delta_a(a, b)
                ds = delta_sigmaL(a, b)
                pi = Pi(da, ds, da_max, ds_max)
                rows.append({"A": a_id, "B": b_id, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi, hue_name: f"{tagA}×{tagB}"})
        return pd.DataFrame(rows)

    L = build_role_lists()
    acids = L["acids"]
    isos = L["isos"]
    polyols = L["polyols"]
    extenders = L["extenders"]
    crosslinkers = L["crosslinkers"]
    ep_resins = L["ep_resins"]
    ep_hards = L["ep_hards"]
    mons = L["mons"]
    solv = L["solv"]
    sil = L["sil"]

    # Use same epoxy diluent expansion as elsewhere
    ep_dils = list(L["ep_dils"])
    if st.session_state.get("ep_include_est_mono", True):
        ep_dils = sorted(set(ep_dils + [ab for ab in T.index if is_monofunctional_epoxy(ab)]))
    if st.session_state.get("ep_mono_only", True):
        _kept = []
        for ab in ep_dils:
            v = EP_FUNC.get(ab, None)
            if v is None:
                if (ab in L["ep_dils"]) or (L["roles"].get(ab, "").lower().find("diluent") >= 0):
                    _kept.append(ab)
            else:
                if int(v) == 1:
                    _kept.append(ab)
        ep_dils = sorted(set(_kept))
    if not ep_dils:
        cand = [ab for ab in T.index if looks_like_epoxy_candidate(ab)]
        if st.session_state.get("ep_mono_only", True):
            cand = [ab for ab in cand if is_monofunctional_epoxy(ab)]
        ep_dils = sorted(set(cand))

    if fam == "Polyesters":
        A = st.multiselect("Polyols", polyols, default=_def(polyols, 10), format_func=label)
        B = st.multiselect("Acids/anhydrides", acids, default=_def(acids, 10), format_func=label)
        dfp = make_pairs(A, B, tagA="polyol", tagB="acid")
    elif fam == "Polyurethanes":
        iso_sel = st.multiselect("Isocyanates", isos, default=_def(isos, 8), format_func=label)
        pol_sel = st.multiselect("Polyols", polyols, default=_def(polyols, 10), format_func=label)
        alc_sel = st.multiselect("Alcohols (extenders + crosslinkers)", sorted(set(extenders + crosslinkers)),
                                 default=_def(sorted(set(extenders + crosslinkers)), 10), format_func=label)
        df1 = make_pairs(pol_sel, iso_sel, tagA="polyol", tagB="iso")
        df2 = make_pairs([a for a in alc_sel if a in T.index], iso_sel, tagA="alc", tagB="iso")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()
    elif fam == "Epoxies":
        res_sel = st.multiselect("Resins", ep_resins, default=_def(ep_resins, 5), format_func=label)
        hard_sel = st.multiselect("Hardeners", ep_hards, default=_def(ep_hards, 10), format_func=label)
        dil_sel = st.multiselect("Reactive diluents (monofunctional, f≈1)", ep_dils, default=_def(ep_dils, 12), format_func=label)
        df1 = make_pairs(hard_sel, res_sel, tagA="hard", tagB="resin")
        df2 = make_pairs(dil_sel, res_sel, tagA="dil", tagB="resin")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()
    elif fam == "Vinyls":
        mon_sel = st.multiselect("Monomers", mons, default=mons, format_func=label)
        sol_sel = st.multiselect("Solvents/plasticizers", solv, default=solv, format_func=label)
        dfp = make_pairs(mon_sel, sol_sel, tagA="mon", tagB="solv")
    else:
        sil_sel = st.multiselect("Silanes", sil, default=_def(sil, 10), format_func=label)
        target_pool = []
        for ab in T.index:
            nm = _name_of(ab).lower()
            if "water" in nm or "alcohol" in nm:
                target_pool.append(ab)
        target_pool = sorted(set(target_pool + _def(mons, 10) + _def(polyols, 10)))
        tar_sel = st.multiselect("Targets", target_pool, default=_def(target_pool, 12), format_func=label)
        dfp = make_pairs(sil_sel, tar_sel, tagA="sil", tagB="target")

    if dfp.empty:
        st.warning("No pairs available (check selections).")
        st.stop()

    st.pyplot(plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)"))
    st.pyplot(plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)"))

    st.markdown("#### Data used in figures")
    st.dataframe(dfp.sort_values("Ra").round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3}), use_container_width=True, hide_index=True)
