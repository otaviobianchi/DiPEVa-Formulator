# ============================================================
# DiPEVa Formulator — FULL UPDATED SCRIPT
# Fixes:
#   1) BDGE is ALWAYS treated as epoxy resin (difunctional), never reactive diluent.
#   2) Reactive diluents pool includes ALL candidates using 3 independent signals:
#        (a) Effective class == reactive_diluent (from DB class, role, or inference)
#        (b) Epoxy functionality estimate == 1 (SMILES/name/abbr heuristics)
#        (c) Name/abbr token cues (Glycidyl / epoxy-alkyl / epichlorohydrin-like)
#   3) Preserves an existing class column from your Excel if present (Class/__class__/Type...).
#   4) Removes the previous “Debug — Epoxy detection / pools” block entirely.
# ============================================================

import re
from pathlib import Path
from typing import Optional, Callable, Dict, List

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

# -------------------------
# Column detection
# -------------------------
COL: Dict[str, Optional[str]] = {}

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
COL["mw"]     = _find_col(["MW", "Molecular weight", "MolWt", "Molar mass"])
COL["dD"]     = _find_col(["δD (MPa1/2)", "deltaD", "dD", "δD"])
COL["dP"]     = _find_col(["δP (MPa1/2)", "deltaP", "dP", "δP"])
COL["dH"]     = _find_col(["δH (MPa1/2)", "deltaH", "dH", "δH"])
COL["sig"]    = _find_col(["σL (mN·m-1)", "sigmaL", "σL", "surface tension"])
COL["da"]     = _find_col(["δa (MPa1/2)", "delta a", "δa", "delta_a"])

# Optional pre-existing classification column in DB
COL["db_class"] = _find_col(["__class__", "class", "Class", "ChemicalClass", "chemical_class", "Type", "type", "Family", "family"])

missing = [k for k in ["abbr", "dD", "dP", "dH"] if COL.get(k) is None]
if missing:
    st.error(
        "Database is missing required columns: "
        f"{missing}\nFound columns: {list(df_raw.columns)}"
    )
    st.stop()

# -------------------------
# Build indexed table by Abbrev
# -------------------------
T = df_raw.copy()
T["__abbr__"] = (
    T[COL["abbr"]].astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
T = T[T["__abbr__"].ne("")].copy()
T["__abbr__"] = T["__abbr__"].str.upper()

T = T.drop_duplicates(subset="__abbr__", keep="first").set_index("__abbr__", drop=True)

# numeric coercion
for k in ["dD", "dP", "dH", "sig", "da", "mw"]:
    c = COL.get(k)
    if c and c in T.columns:
        T[c] = pd.to_numeric(T[c], errors="coerce")

# =========================
# DEFAULT EQUIVALENTS LIBRARY (editable)
# =========================
DEFAULT_EQUIV = [
    # Epoxy resins / key entries
    {"Abbrev": "DGEBA", "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 185.0, "AHEW_g_eq": None},
    {"Abbrev": "BDGE",  "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.0,  "AHEW_g_eq": None},

    # Reactive diluents (add yours here; UI editor can extend)
    {"Abbrev": "ECH",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.5, "AHEW_g_eq": None},
    {"Abbrev": "BGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "PGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "CGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "GMA",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Hardeners
    {"Abbrev": "DDM",   "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 30.0},
    {"Abbrev": "DDS",   "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 62.0},
    {"Abbrev": "DICY",  "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": 42.0},
    {"Abbrev": "DETA",  "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TETA",  "Role": "Hardener",  "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
]

def _norm_abbr(s: str) -> str:
    return str(s).strip().upper()

if "equiv_lib" not in st.session_state:
    st.session_state["equiv_lib"] = pd.DataFrame(DEFAULT_EQUIV)

def get_equiv_df() -> pd.DataFrame:
    df = st.session_state["equiv_lib"].copy()
    if not df.empty and "Abbrev" in df.columns:
        df["Abbrev"] = df["Abbrev"].map(_norm_abbr)
    return df

def equiv_lookup(abbr: str) -> dict:
    ab = _norm_abbr(abbr)
    df = get_equiv_df()
    if df.empty or "Abbrev" not in df.columns:
        return {}
    hit = df[df["Abbrev"] == ab]
    return hit.iloc[0].to_dict() if (not hit.empty) else {}

def _role_map_from_lib() -> Dict[str, str]:
    df = get_equiv_df()
    if df.empty or "Abbrev" not in df.columns or "Role" not in df.columns:
        return {}
    m = {}
    for a, r in zip(df["Abbrev"].astype(str), df["Role"].astype(str)):
        a = _norm_abbr(a)
        r = str(r).strip().lower()
        if a:
            m[a] = r
    return m

# =========================
# HELPERS: labels/props
# =========================
def _name_of(abbr: str) -> str:
    if COL["name"] and COL["name"] in T.columns:
        v = T.loc[abbr, COL["name"]]
        return str(v) if not pd.isna(v) else abbr
    return abbr

def label(abbr: str) -> str:
    nm = _name_of(abbr)
    base = str(abbr).strip()
    nm_s = str(nm).strip()
    if nm_s.lower() in ("", "nan", "none") or nm_s.upper() == base.upper():
        return base
    return f"{base} — {nm_s}"

def _smiles_of(abbr: str) -> str:
    c = COL.get("smiles")
    if not c or c not in T.columns:
        return ""
    v = T.loc[abbr, c]
    return "" if pd.isna(v) else str(v)

def _mw_of(abbr: str) -> float:
    c = COL.get("mw")
    if not c or c not in T.columns:
        return float("nan")
    v = T.loc[abbr, c]
    try:
        return float(v)
    except Exception:
        return float("nan")

# =========================
# METRICS
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
    nda = da/(da_max + 1e-12)
    if np.isfinite(ds):
        nds = ds/(ds_max + 1e-12)
        return float(nda + nds)
    return float(nda)

def score_percent(ra: float, da: float, pi: float, ra_max: float, da_max: float, pi_max: float) -> float:
    n_ra = ra/(ra_max + 1e-12)
    n_da = da/(da_max + 1e-12)
    n_pi = pi/(pi_max + 1e-12)
    s = 100.0*(1.0 - (n_ra + n_da + n_pi)/3.0)
    return float(np.clip(s, 0, 100))

# =========================
# EPOXY: ring counting + function (robust)
# =========================
def _count_epoxide_rings(smiles: str) -> int:
    if not smiles:
        return 0
    s = smiles.replace(" ", "")
    n = 0
    n += len(re.findall(r"C(\d)OC\1", s))
    n += len(re.findall(r"C(\d)CO\1", s))
    n += len(re.findall(r"O(\d)CC\1", s))
    n += len(re.findall(r"OC(\d)OC\1", s))
    return int(n)

def estimate_epoxy_function(abbr: str) -> int:
    ab = str(abbr).strip().upper()
    if ab not in T.index:
        return 0

    # Hard overrides
    if ab == "BDGE":
        return 2
    if ab == "ECH":
        return 1
    if ab in {"DGEBA", "DGEBF", "EPN", "ECN"}:
        return 2

    # 1) SMILES-based
    smi = _smiles_of(ab)
    if smi:
        cnt = _count_epoxide_rings(smi)
        if cnt > 0:
            return int(min(cnt, 8))

    # 2) Name/abbr heuristics (broad)
    name = _name_of(ab).lower()
    a = ab.lower()

    # strong epoxy cues
    if not any(k in name for k in ["epoxy", "oxirane", "epoxide", "glycid"]) and not any(k in a for k in ["dge", "gde", "gma", "ech", "bdge"]):
        return 0

    # explicit n- glycidyl
    if any(k in name for k in ["tetraglycidyl", "tetra-glycidyl"]):
        return 4
    if any(k in name for k in ["triglycidyl", "tri-glycidyl"]):
        return 3
    if any(k in name for k in ["diglycidyl", "di-glycidyl"]):
        return 2

    # resin cues
    if any(k in name for k in ["bisphenol", "novolac", "resin"]):
        return 2

    # monoglycidyl cues (reactive diluent)
    mono_signals = [
        "monoglycidyl", "mono glycidyl",
        "glycidyl ether", "glycidyl ester",
        "glycidyl methacrylate", "glycidyl acrylate",
        "epichlorohydrin"
    ]
    if any(k in name for k in mono_signals):
        return 1

    # abbr: DGE/BDGE usually di-epoxy, GMA/ECH usually mono
    if "bdge" in a or "dge" in a:
        return 2
    if any(tok in a for tok in ["gma", "ech"]):
        return 1

    # fallback: epoxy-like but unknown => assume 1 (safer to populate diluents pool)
    return 1

if "__epoxy_fn__" not in T.columns:
    T["__epoxy_fn__"] = [estimate_epoxy_function(a) for a in T.index]
T["__epoxy_fn__"] = pd.to_numeric(T["__epoxy_fn__"], errors="coerce").fillna(0).astype(int)

# =========================
# CLASSIFICATION (preserve DB class if present)
# =========================
def normalize_db_class(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "_")
    s = s.replace("reactive_diluents", "reactive_diluent")
    s = s.replace("diluent_reactive", "reactive_diluent")
    s = s.replace("epoxyresin", "epoxy_resin")
    s = s.replace("hardener_epoxy", "epoxy_hardener")
    return s

KNOWN_CLASSES = {
    "epoxy_resin","reactive_diluent","epoxy_hardener",
    "isocyanate","polyol","extender","crosslinker",
    "acid_anhydride","vinyl_monomer","solvent_plasticizer","silane","other"
}

def classify_row_infer(abbr: str) -> str:
    name = _name_of(abbr).lower()
    a = str(abbr).lower()

    # Epoxy hardeners
    if ("amine" in name) or ("hardener" in name) or ("curing" in name) or re.search(r"\bdeta\b|\bteta\b|\bddm\b|\bdds\b|\bdicy\b|\bjeffamine\b", a):
        return "epoxy_hardener"

    # Epoxy family
    fn = int(T.loc[abbr, "__epoxy_fn__"])
    if (fn > 0) or any(k in name for k in ["epoxy", "oxirane", "glycid"]):
        if abbr.upper() == "BDGE":
            return "epoxy_resin"
        if fn >= 2:
            return "epoxy_resin"
        if fn == 1:
            return "reactive_diluent"
        return "reactive_diluent"

    # PU family
    if ("isocyanate" in name) or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", a + " " + name):
        return "isocyanate"

    # Triols
    if ("triol" in name) or ("glycer" in name) or ("trimethylol" in name) or ("sorbit" in name) or ("pentaerythrit" in name) or re.search(r"\btmp\b|\bgly\b|\bper\b|\bsor\b", a):
        return "crosslinker"

    # Chain extenders
    if ("butanediol" in name) or ("hexanediol" in name) or ("propanediol" in name) or ("ethylene glycol" in name) or ("diethylene glycol" in name) or ("triethylene glycol" in name) or re.search(r"\beg\b|\bdeg\b|\bteg\b", a):
        return "extender"

    # Polyols
    if ("polyol" in name) or ("polyether" in name) or ("polyester" in name) or ("polycarbonate" in name) or re.search(r"\bpeg\d+|\bppg\d+|\bptmeg\d*|\bpcl\d+|\bpcdl\d+|\bpbd\d*", a):
        return "polyol"

    # Acids/anhydrides
    if ("acid" in name) or ("anhydride" in name) or re.search(r"\b(phthalic|isophthalic|terephthalic|succinic|sebacic|adipic|glutaric|maleic|fumaric)\b", name) or re.search(r"\bmah\b|\bhhpa\b|\bmhhpa\b", a):
        return "acid_anhydride"

    # Vinyl monomers
    if ("methacrylate" in name) or ("acrylate" in name) or ("styrene" in name) or ("vinyl acetate" in name) or re.search(r"\bst\b|\ban\b|\bmma\b|\bvac\b", a):
        return "vinyl_monomer"

    # Solvents/plasticizers
    if ("solvent" in name) or ("plasticizer" in name) or re.search(r"\b(dmso|dmf|nmp|thf|toluene|xylene|acetone|ethyl acetate|butyl acetate)\b", name):
        return "solvent_plasticizer"

    # Silanes
    if ("silane" in name) or re.search(r"\bteos\b|\baptes\b|\bgpts\b|\bgptms\b", a):
        return "silane"

    return "other"

def classify_row_base(abbr: str) -> str:
    dbc = COL.get("db_class")
    if dbc and dbc in T.columns:
        v = T.loc[abbr, dbc]
        if not pd.isna(v) and str(v).strip():
            cls = normalize_db_class(v)
            if cls in KNOWN_CLASSES:
                return cls
    return classify_row_infer(abbr)

T["__class__"] = [classify_row_base(a) for a in T.index]

# effective class with Role override
def effective_class(abbr: str, role_map: Dict[str, str]) -> str:
    base = str(T.loc[abbr, "__class__"])
    role = str(role_map.get(abbr, "")).strip().lower()
    fn = int(T.loc[abbr, "__epoxy_fn__"])
    name = _name_of(abbr).lower()

    # Hard override BDGE
    if abbr == "BDGE":
        return "epoxy_resin"

    # Role override
    if role:
        if role == "epoxy resin":
            return "epoxy_resin"
        if "reactive diluent" in role or role == "diluent":
            return "reactive_diluent"
        if "hardener" in role or "amine" in role:
            return "epoxy_hardener"
        if "isocyanate" in role:
            return "isocyanate"
        if "polyol" in role:
            return "polyol"
        if "extender" in role:
            return "extender"
        if "crosslinker" in role or "triol" in role:
            return "crosslinker"
        if "acid" in role or "anhydride" in role:
            return "acid_anhydride"
        if "vinyl" in role or "monomer" in role:
            return "vinyl_monomer"
        if "solvent" in role or "plasticizer" in role:
            return "solvent_plasticizer"
        if "silane" in role:
            return "silane"

    # Preserve DB epoxy classes if already set
    if base in ("epoxy_resin", "reactive_diluent", "epoxy_hardener"):
        return base

    # If epoxy-like, use fn split
    epoxy_like = (fn > 0) or any(k in name for k in ["epoxy", "oxirane", "glycid"])
    if epoxy_like:
        if fn >= 2:
            return "epoxy_resin"
        if fn == 1:
            return "reactive_diluent"
        return "reactive_diluent"

    return base

# =========================
# Epoxy pools (robust)
# =========================
DILUENT_NAME_RE = re.compile(
    r"(monoglycidyl|glycidyl\s+(ether|ester)|epichlorohydrin|alkyl\s+glycidyl|glycidyl\s+acrylate|glycidyl\s+methacrylate)",
    re.I
)
DILUENT_ABBR_RE = re.compile(r"(^|[^A-Z])(ECH|BGE|PGE|CGE|GMA|AGE|FGE|HGE|C12GE|C8GE)([^A-Z]|$)", re.I)

def build_epoxy_pools() -> Dict[str, List[str]]:
    role_map = _role_map_from_lib()
    T["__role__"] = [role_map.get(a, "") for a in T.index]
    T["__class_eff__"] = [effective_class(a, role_map) for a in T.index]

    candidates = []
    for ab in T.index:
        fn = int(T.loc[ab, "__epoxy_fn__"])
        nm = _name_of(ab).lower()
        role = str(role_map.get(ab, "")).lower()
        eff = str(T.loc[ab, "__class_eff__"])
        epoxy_like = (
            (fn > 0)
            or any(k in nm for k in ["epoxy", "oxirane", "glycid"])
            or ("epoxy" in role) or ("diluent" in role)
            or (eff in ("epoxy_resin", "reactive_diluent", "epoxy_hardener"))
        )
        if epoxy_like:
            candidates.append(ab)
    candidates = sorted(set(candidates))

    resins = sorted(set([ab for ab in candidates if int(T.loc[ab, "__epoxy_fn__"]) >= 2] + [ab for ab in T.index if T.loc[ab, "__class_eff__"] == "epoxy_resin"]))
    hardeners = sorted(set([ab for ab in T.index if T.loc[ab, "__class_eff__"] == "epoxy_hardener"]))

    # This is the key fix: diluents = union of three independent signals
    diluents = set()

    # (a) effective class says reactive_diluent
    diluents.update([ab for ab in T.index if T.loc[ab, "__class_eff__"] == "reactive_diluent"])

    # (b) epoxy_fn == 1
    diluents.update([ab for ab in candidates if int(T.loc[ab, "__epoxy_fn__"]) == 1])

    # (c) name/abbr token cues (even if fn==0 due to missing SMILES)
    for ab in T.index:
        nm = _name_of(ab)
        if DILUENT_NAME_RE.search(nm) or DILUENT_ABBR_RE.search(ab):
            diluents.add(ab)

    # Enforce BDGE rule
    if "BDGE" in diluents:
        diluents.remove("BDGE")
    if "BDGE" in T.index and "BDGE" not in resins:
        resins.append("BDGE")
        resins = sorted(set(resins))

    # Clean
    diluents = sorted([ab for ab in diluents if ab in T.index])
    resins = [ab for ab in resins if ab in T.index]
    hardeners = [ab for ab in hardeners if ab in T.index]
    candidates = [ab for ab in candidates if ab in T.index]

    return {"candidates": candidates, "resins": resins, "diluents": diluents, "hardeners": hardeners}

# =========================
# EEW/AHEW helpers
# =========================
def estimate_EEW_from_db_or_mw(abbr: str) -> float:
    eq = equiv_lookup(abbr)
    if eq and (eq.get("EEW_g_eq") is not None) and (not (isinstance(eq.get("EEW_g_eq"), float) and np.isnan(eq.get("EEW_g_eq")))):
        return float(eq["EEW_g_eq"])

    eew_col = None
    for c in T.columns:
        if "eew" in str(c).lower():
            eew_col = c
            break
    if eew_col:
        v = T.loc[abbr, eew_col]
        try:
            if not pd.isna(v):
                return float(v)
        except Exception:
            pass

    mw = _mw_of(abbr)
    fn = int(T.loc[abbr, "__epoxy_fn__"])
    if np.isfinite(mw) and fn > 0:
        return float(mw / fn)
    return float("nan")

# =========================
# UI widgets & plots
# =========================
def equiv_editor():
    with st.expander("📚 Library of equivalents (editable)", expanded=False):
        st.caption("Role can override classification. Add your missing diluents here as **Role = Reactive diluent**.")
        df = get_equiv_df()
        df_edit = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="equiv_editor")
        df_edit = df_edit.copy()
        if "Abbrev" in df_edit.columns:
            df_edit["Abbrev"] = df_edit["Abbrev"].map(_norm_abbr)
        st.session_state["equiv_lib"] = df_edit

def select_abbr(title: str, options: List[str], key: str):
    options = list(dict.fromkeys(options))
    if not options:
        st.warning(f"No options found for: {title}")
        st.session_state[key] = None
        return None
    widget_key = f"{key}__label"
    choice = st.selectbox(title, options, key=widget_key, format_func=label)
    st.session_state[key] = choice
    return choice

def show_props(abbr: str):
    if not abbr:
        return
    row = T.loc[abbr]
    cols_show = []
    for k in ["name", "cas", "smiles", "mw", "dD", "dP", "dH", "da", "sig"]:
        c = COL.get(k)
        if c and c in row.index:
            cols_show.append(c)

    out = {}
    for k in cols_show:
        v = row[k]
        if isinstance(v, (float, int, np.floating, np.integer)) and not pd.isna(v):
            if k in {COL["dD"], COL["dP"], COL["dH"], COL.get("da"), COL.get("sig")}:
                out[k] = f"{float(v):.2f}"
            elif k == COL.get("mw"):
                out[k] = f"{float(v):.2f}"
            else:
                out[k] = str(v)
        else:
            out[k] = "" if pd.isna(v) else str(v)

    out["__epoxy_fn__"] = str(int(T.loc[abbr, "__epoxy_fn__"]))
    out["__class__"] = str(T.loc[abbr, "__class__"])
    out["__class_eff__"] = str(T.loc[abbr, "__class_eff__"]) if "__class_eff__" in T.columns else ""
    ee = estimate_EEW_from_db_or_mw(abbr)
    out["EEW_est (g/eq)"] = "" if (not np.isfinite(ee)) else f"{ee:.1f}"
    st.write(pd.DataFrame.from_dict(out, orient="index", columns=["value"]))

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
# Tabs
# =========================
tab_form, tab_maps, tab_figs = st.tabs(["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"])

# =========================
# FORMULATOR TAB
# =========================
with tab_form:
    st.subheader("Formulator")
    equiv_editor()

    pools = build_epoxy_pools()
    ep_resins = pools["resins"]
    ep_dils_all = pools["diluents"]
    ep_hards = pools["hardeners"]
    EPOXY_CANDIDATES = pools["candidates"]

    st.markdown("### Epoxy stoichiometry (EEW/AHEW)")
    st.caption("Reactive diluent pool is robust: class OR epoxy_fn==1 OR name/abbr cues. BDGE is forced resin (fn=2).")

    c1, c2, c3 = st.columns(3)
    with c1:
        resin = select_abbr("Resin", ep_resins, key="ep_resin")
    with c2:
        hard = select_abbr("Hardener", ep_hards, key="ep_hard")
    with c3:
        use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

    fn_target = st.slider("Reactive diluent — epoxy functionality filter (estimated)", 1, 4, 1, step=1, key="ep_fn_target")
    ep_dils = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
    if use_dil and (not ep_dils):
        ep_dils = list(ep_dils_all)

    dil = None
    if use_dil:
        dil = select_abbr("Reactive diluent", ep_dils, key="ep_dil")

    # Auto-fill from equivalents library (optional)
    lock = st.checkbox("🔒 Auto-fill EEW/AHEW from equivalents library", value=True, key="lock_to_lib_epoxy")

    if lock and resin:
        eq = equiv_lookup(resin)
        if eq.get("EEW_g_eq") is not None and not (isinstance(eq.get("EEW_g_eq"), float) and np.isnan(eq.get("EEW_g_eq"))):
            st.session_state["ep_eew"] = float(eq["EEW_g_eq"])
    if lock and hard:
        eq = equiv_lookup(hard)
        if eq.get("AHEW_g_eq") is not None and not (isinstance(eq.get("AHEW_g_eq"), float) and np.isnan(eq.get("AHEW_g_eq"))):
            st.session_state["ep_ahew"] = float(eq["AHEW_g_eq"])

    if use_dil and dil:
        ee = estimate_EEW_from_db_or_mw(dil)
        if np.isfinite(ee):
            st.session_state["ep_dil_eew"] = float(ee)

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
        ratio = st.number_input("Epoxy/Active-H eq ratio", min_value=0.5, max_value=2.0,
                                value=1.0, step=0.01, key="ep_ratio")

    mass_mode = st.radio("Mass target mode",
                         ["Total mass (includes hardener)", "Resin-side mass (resin+diluent)"],
                         horizontal=True, key="ep_mass_mode")
    target = st.number_input("Target mass (g)", min_value=1.0, value=100.0, step=1.0, key="ep_target")

    dil_frac = 0.0
    EEW_dil = np.inf
    if use_dil and dil:
        dil_frac = st.slider("Diluente mass fraction (in resin-side)", 0.0, 0.8, 0.2, step=0.01, key="ep_dil_frac")
        EEW_dil = st.number_input("Diluente EEW (g/eq)", min_value=50.0, max_value=3000.0,
                                  value=float(st.session_state.get("ep_dil_eew", 150.0)),
                                  step=1.0, key="ep_dil_eew")

    term = (1.0 - dil_frac) / max(EEW_res, 1e-12) + (dil_frac / max(EEW_dil, 1e-12) if (use_dil and dil) else 0.0)
    g = (AHEW / max(ratio, 1e-12)) * term

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

        st.markdown("#### Selected properties")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Resin**"); show_props(resin)
        with c2:
            st.write("**Hardener**"); show_props(hard)
        with c3:
            st.write("**Diluent**"); show_props(dil) if (use_dil and dil) else None

    st.markdown("#### Current epoxy pools (for sanity)")
    st.write("**Epoxy resins:**", ep_resins[:200])
    st.write("**Reactive diluents:**", ep_dils_all[:200])
    st.write("**Hardeners:**", ep_hards[:200])

# =========================
# HEATMAPS TAB
# =========================
with tab_maps:
    st.subheader("Heatmaps (Epoxy only in this trimmed build)")
    metric = st.radio("Heatmap metric", ["Ra", "Δδa", "Π"], horizontal=True, key="hm_metric")

    pools = build_epoxy_pools()
    ep_resins = pools["resins"]
    ep_dils_all = pools["diluents"]
    ep_hards = pools["hardeners"]

    da_max_hm = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max_hm = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

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
                else:
                    mat.loc[r, c] = Pi(da, ds, da_max_hm, ds_max_hm)
        return mat

    cbar_label = (
        "Ra (MPa$^{1/2}$)" if metric == "Ra" else
        "Δδa (MPa$^{1/2}$)" if metric == "Δδa" else
        "Π (dimensionless)"
    )

    st.caption("Epoxy resins × hardeners (heatmap)")
    res_sel = st.multiselect("Resins (cols)", ep_resins, default=ep_resins[:12], format_func=label, key="hm_ep_resins")
    hard_sel = st.multiselect("Hardeners (rows)", ep_hards, default=ep_hards[:18], format_func=label, key="hm_ep_hards")
    if hard_sel and res_sel:
        mat = build_matrix(hard_sel, res_sel)
        st.pyplot(plot_heatmap(mat, f"EPOXY — Hardeners × Resins ({metric})", cbar_label))

    st.caption("Reactive diluents × resins (heatmap)")
    fn_target = st.slider("Reactive diluents — epoxy functionality (heatmap)", 1, 4, 1, step=1, key="hm_ep_fn")
    dil_pool = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
    if not dil_pool:
        dil_pool = list(ep_dils_all)

    dil_sel = st.multiselect("Reactive diluents (rows)", dil_pool, default=dil_pool[:18], format_func=label, key="hm_ep_dils")
    if dil_sel and res_sel:
        mat2 = build_matrix(dil_sel, res_sel)
        st.pyplot(plot_heatmap(mat2, f"EPOXY — Reactive diluents × Resins ({metric})", cbar_label))

# =========================
# FIGURES TAB
# =========================
with tab_figs:
    st.subheader("FIGURES — Ra vs Δδa and Ra vs Π (Epoxy)")
    st.caption("Axes follow the manuscript: Ra on x-axis; Δδa or Π on y-axis.")

    pools = build_epoxy_pools()
    ep_resins = pools["resins"]
    ep_dils_all = pools["diluents"]
    ep_hards = pools["hardeners"]

    da_max_f = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max_f = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    def make_pairs(A_list, B_list, tag):
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
                pi = Pi(da, ds, da_max_f, ds_max_f)
                rows.append({"A": a_id, "B": b_id, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi, "pair": tag})
        return pd.DataFrame(rows)

    res_sel = st.multiselect("Resins", ep_resins, default=ep_resins[:8], format_func=label, key="fig_ep_resins")
    hard_sel = st.multiselect("Hardeners", ep_hards, default=ep_hards[:12], format_func=label, key="fig_ep_hards")

    fn_target = st.slider("Reactive diluents — epoxy functionality (figures)", 1, 4, 1, step=1, key="fig_ep_fn")
    dils = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
    if not dils:
        dils = list(ep_dils_all)
    dil_sel = st.multiselect("Reactive diluents", dils, default=dils[:12], format_func=label, key="fig_ep_dils")

    df1 = make_pairs(hard_sel, res_sel, tag="hard×resin") if (hard_sel and res_sel) else pd.DataFrame()
    df2 = make_pairs(dil_sel, res_sel, tag="dil×resin") if (dil_sel and res_sel) else pd.DataFrame()
    dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()

    if dfp.empty:
        st.warning("No pairs available (check selections).")
    else:
        st.pyplot(plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)"))
        st.pyplot(plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)"))
        st.markdown("#### Data used in figures")
        st.dataframe(dfp.sort_values("Ra").round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3}), use_container_width=True, hide_index=True)
