# DiPEVa Formulator — UPDATED FULL SCRIPT
# Fixes:
# 1) Robust epoxy functionality estimation (mono vs di/multi) so diluents beyond ECH appear.
# 2) Role override layer (equiv library) to force Resin / Reactive diluent / Hardener.
# 3) Unified epoxy pool builder used consistently in Formulator + Heatmaps + Figures.
# 4) BDGE explicitly treated as DIFUNCTIONAL epoxy (epoxy resin, NOT reactive diluent).
#
# Place this as app.py in the same folder as Database_final.xlsx

import re
from pathlib import Path
import math
from typing import Optional, Callable, Dict, List, Tuple

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
for k in ["dD", "dP", "dH", "sig", "da", "mw"]:
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

    # Difunctional alcohols (Extenders)
    {"Abbrev": "EG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1806.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "DEG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1058.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TEG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 748.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,3-BDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1246.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,4-BDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1246.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "1,6-HDO", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 967.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "MPG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1475.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "NPG", "Role": "Extender", "NCO_%": None, "OH_mgKOH_g": 1197.0, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Triols / Crosslinkers
    {"Abbrev": "GLY", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1828.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TMP", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1253.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "SOR", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1217.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "PER", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1760.0, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Epoxy resins / hardeners (EEW/AHEW)
    {"Abbrev": "DGEBA", "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 185.0, "AHEW_g_eq": None},

    # IMPORTANT: BDGE is DIFUNCTIONAL epoxy (di-glycidyl ether) => epoxy resin (aliphatic epoxy resin)
    {"Abbrev": "BDGE",  "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.0, "AHEW_g_eq": None},

    # Some common mono-epoxy reactive diluents (OPTIONAL—edit in UI as needed)
    # If you have them in your DB, setting Role guarantees they appear even if SMILES/name is messy.
    {"Abbrev": "ECH",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.5, "AHEW_g_eq": None},
    {"Abbrev": "GMA",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "PGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "CGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "BGE",   "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},

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
# LABELS / HELPERS
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
# FALLBACK ESTIMATORS
# =========================
_MW_RE = re.compile(r"(?:^|[^0-9])([0-9]{2,5})(?:$|[^0-9])")

def estimate_OH_number_from_abbr(abbr: str) -> Optional[float]:
    if not abbr:
        return None
    a = str(abbr).upper().strip()
    triol_hints = ["TRIOL", "GLY", "GLYC", "TMP", "SOR", "PER", "GLYCER", "SORBIT", "PENTAERYTHR"]
    f = 3 if any(h in a for h in triol_hints) else 2

    mm = None
    m = re.search(r"(PEG|PPG|PTMEG|PCL|PCDL|PBD|PDMS)\s*([0-9]{2,5})", a)
    if m:
        mm = float(m.group(2))
    else:
        m2 = _MW_RE.search(a)
        if m2:
            mm = float(m2.group(1))
    if not mm or mm <= 0:
        return None
    return float(56100.0 * f / mm)

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
# EPOXY: ring counting + functionality (FIXED)
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
    """Robust epoxy functionality:
    Priority: SMILES ring counting; otherwise name-based multiplicity rules.
    Key fixes:
      - Monoglycidyl / glycidyl ether/ester/methacrylate defaults to 1 (NOT 2).
      - Explicit di/tri/tetra keywords respected.
      - BDGE forced to 2 (di-glycidyl ether; aliphatic epoxy resin).
      - ECH forced to 1.
      - If epoxy-like but ambiguous -> default 1 (safer for diluent discovery; can be overridden by Role).
    """
    ab = str(abbr).strip().upper()
    if ab not in T.index:
        return 0

    # Hard overrides for known abbreviations
    if ab == "BDGE":
        return 2
    if ab == "ECH":
        return 1
    if ab in {"DGEBA", "DGEBF", "EPN", "ECN"}:
        return 2

    # 1) SMILES ring counting
    smi = _smiles_of(ab)
    if smi:
        cnt = _count_epoxide_rings(smi)
        if cnt > 0:
            return int(min(cnt, 8))

    # 2) Name-based
    name = _name_of(ab).lower()

    epoxy_like = any(k in name for k in ["oxirane", "epoxy", "epoxide", "glycid"])
    if not epoxy_like and not any(k in ab for k in ["GMA", "GDE", "DGE", "ECH", "BDGE"]):
        return 0

    # explicit multiplicity keywords
    if any(k in name for k in ["tetraglycidyl", "tetra-glycidyl"]):
        return 4
    if any(k in name for k in ["triglycidyl", "tri-glycidyl"]):
        return 3
    if any(k in name for k in ["diglycidyl", "di-glycidyl"]):
        return 2

    # very common monofunctional reactive diluents:
    mono_signals = [
        "monoglycidyl", "mono glycidyl",
        "glycidyl ether", "glycidyl ester",
        "glycidyl methacrylate", "glycidyl acrylate"
    ]
    if any(k in name for k in mono_signals):
        return 1

    # resin-ish cues (usually 2+)
    resin_cues = ["bisphenol", "novolac", "epoxy resin", "resin"]
    if any(k in name for k in resin_cues):
        return 2

    # if epoxy-like but ambiguous:
    return 1

def estimate_EEW_from_db_or_mw(abbr: str) -> float:
    eq = equiv_lookup(abbr)
    if eq and (eq.get("EEW_g_eq") is not None) and (not (isinstance(eq.get("EEW_g_eq"), float) and np.isnan(eq.get("EEW_g_eq")))):
        return float(eq["EEW_g_eq"])

    # search EEW column in DB if exists
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
    fn = int(T.loc[abbr, "__epoxy_fn__"]) if "__epoxy_fn__" in T.columns else 0
    if np.isfinite(mw) and fn > 0:
        return float(mw / fn)
    return float("nan")

# =========================
# CLASSIFICATION (base)
# =========================
def classify_row_base(abbr: str) -> str:
    name = _name_of(abbr).lower()
    a = str(abbr).lower()

    # PU family
    if (
        "isocyanate" in name
        or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", a)
        or re.search(r"\bmdi\b|\btdi\b|\bhdi\b|\bipdi\b|\bpmdi\b|\bh12mdi\b", name)
    ):
        return "isocyanate"

    # Triols
    if (
        "triol" in name
        or "glycer" in name
        or "trimethylol" in name
        or "sorbit" in name
        or "pentaerythrit" in name
        or re.search(r"\btmp\b|\bgly\b|\bper\b|\bsor\b", a)
    ):
        return "crosslinker"

    # Chain extenders (diols)
    if (
        "butanediol" in name
        or "hexanediol" in name
        or "propanediol" in name
        or "ethylene glycol" in name
        or "diethylene glycol" in name
        or "triethylene glycol" in name
        or re.search(r"\beg\b|\bdeg\b|\bteg\b", a)
        or re.search(r"\b1,?3-?bdo\b|\b1,?4-?bdo\b|\b1,?6-?hdo\b", a)
        or re.search(r"\bmpg\b|\bpg\b|\bdpg\b|\bnpg\b", a)
    ):
        return "extender"

    # Polyols
    if (
        "polyol" in name
        or "polyether" in name
        or "polyester" in name
        or "polycarbonate" in name
        or ("diol" in name and "butanediol" not in name and "hexanediol" not in name and "propanediol" not in name)
        or re.search(r"\bpeg\d+|\bppg\d+|\bptmeg\d*|\bpcl\d+|\bpcdl\d+|\bpbd\d*", a)
    ):
        return "polyol"

    # Polyester acids/anhydrides
    acid_abbr = {"aa","sa","pa","ga","ma","la","oa","fa","ipa","tpa","sua","sea","mah","hhpa","mhhpa"}
    if (
        "acid" in name
        or "diacid" in name
        or "dicarbox" in name
        or "anhydride" in name
        or a in acid_abbr
        or re.search(r"\b(phthalic|isophthalic|terephthalic|succinic|sebacic|adipic|glutaric|maleic|fumaric)\b", name)
        or re.search(r"\bmah\b|\bhhpa\b|\bmhhpa\b", a)
    ):
        return "acid_anhydride"

    # Hardeners
    if (
        "amine" in name
        or "diamin" in name
        or "polyamine" in name
        or "hardener" in name
        or "curing" in name
        or re.search(r"\bdeta\b|\bteta\b|\bddm\b|\bdds\b|\bdicy\b|\bipda\b|\bjeffamine\b", a)
    ):
        return "epoxy_hardener"

    # Vinyl monomers
    if (
        "styrene" in name
        or "acrylonitrile" in name
        or "methacrylate" in name
        or "acrylate" in name
        or "vinyl acetate" in name
        or "alpha-methylstyrene" in name
        or re.search(r"\bst\b|\ban\b|\bmma\b|\bva\b|\bvac\b|\bfn\b|\ba-ms\b|\bα-ms\b", a)
    ):
        return "vinyl_monomer"

    # Solvents/plasticizers
    if (
        "alcohol" in name
        or "solvent" in name
        or "plasticizer" in name
        or "phthalate" in name
        or "phosphate" in name
        or "adipate" in name
        or "benzoate" in name
        or "citrate" in name
        or re.search(r"\b(acetone|mek|mibk|thf|dmf|dmso|nmp|dmac|toluene|xylene|ethyl acetate|butyl acetate|heptane|hexane|cyclohexane|isopropanol|ethanol|methanol|butanol|dioxane|chloroform|dichloromethane)\b", name)
        or re.search(r"\b(etoh|meoh|ipoh|ipa|nproh|bnoh|thf|dmf|dmso|nmp|dmac|tol|xyl|ea|ba|mek|mibk|dcm|chl|dbp|dehp|dinp)\b", a)
    ):
        return "solvent_plasticizer"

    # Silanes
    if (
        "silane" in name
        or re.search(r"\bteos\b|\bvtms\b|\baptes\b|\bmptes\b|\bgpts\b|\bgptms\b", a)
    ):
        return "silane"

    return "other"

# =========================
# PRECOMPUTE epoxy_fn, Role override, and effective epoxy class
# =========================
def _role_map_from_lib() -> Dict[str, str]:
    eqdf = get_equiv_df()
    if eqdf.empty or "Abbrev" not in eqdf.columns:
        return {}
    if "Role" not in eqdf.columns:
        return {}
    m = {}
    for ab, role in zip(eqdf["Abbrev"].astype(str), eqdf["Role"].astype(str)):
        abn = _norm_abbr(ab)
        r = str(role).strip().lower()
        if abn:
            m[abn] = r
    return m

# epoxy functionality
if "__epoxy_fn__" not in T.columns:
    try:
        T["__epoxy_fn__"] = [estimate_epoxy_function(a) for a in T.index]
        T["__epoxy_fn__"] = pd.to_numeric(T["__epoxy_fn__"], errors="coerce").fillna(0).astype(int)
    except Exception:
        T["__epoxy_fn__"] = 0

# base class
T["__class__"] = [classify_row_base(i) for i in T.index]

# role override + effective class for epoxy items
role_map = _role_map_from_lib()
T["__role__"] = [role_map.get(a, "") for a in T.index]

def _effective_class(ab: str) -> str:
    base = str(T.loc[ab, "__class__"])
    role = str(T.loc[ab, "__role__"]).strip().lower()
    fn = int(T.loc[ab, "__epoxy_fn__"]) if "__epoxy_fn__" in T.columns else 0

    # Role overrides (highest priority)
    if role:
        if "isocyanate" in role:
            return "isocyanate"
        if "extender" in role:
            return "extender"
        if "crosslinker" in role or "triol" in role:
            return "crosslinker"
        if "polyol" in role:
            return "polyol"
        if role == "epoxy resin":
            return "epoxy_resin"
        if "reactive diluent" in role or role == "diluent":
            return "reactive_diluent"
        if "hardener" in role or "amine" in role:
            return "epoxy_hardener"
        if "silane" in role:
            return "silane"
        if "solvent" in role or "plasticizer" in role:
            return "solvent_plasticizer"
        if "vinyl" in role or "monomer" in role:
            return "vinyl_monomer"
        if "acid" in role or "anhydride" in role:
            return "acid_anhydride"

    # Epoxy inference if epoxy-like (fn>0 or name hints)
    name = _name_of(ab).lower()
    epoxy_like = (fn > 0) or any(k in name for k in ["epoxy", "oxirane", "glycid"])
    if epoxy_like:
        # BDGE is resin, already fn=2 above; keep rule:
        if fn >= 2:
            return "epoxy_resin"
        if fn == 1:
            return "reactive_diluent"
        # If unknown, prefer base if it already says something
        if base in ("epoxy_hardener", "solvent_plasticizer", "vinyl_monomer"):
            return base
        # fallback
        return "reactive_diluent"

    return base

T["__class_eff__"] = [_effective_class(a) for a in T.index]

def list_by_class(cls: str) -> List[str]:
    opts = [i for i in T.index if str(T.loc[i, "__class_eff__"]) == cls]
    return list(dict.fromkeys(opts))

# =========================
# EPOXY POOLS (single source of truth)
# =========================
def build_epoxy_pools() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns:
      EPOXY_CANDIDATES, ep_resins, ep_dils, ep_hards
    """
    role_map_local = _role_map_from_lib()

    # epoxy candidates: anything epoxy-like by fn/name OR by role OR by effective class
    candidates = []
    for ab in T.index:
        fn = int(T.loc[ab, "__epoxy_fn__"]) if "__epoxy_fn__" in T.columns else 0
        name = _name_of(ab).lower()
        role = role_map_local.get(ab, "")
        eff = str(T.loc[ab, "__class_eff__"])
        epoxy_like = (fn > 0) or any(k in name for k in ["epoxy", "oxirane", "glycid"]) or ("epoxy" in role) or ("diluent" in role)
        if epoxy_like or eff in ("epoxy_resin", "reactive_diluent", "epoxy_hardener"):
            candidates.append(ab)

    candidates = sorted(set(candidates))

    ep_resins = sorted(set([ab for ab in candidates if int(T.loc[ab, "__epoxy_fn__"]) >= 2] + list_by_class("epoxy_resin")))
    ep_dils   = sorted(set([ab for ab in candidates if int(T.loc[ab, "__epoxy_fn__"]) == 1] + list_by_class("reactive_diluent")))
    ep_hards  = sorted(set(list_by_class("epoxy_hardener")))

    # Safety: BDGE never in diluents
    ep_dils = [ab for ab in ep_dils if ab != "BDGE"]
    # Ensure BDGE in resins if present
    if "BDGE" in T.index and "BDGE" not in ep_resins:
        ep_resins.append("BDGE")
        ep_resins = sorted(set(ep_resins))

    # keep only in DB index
    ep_resins = [e for e in ep_resins if e in T.index]
    ep_dils   = [e for e in ep_dils if e in T.index]
    ep_hards  = [e for e in ep_hards if e in T.index]
    candidates = [e for e in candidates if e in T.index]

    return candidates, ep_resins, ep_dils, ep_hards

# =========================
# UI helpers
# =========================
def select_abbr(title: str, options: List[str], key: str):
    if not options:
        st.warning(f"No options found for: {title}")
        st.session_state[key] = None
        return None

    options = list(dict.fromkeys(options))
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

    out["__class_eff__"] = str(T.loc[abbr, "__class_eff__"])
    out["__role__"] = str(T.loc[abbr, "__role__"])
    out["epoxy_function_est"] = str(int(T.loc[abbr, "__epoxy_fn__"]))
    ee = estimate_EEW_from_db_or_mw(abbr)
    out["EEW_est (g/eq)"] = "" if (not np.isfinite(ee)) else f"{ee:.1f}"
    st.write(pd.DataFrame.from_dict(out, orient="index", columns=["value"]))

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
# EQUIV LIB EDITOR
# =========================
def equiv_editor():
    with st.expander("📚 Library of equivalents (editable)", expanded=False):
        st.caption("Matching is done by **Abbrev** (case-insensitive). Role can override classification.")
        df = get_equiv_df()
        df_edit = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="equiv_editor",
        )
        df_edit = df_edit.copy()
        if "Abbrev" in df_edit.columns:
            df_edit["Abbrev"] = df_edit["Abbrev"].map(_norm_abbr)
        st.session_state["equiv_lib"] = df_edit

def sync_from_library(selection_key: str, value_key_map: dict, enabled: bool, fallback: Optional[Callable] = None):
    if not enabled:
        return
    abbr = st.session_state.get(selection_key)
    if not abbr:
        return
    eq = equiv_lookup(abbr)
    if not eq and fallback is not None:
        try:
            eq = fallback(abbr) or {}
        except Exception:
            eq = {}
    if not eq:
        return
    for col_name, state_key in value_key_map.items():
        v = eq.get(col_name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        st.session_state[state_key] = float(v)

# =========================
# UI TABS
# =========================
tab_form, tab_maps, tab_figs, tab_debug = st.tabs(
    ["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π", "DEBUG"]
)

# -------------------------
# FORMULATOR
# -------------------------
with tab_form:
    st.subheader("Formulator")
    equiv_editor()

    # Refresh role_map + effective classes after edits (important!)
    role_map = _role_map_from_lib()
    T["__role__"] = [role_map.get(a, "") for a in T.index]
    T["__class_eff__"] = [_effective_class(a) for a in T.index]

    mode = st.selectbox(
        "Mode",
        ["Pair screening", "PU stoichiometry (NCO/OH)", "Epoxy stoichiometry (EEW/AHEW)", "Ranking (best combinations)"],
        key="mode_select",
    )

    # Candidate pools
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")

    # Epoxy pools (single source of truth)
    EPOXY_CANDIDATES, ep_resins, ep_dils_all, ep_hards = build_epoxy_pools()

    # global maxima for Π normalization
    da_max = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

    lock = st.checkbox("🔒 Auto-fill stoichiometry inputs from the equivalents library", value=True, key="lock_to_lib")

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
        st.caption("Uses %NCO and OH number. Inputs auto-update when values exist in the equivalents library.")

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

        sync_from_library("pu_iso", {"NCO_%": "pu_nco"}, lock)
        sync_from_library("pu_pol", {"OH_mgKOH_g": "pu_oh_pol"}, lock, fallback=lambda ab: {"OH_mgKOH_g": estimate_OH_number_from_abbr(ab)})
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

        ext_frac = 0.0
        xl_frac = 0.0
        if use_ext and ext:
            ext_frac = st.slider("Extender mass fraction (in polyol-side)", 0.0, 0.8, 0.2, step=0.01, key="pu_ext_frac")
        if use_xl and xl:
            max_xl = max(0.0, 0.9 - ext_frac)
            xl_frac = st.slider("Crosslinker/triol mass fraction (in polyol-side)", 0.0, float(max_xl), min(0.1, float(max_xl)), step=0.01, key="pu_xl_frac")
        pol_frac = max(0.0, 1.0 - ext_frac - xl_frac)

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

    elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.markdown("### Epoxy stoichiometry (EEW/AHEW)")
        st.caption("Reactive diluent list is built from epoxy_fn==1 OR Role=Reactive diluent. BDGE stays resin (fn=2).")

        c1, c2, c3 = st.columns(3)
        with c1:
            resin = select_abbr("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = select_abbr("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

        # Filter diluents by fn (default: mono epoxies = 1)
        fn_target = st.slider("Reactive diluent — epoxy functionality filter (estimated)", 1, 4, 1, step=1, key="ep_fn_target")
        ep_dils = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        # if none, show all available diluents
        if use_dil and (not ep_dils):
            ep_dils = list(ep_dils_all)

        dil = None
        if use_dil:
            dil = select_abbr("Reactive diluent", ep_dils, key="ep_dil")

        sync_from_library("ep_resin", {"EEW_g_eq": "ep_eew"}, lock)
        sync_from_library("ep_hard", {"AHEW_g_eq": "ep_ahew"}, lock)

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
            ratio = st.number_input("Epoxy/Active-H eq ratio", min_value=0.5, max_value=2.0, value=1.0, step=0.01, key="ep_ratio")

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

    else:
        st.markdown("### Ranking (best combinations)")
        st.caption("Ranks candidates by **Score_%** combining Ra + Δδa + Π (equal weighting, normalized within the candidate set).")

        family = st.selectbox("Family", ["PU", "Epoxy"], key="rk_family")

        if family == "Epoxy":
            resin = select_abbr("Resin (fixed)", ep_resins, key="rk_ep_resin")
            include_dil = st.checkbox("Include reactive diluents in ranking", value=True, key="rk_ep_dil")
            fn_target = st.slider("Reactive diluent epoxy functionality (ranking)", 1, 4, 1, step=1, key="rk_ep_fn")
            dil_cand = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
            cand = list(ep_hards) + (dil_cand if include_dil else [])
            cand = [c for c in cand if c in T.index]
            if resin and cand:
                rows = []
                for c in cand:
                    if c == resin:
                        continue
                    a = T.loc[c]; b = T.loc[resin]
                    ra = Ra(a, b); da = delta_a(a, b); ds = delta_sigmaL(a, b)
                    pi = Pi(da, ds, da_max, ds_max)
                    rows.append({"candidate": c, "type": T.loc[c, "__class_eff__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
                dfc = pd.DataFrame(rows)
                raM = float(dfc["Ra"].max()); daM = float(dfc["Δδa"].max()); piM = float(dfc["Π"].max())
                dfc["Score_%"] = [score_percent(r, d, p, raM, daM, piM) for r, d, p in zip(dfc["Ra"], dfc["Δδa"], dfc["Π"])]
                dfc = dfc.sort_values("Score_%", ascending=False)
                st.dataframe(dfc.round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3, "Score_%": 1}), use_container_width=True, hide_index=True)

        else:
            iso = select_abbr("Isocyanate (fixed)", isos, key="rk_pu_iso")
            include_ext = st.checkbox("Include extenders and crosslinkers in ranking", value=True, key="rk_pu_ext")
            cand = polyols + ((extenders + crosslinkers) if include_ext else [])
            cand = [c for c in cand if c in T.index]
            if iso and cand:
                rows = []
                for c in cand:
                    if c == iso:
                        continue
                    a = T.loc[c]; b = T.loc[iso]
                    ra = Ra(a, b); da = delta_a(a, b); ds = delta_sigmaL(a, b)
                    pi = Pi(da, ds, da_max, ds_max)
                    rows.append({"candidate": c, "type": T.loc[c, "__class_eff__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
                dfc = pd.DataFrame(rows)
                raM = float(dfc["Ra"].max()); daM = float(dfc["Δδa"].max()); piM = float(dfc["Π"].max())
                dfc["Score_%"] = [score_percent(r, d, p, raM, daM, piM) for r, d, p in zip(dfc["Ra"], dfc["Δδa"], dfc["Π"])]
                dfc = dfc.sort_values("Score_%", ascending=False)
                st.dataframe(dfc.round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3, "Score_%": 1}), use_container_width=True, hide_index=True)

# -------------------------
# HEATMAPS
# -------------------------
with tab_maps:
    st.subheader("Heatmaps (grouped by purpose)")

    # refresh after editor
    role_map = _role_map_from_lib()
    T["__role__"] = [role_map.get(a, "") for a in T.index]
    T["__class_eff__"] = [_effective_class(a) for a in T.index]

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

    acids = list_by_class("acid_anhydride")
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")
    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    sil  = list_by_class("silane")

    # epoxy pools (consistent)
    EPOXY_CANDIDATES, ep_resins, ep_dils_all, ep_hards = build_epoxy_pools()

    # Normalization constants for Π
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
                else:
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

        fn_target = st.slider("Reactive diluents — epoxy functionality (heatmap)", 1, 4, 1, step=1, key="hm_ep_fn")
        ep_dils = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        if not ep_dils:
            ep_dils = list(ep_dils_all)

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
            default=_default(solv, 18),
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

    # refresh after editor
    role_map = _role_map_from_lib()
    T["__role__"] = [role_map.get(a, "") for a in T.index]
    T["__class_eff__"] = [_effective_class(a) for a in T.index]

    fam = st.selectbox("Family", ["Polyesters", "Polyurethanes", "Epoxies", "Vinyls", "Silanes"], key="fig_family")

    use_all_fig = st.checkbox("Use full lists (may be slower)", value=True, key="fig_all")
    def _def(opts, n):
        return opts if use_all_fig else opts[:n]

    da_max_f = float(np.nanmax(T[COL["da"]])) if (COL.get("da") and COL["da"] in T.columns) else 1.0
    ds_max_f = float(np.nanmax(T[COL["sig"]])) if (COL.get("sig") and COL["sig"] in T.columns) else 1.0

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
                pi = Pi(da, ds, da_max_f, ds_max_f)
                rows.append({"A": a_id, "B": b_id, "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi, hue_name: f"{tagA}×{tagB}"})
        return pd.DataFrame(rows)

    # epoxy pools for figures
    EPOXY_CANDIDATES, ep_resins, ep_dils_all, ep_hards = build_epoxy_pools()

    if fam == "Polyesters":
        A = st.multiselect("Polyols", sorted(set(list_by_class("polyol") + list_by_class("extender") + list_by_class("crosslinker"))),
                           default=_def(sorted(set(list_by_class("polyol") + list_by_class("extender") + list_by_class("crosslinker"))), 10),
                           format_func=label)
        B = st.multiselect("Acids/anhydrides", list_by_class("acid_anhydride"), default=_def(list_by_class("acid_anhydride"), 10), format_func=label)
        dfp = make_pairs(A, B, tagA="polyol", tagB="acid")

    elif fam == "Polyurethanes":
        iso_sel = st.multiselect("Isocyanates", list_by_class("isocyanate"), default=_def(list_by_class("isocyanate"), 8), format_func=label)
        pol_sel = st.multiselect("Polyols", list_by_class("polyol"), default=_def(list_by_class("polyol"), 10), format_func=label)
        alc_sel = st.multiselect("Alcohols (extenders + crosslinkers)", sorted(set(list_by_class("extender") + list_by_class("crosslinker"))),
                                 default=_def(sorted(set(list_by_class("extender") + list_by_class("crosslinker"))), 10),
                                 format_func=label)
        df1 = make_pairs(pol_sel, iso_sel, tagA="polyol", tagB="iso")
        df2 = make_pairs([a for a in alc_sel if a in T.index], iso_sel, tagA="alc", tagB="iso")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()

    elif fam == "Epoxies":
        res_sel = st.multiselect("Resins", ep_resins, default=_def(ep_resins, 8), format_func=label)
        hard_sel = st.multiselect("Hardeners", ep_hards, default=_def(ep_hards, 12), format_func=label)

        fn_target = st.slider("Reactive diluents — epoxy functionality (figures)", 1, 4, 1, step=1, key="fig_ep_fn")
        dils = [a for a in ep_dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        if not dils:
            dils = list(ep_dils_all)

        dil_sel = st.multiselect("Reactive diluents", dils, default=_def(dils, 12), format_func=label)
        df1 = make_pairs(hard_sel, res_sel, tagA="hard", tagB="resin")
        df2 = make_pairs(dil_sel, res_sel, tagA="dil", tagB="resin")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()

    elif fam == "Vinyls":
        mon_sel = st.multiselect("Monomers", list_by_class("vinyl_monomer"), default=_def(list_by_class("vinyl_monomer"), 18), format_func=label)
        sol_sel = st.multiselect("Solvents/plasticizers", list_by_class("solvent_plasticizer"), default=_def(list_by_class("solvent_plasticizer"), 18), format_func=label)
        dfp = make_pairs(mon_sel, sol_sel, tagA="mon", tagB="solv")

    else:
        sil_sel = st.multiselect("Silanes", list_by_class("silane"), default=_def(list_by_class("silane"), 10), format_func=label)
        target_pool = []
        for ab in T.index:
            nm = _name_of(ab).lower()
            if "water" in nm or "alcohol" in nm:
                target_pool.append(ab)
        target_pool = sorted(set(target_pool + _def(list_by_class("vinyl_monomer"), 10) + _def(list_by_class("polyol"), 10)))
        tar_sel = st.multiselect("Targets", target_pool, default=_def(target_pool, 12), format_func=label)
        dfp = make_pairs(sil_sel, tar_sel, tagA="sil", tagB="target")

    if dfp.empty:
        st.warning("No pairs available (check selections).")
    else:
        st.pyplot(plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)"))
        st.pyplot(plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)"))

        st.markdown("#### Data used in figures")
        st.dataframe(dfp.sort_values("Ra").round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3}), use_container_width=True, hide_index=True)

# -------------------------
# DEBUG TAB (to confirm diluents show up)
# -------------------------
with tab_debug:
    st.subheader("Debug — Epoxy detection / pools")

    # refresh
    role_map = _role_map_from_lib()
    T["__role__"] = [role_map.get(a, "") for a in T.index]
    T["__class_eff__"] = [_effective_class(a) for a in T.index]

    EPOXY_CANDIDATES, ep_resins, ep_dils_all, ep_hards = build_epoxy_pools()

    st.write("**Epoxy resins (first 100):**", ep_resins[:100])
    st.write("**Reactive diluents (first 100):**", ep_dils_all[:100])
    st.write("**Hardeners (first 100):**", ep_hards[:100])

    dbg = pd.DataFrame({
        "Abbrev": list(T.index),
        "Name": [ _name_of(a) for a in T.index ],
        "Role_lib": [ str(T.loc[a, "__role__"]) for a in T.index ],
        "Class_eff": [ str(T.loc[a, "__class_eff__"]) for a in T.index ],
        "epoxy_fn": [ int(T.loc[a, "__epoxy_fn__"]) for a in T.index ],
        "SMILES": [ _smiles_of(a) for a in T.index ],
    })

    st.markdown("### All epoxy_fn == 1 (expected reactive diluents unless Role says otherwise)")
    st.dataframe(dbg[dbg["epoxy_fn"] == 1].sort_values(["Role_lib","Abbrev"]).head(500), use_container_width=True, hide_index=True)

    st.markdown("### 'glycidyl' in name but epoxy_fn == 0 (likely missing SMILES / naming issue)")
    miss = dbg[(dbg["epoxy_fn"] == 0) & (dbg["Name"].str.lower().str.contains("glycidyl", na=False))]
    st.dataframe(miss.head(500), use_container_width=True, hide_index=True)

    st.markdown("### BDGE sanity check (must be epoxy_fn=2 and Class_eff=epoxy_resin)")
    if "BDGE" in T.index:
        show_props("BDGE")
    else:
        st.info("BDGE not found in your database index.")
