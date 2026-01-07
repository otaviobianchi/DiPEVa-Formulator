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
