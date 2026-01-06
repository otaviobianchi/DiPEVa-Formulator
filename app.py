
import re
from pathlib import Path
import math
from typing import Optional, Callable

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
    {"Abbrev": "Polymeric MDI", "Role": "Isocyanate", "NCO_%": 31.5, "OH_mgKOH_g": None, "EEW_g_eq": None, "AHEW_g_eq": None},

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

    # Triols / Crosslinkers
    {"Abbrev": "GLY", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1828.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "TMP", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1253.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "SOR", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1217.0, "EEW_g_eq": None, "AHEW_g_eq": None},
    {"Abbrev": "PER", "Role": "Crosslinker/triol", "NCO_%": None, "OH_mgKOH_g": 1760.0, "EEW_g_eq": None, "AHEW_g_eq": None},

    # Epoxy resins / hardeners (EEW/AHEW)
    {"Abbrev": "DGEBA", "Role": "Epoxy resin", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 185.0, "AHEW_g_eq": None},
    {"Abbrev": "BDGE",  "Role": "Reactive diluent", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.0, "AHEW_g_eq": None},
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
# FALLBACK ESTIMATORS (when equivalents are missing in library)
# =========================
_MW_RE = re.compile(r"(?:^|[^0-9])([0-9]{2,5})(?:$|[^0-9])")

def estimate_OH_number_from_abbr(abbr: str) -> Optional[float]:
    """Best-effort OH number (mgKOH/g) estimator for polymeric polyols when the library lacks OH.

    Heuristic rules (transparent + editable in code):
    - If Abbrev encodes a nominal Mn (e.g., PEG200, PPG1000, PTMEG2000, PCL1000, PCDL1000), use that Mn.
    - Functionality defaults to 2 (diol). If the name/abbr hints triol, use 3.
    - OH# ≈ 56100 * f / Mn  (mgKOH/g), assuming all OH are primary and Mn in g/mol.

    Returns None if Mn cannot be inferred.
    """
    if not abbr:
        return None
    a = str(abbr).upper().strip()

    # common triol hints
    triol_hints = ["TRIOL", "GLY", "GLYC", "TMP", "SOR", "PER", "GLYCER", "SORBIT", "PENTAERYTHR"]
    f = 3 if any(h in a for h in triol_hints) else 2

    # Try to infer Mn from typical polymeric polyol abbreviations
    # Examples: PEG200, PEG1000, PPG400, PTMEG2000, PCL1000, PCDL1000
    mm = None
    m = re.search(r"(PEG|PPG|PTMEG|PCL|PCDL|PBD|PDMS)\s*([0-9]{2,5})", a)
    if m:
        mm = float(m.group(2))
    else:
        # fallback: any 2-5 digit number in abbr
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

# --- Epoxy function estimation (for reactive diluent filtering) ---
# We intentionally avoid RDKit to keep deployment simple.
# This is a *best-effort* estimator used only for UI filtering and fallback EEW.
_EPOXY_HINT_RE = re.compile(r"epoxy|oxirane|glycidyl|epichlorohydrin|dige|dige|dge", re.I)

def _count_epoxide_rings(smiles: str) -> int:
    """Count oxirane rings in SMILES with conservative regex.

    Handles common variants:
      - C1OC1, C2OC2, ...
      - C1CO1, C2CO2, ...
      - O1CC1, O2CC2, ...
    """
    if not smiles:
        return 0
    s = smiles.replace(" ", "")
    n = 0
    # ring digit backreference (1..9) – good enough for typical monomer SMILES
    n += len(re.findall(r"C(\d)OC\1", s))
    n += len(re.findall(r"C(\d)CO\1", s))
    n += len(re.findall(r"O(\d)CC\1", s))
    # some tool exports may contain "OC1OC1" fragments
    n += len(re.findall(r"OC(\d)OC\1", s))
    return int(n)

def estimate_epoxy_function(abbr: str) -> int:
    """Estimate epoxy functionality (number of epoxide groups).

    Priority:
      1) If SMILES/BigSMILES exists, count epoxide rings by common substructures.
      2) Otherwise, fall back to name/abbrev heuristics.

    Returns:
      0 if no epoxy detected, otherwise a positive integer (capped at 4).
    """
    ab = str(abbr).strip().upper()
    if ab not in T.index:
        return 0

    # --- 1) SMILES-based (best effort)
    smi_col = COL.get("smiles")
    smi = None
    if smi_col and smi_col in T.columns:
        v = T.loc[ab, smi_col]
        if isinstance(v, str) and v.strip() and v.strip().lower() not in ("nan", "none"):
            smi = v.strip()

    if smi:
        s = smi
        # Common epoxide ring patterns in SMILES:
        #   - C1OC1 (oxirane)
        #   - C1CO1 / O1CC1 (equivalent orderings)
        # Also catch substituted patterns like O1C(C)C1 etc.
        pats = [
            r"C1OC1",
            r"C1CO1",
            r"O1CC1",
            r"O1C[^\n]{0,8}?C1",  # short ring fragment
        ]
        cnt = 0
        for p in pats[:3]:
            cnt = max(cnt, len(re.findall(p, s)))
        # If the strict patterns fail, try the looser one once
        if cnt == 0 and re.search(pats[3], s):
            cnt = 1
        if cnt > 0:
            return int(min(cnt, 4))

    # --- 2) Heuristics (name/abbr)
    name = _name_of(ab).lower()

    # strong cues
    if any(k in name for k in ["oxirane", "epoxy", "epoxide", "glycidyl"]):
        # mono-glycidyl compounds are typically monofunctional
        if any(k in name for k in ["mono", "monoglycidyl", "glycidyl ether", "glycidyl ester"]):
            return 1

        # Strong cues for multi-functional epoxies (resins) even when SMILES is missing
        if ("cycloaliphatic" in name) or ("aliphatic epoxy" in name) or ("epoxy cycloaliphatic" in name):
            return 2
        if "triglycidyl" in name or "glycidyl" in name and "tri" in name:
            return 3
        if "tetraglycidyl" in name or "tetra" in name and "glycidyl" in name:
            return 4


    # Abbrev / resin heuristics
    if ab in {"DGEBA", "DGEBF", "EPN", "ECN"}:
        return 2
    if "DGE" in ab or "BDGE" in ab or "GDE" in ab:
        # Many glycidyl ethers are difunctional, but keep conservative:
        return 1 if "MONO" in ab else 2

    # If it looks like a reactive diluent by name, assume 1
    if "diluent" in name:
        return 1

    return 0


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
# CLASSIFICATION
# =========================

# =========================
# PRECOMPUTE EPOXY FUNCTIONALITY (for reactive diluent detection)
# =========================
# Must be computed BEFORE classify_row() uses it.
if "__epoxy_fn__" not in T.columns:
    try:
        T["__epoxy_fn__"] = [estimate_epoxy_function(a) for a in T.index]
        T["__epoxy_fn__"] = pd.to_numeric(T["__epoxy_fn__"], errors="coerce").fillna(0).astype(int)
    except Exception:
        # Fail-safe: keep app running even if SMILES are missing/unparseable
        T["__epoxy_fn__"] = 0

# Precomputed epoxy candidate pools (used for reactive diluents/resins filters)
EPOXY_CANDIDATES = []
if "__epoxy_fn__" in T.columns:
    try:
        EPOXY_CANDIDATES = [a for a in T.index if int(T.loc[a, "__epoxy_fn__"]) > 0]
    except Exception:
        EPOXY_CANDIDATES = []
EPOXY_MONO = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) == 1] if EPOXY_CANDIDATES else []
EPOXY_MULTI = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) >= 2] if EPOXY_CANDIDATES else []


def classify_row(abbr: str, row: pd.Series) -> str:
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
        or re.search(r"\b1,?3-?bdo\b|\b1,?4-?bdo\b|\b2,?3-?bdo\b|\b1,?5-?pedo\b|\b1,?6-?hdo\b", a)
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
        or re.search(r"\b(phthalic|isophthalic|terephthalic|succinic|sebacic|adipic|glutaric|malonic|maleic|fumaric)\b", name)
        or re.search(r"\bmah\b|\bhhpa\b|\bmhhpa\b", a)
    ):
        return "acid_anhydride"

    # Epoxy family
    if (
        "epoxy" in name
        or "oxirane" in name
        or "glycid" in name
        or int(T.loc[abbr, "__epoxy_fn__"]) > 0
        or re.search(r"\bdgeba\b|\bdgebf\b|\bdgef\b|\bdge\b|\bge\b", a)
    ):
        if (
            "bisphenol" in name
            or "novolac" in name
            or "resin" in name
            or re.search(r"\bdgeba\b|\bdgebf\b|\bdgef\b", a)
        ):
            return "epoxy_resin"
        return "reactive_diluent"

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

    # Solvents/plasticizers (expanded)
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
        or re.search(r"\bteos\b|\bvtms\b|\bvtmsoh\b|\baptes\b|\bmptes\b|\bgpts\b|\bgptms\b", a)
    ):
        return "silane"

    return "other"

T["__class__"] = [classify_row(i, T.loc[i]) for i in T.index]

def list_by_class(cls: str) -> list[str]:
    opts = [i for i in T.index if T.loc[i, "__class__"] == cls]
    return list(dict.fromkeys(opts))

def select_abbr(title: str, options: list[str], key: str):
    """Select by Abbrev, but keep widget values stable and also store the selected
    Abbrev (not the rendered label) into st.session_state[key].

    This is critical for auto-fill callbacks (e.g., OH/EEW/AHEW), because Streamlit
    widgets store their own value and we don't want labels/formatting to break lookups.
    """
    if not options:
        st.warning(f"No options found for: {title}")
        st.session_state[key] = None
        return None

    options = list(dict.fromkeys(options))

    # Use a dedicated widget key to avoid clobbering the stored Abbrev key.
    widget_key = f"{key}__label"

    # The widget value is the Abbrev itself; label() is only for display.
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
# UI TABS
# =========================
tab_form, tab_maps, tab_figs = st.tabs(["Formulator", "Heatmaps", "FIGURES: Ra vs Δδa and Ra vs Π"])

# =========================
# EQUIV LIB EDITOR
# =========================
def equiv_editor():
    with st.expander("📚 Library of equivalents (editable)", expanded=False):
        st.caption("Matching is done by **Abbrev** (case-insensitive). You can edit values and they will be used to auto-fill stoichiometry inputs.")
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
        # fallback may return a dict of equivalent-like values
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

    # Candidate pools
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")

    # -------------------------
    # Epoxy pools (robust)
    # -------------------------
    # We build epoxy candidates using:
    #   (i) estimated epoxy functionality from SMILES/name (T['__epoxy_fn__'])
    #   (ii) editable equivalents library roles (Epoxy resin / Reactive diluent)
    #   (iii) heuristic class labels (epoxy_resin / reactive_diluent)
    #
    # This ensures aliphatic epoxies and other non-obvious entries still appear.

    def _epoxy_fn(ab: str) -> int:
        try:
            v = T.loc[ab, "__epoxy_fn__"] if "__epoxy_fn__" in T.columns else 0
            if pd.isna(v):
                return 0
            return int(v)
        except Exception:
            return 0

    epoxy_candidates = [ab for ab in T.index if _epoxy_fn(ab) > 0 or T.loc[ab, "__class__"] in ("epoxy_resin", "reactive_diluent")]

    # roles from library
    eqdf = get_equiv_df()
    role_map = eqdf.set_index("Abbrev")["Role"].astype(str).str.lower()

    epoxy_resins_role = [ab for ab, r in role_map.items() if r.strip() == "epoxy resin"]
    epoxy_dils_role   = [ab for ab, r in role_map.items() if "diluent" in r]

    # classify by epoxy functionality
    ep_resins = sorted(set(
        [ab for ab in epoxy_candidates if _epoxy_fn(ab) >= 2]
        + list_by_class("epoxy_resin")
        + epoxy_resins_role
    ))
    ep_dils = sorted(set(
        [ab for ab in epoxy_candidates if _epoxy_fn(ab) == 1]
        + list_by_class("reactive_diluent")
        + epoxy_dils_role
    ))
    ep_hards = sorted(set(
        list_by_class("epoxy_hardener")
        + [ab for ab, r in role_map.items() if ("hardener" in r or "amine" in r)]
    ))

    # Filter to compounds that exist in the database
    ep_resins = [e for e in ep_resins if e in T.index]
    ep_hards  = [e for e in ep_hards if e in T.index]
    ep_dils   = [e for e in ep_dils if e in T.index]
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
        st.caption("Uses %NCO and OH number. Inputs auto-update when you change Polyol/Extender/Crosslinker (if values exist in the equivalents library).")

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

        # auto-fill from library (will work because session_state stores Abbrev)
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

    elif mode == "Epoxy stoichiometry (EEW/AHEW)":
        st.markdown("### Epoxy stoichiometry (EEW/AHEW)")
        st.caption("Reactive diluent accepts any epoxy candidate filtered by **estimated epoxy functionality** (from SMILES/name).")

        c1, c2, c3 = st.columns(3)
        with c1:
            resin = select_abbr("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = select_abbr("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

        fn_target = st.slider("Reactive diluent — epoxy functionality filter (estimated)", 1, 4, 1, step=1, key="ep_fn_target")
        ep_dils = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        if use_dil and (not ep_dils):
            ep_dils = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) >= 1]

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
            dil_cand = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
            cand = ep_hards + (dil_cand if include_dil else [])
            cand = [c for c in cand if c in T.index]
            if resin and cand:
                rows = []
                for c in cand:
                    if c == resin:
                        continue
                    a = T.loc[c]; b = T.loc[resin]
                    ra = Ra(a, b); da = delta_a(a, b); ds = delta_sigmaL(a, b)
                    pi = Pi(da, ds, da_max, ds_max)
                    rows.append({"candidate": c, "type": T.loc[c, "__class__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
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
                    rows.append({"candidate": c, "type": T.loc[c, "__class__"], "Ra": ra, "Δδa": da, "ΔσL": ds, "Π": pi})
                dfc = pd.DataFrame(rows)
                raM = float(dfc["Ra"].max()); daM = float(dfc["Δδa"].max()); piM = float(dfc["Π"].max())
                dfc["Score_%"] = [score_percent(r, d, p, raM, daM, piM) for r, d, p in zip(dfc["Ra"], dfc["Δδa"], dfc["Π"])]
                dfc = dfc.sort_values("Score_%", ascending=False)
                st.dataframe(dfc.round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3, "Score_%": 1}), use_container_width=True, hide_index=True)

# -------------------------
# HEATMAPS (as in your build, but fixed lists)
# -------------------------
with tab_maps:
    st.subheader("Heatmaps (grouped by purpose)")

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

    # Candidate pools for heatmaps
    acids = list_by_class("acid_anhydride")
    isos = list_by_class("isocyanate")
    polyols = list_by_class("polyol")
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")

    # Epoxy pools for heatmaps (same logic as Formulator)
    def _epoxy_fn_hm(ab: str) -> int:
        try:
            v = T.loc[ab, "__epoxy_fn__"] if "__epoxy_fn__" in T.columns else 0
            if pd.isna(v):
                return 0
            return int(v)
        except Exception:
            return 0

    epoxy_candidates = [ab for ab in T.index if _epoxy_fn_hm(ab) > 0 or T.loc[ab, "__class__"] in ("epoxy_resin", "reactive_diluent")]

    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    # Fallback: if solvent/plasticizer class is empty (depends on database naming),
    # derive a reasonable solvent/plasticizer pool directly from names/abbreviations.
    if not solv:
        solv_guess = []
        solv_keys = [
            "solvent","plasticizer","diluent","thinner",
            "acetone","methyl ethyl ketone","mek","mibk","ketone",
            "toluene","xylene","ethyl acetate","butyl acetate","acetate",
            "thf","tetrahydrofuran","dmf","dms(o","dmso","nmp",
            "chloroform","dichloromethane","dcm","hexane","heptane","cyclohexane",
            "ethanol","methanol","propanol","butanol","isopropanol",
            "phthalate","adipate","sebacate","citrate","phosphate","benzoate","trimellitate"
        ]
        solv_abbr = {
            "THF","DMF","DMSO","NMP","MEK","MIBK","TOL","XYL","EA","ETAC","BUAC","IPA","ETOH","MEOH",
            "DBP","DEHP","DINP","DIDP","DOS","DOA","TOTM","TBP","TCP","TEP"
        }
        for ab in T.index:
            nm = _name_of(ab).lower()
            a = str(ab).upper()
            if (a in solv_abbr) or any(k in nm for k in solv_keys):
                solv_guess.append(ab)
        solv = sorted(set(solv_guess))
    sil = list_by_class("silane")

    # Enrich lists using the editable equivalents library (roles)
    eqdf_hm = get_equiv_df()
    if "Role" in eqdf_hm.columns:
        _role = eqdf_hm.set_index("Abbrev")["Role"].astype(str).str.lower()
    else:
        _role = pd.Series(dtype=str)

    acids = sorted(set(acids + [a for a,r in _role.items() if r.strip() in ("diacid","acid","anhydride","acid/anhydride")]))
    extenders = sorted(set(extenders + [a for a,r in _role.items() if "extender" in r]))
    crosslinkers = sorted(set(crosslinkers + [a for a,r in _role.items() if ("crosslinker" in r or "triol" in r)]))

    ep_resins = sorted(set(
        [ab for ab in epoxy_candidates if _epoxy_fn_hm(ab) >= 2]
        + list_by_class("epoxy_resin")
        + [a for a,r in _role.items() if r.strip() == "epoxy resin"]
    ))
    ep_hards = sorted(set(
        list_by_class("epoxy_hardener")
        + [a for a,r in _role.items() if ("hardener" in r or "amine" in r)]
    ))
    ep_dils = sorted(set(
        [ab for ab in epoxy_candidates if _epoxy_fn_hm(ab) == 1]
        + list_by_class("reactive_diluent")
        + [a for a,r in _role.items() if "diluent" in r]
    ))

    mons = sorted(set(mons + [a for a,r in _role.items() if "vinyl" in r or "monomer" in r]))
    solv = sorted(set(solv + [a for a,r in _role.items() if "solvent" in r or "plasticizer" in r]))
    sil  = sorted(set(sil  + [a for a,r in _role.items() if "silane" in r]))

    # Filter to compounds that exist in the database
    acids = [a for a in acids if a in T.index]
    extenders = [a for a in extenders if a in T.index]
    crosslinkers = [a for a in crosslinkers if a in T.index]
    ep_resins = [a for a in ep_resins if a in T.index]
    ep_hards  = [a for a in ep_hards if a in T.index]
    ep_dils   = [a for a in ep_dils if a in T.index]
    mons = [a for a in mons if a in T.index]
    solv = [a for a in solv if a in T.index]
    sil  = [a for a in sil if a in T.index]

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
        ep_dils = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        if not ep_dils:
            ep_dils = [a for a in EPOXY_CANDIDATES if int(T.loc[a, "__epoxy_fn__"]) >= 1]

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
        if not solv:
            st.warning("No solvents/plasticizers found. Check classification keywords or add them to the equivalents library with Role='Solvent'/'Plasticizer'.")
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
        res_sel = st.multiselect("Resins", list_by_class("epoxy_resin"), default=_def(list_by_class("epoxy_resin"), 5), format_func=label)
        hard_sel = st.multiselect("Hardeners", list_by_class("epoxy_hardener"), default=_def(list_by_class("epoxy_hardener"), 10), format_func=label)
        fn_target = st.slider("Reactive diluents — epoxy functionality (figures)", 1, 4, 1, step=1, key="fig_ep_fn")
        dils_all = sorted(set(list_by_class("reactive_diluent") + [a for a in T.index if (int(T.loc[a, "__epoxy_fn__"]) >= 1)]))
        dils = [a for a in dils_all if int(T.loc[a, "__epoxy_fn__"]) == int(fn_target)]
        if not dils:
            dils = [a for a in dils_all if int(T.loc[a, "__epoxy_fn__"]) >= 1]
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
        st.stop()

    st.pyplot(plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)"))
    st.pyplot(plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)"))

    st.markdown("#### Data used in figures")
    st.dataframe(dfp.sort_values("Ra").round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3}), use_container_width=True, hide_index=True)
