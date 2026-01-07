# DiPEVa Formulator — Streamlit App

# -----------------------------
# Safe numeric parsing helpers
# -----------------------------
def _to_float(x):
    """Safe float conversion (no try/except blocks)."""
    try:
        return float(x)
    except Exception:
        return float("nan")

# Academic / research use only — screening tool, not a standalone decision device.

import re
from pathlib import Path
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

HERE = Path(__file__).resolve().parent

# Prefer the classified database (recommended for the manuscript figures).
DB_FILE = HERE / "Database_final_classified_strict.xlsx"
if not DB_FILE.exists():
    DB_FILE = HERE / "Database_final_classified_full.xlsx"
if not DB_FILE.exists():
    DB_FILE = HERE / "Database_final.xlsx"

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
COL["cls"]    = _find_col(["Class", "Classification", "Function", "__class__", "Role"])

missing = [k for k in ["abbr", "dD", "dP", "dH"] if COL.get(k) is None]
if missing:
    st.error(
        "Database is missing required columns: "
        f"{missing}\nFound columns: {list(df_raw.columns)}"
    )
    st.stop()

if COL.get("cls") is None:
    st.warning(
        "No classification column was found in the database. "
        "The app will use internal heuristics (less reliable). "
        "Recommended: add a 'Class'/'Classification' column."
    )

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
# CLASS NORMALIZATION
# =========================
def _norm_class(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

# Map (database Class) -> (app UI family)
# Keep these labels stable (used across tabs + manuscript figures).
_CLASS_MAP = {
    # PU
    "isocyanate": "isocyanate",
    "polyol": "polyol",
    "polyol_polymeric": "polyol",
    "polyol_or_ether": "polyol",
    "chain_extender_diol": "extender",
    "extender": "extender",
    "crosslinker_triol": "crosslinker",
    "crosslinker": "crosslinker",
    # Polyesters / acids
    "acid_diacid": "acid_anhydride",
    "acid_mono": "acid_anhydride",
    "anhydride": "acid_anhydride",
    "acid_anhydride": "acid_anhydride",
    # Epoxy
    "epoxy_resin": "epoxy_resin",
    "reactive_diluent": "reactive_diluent",
    "epoxy_hardener": "epoxy_hardener",
    "epoxy_hardener_amine": "epoxy_hardener",
    # Vinyls
    "vinyl_monomer": "vinyl_monomer",
    # Solvents / plasticizers
    "solvent": "solvent_plasticizer",
    "plasticizer_or_ester_solvent": "solvent_plasticizer",
    "solvent_plasticizer": "solvent_plasticizer",
    # Silanes
    "silane": "silane",
    # Additives / misc
    "additive_misc": "additive",
    "catalyst": "additive",
    "filler_pigment": "additive",
    "stabilizer": "additive",
}

def _class_of(abbr: str) -> str:
    if COL.get("cls") and (COL["cls"] in T.columns):
        raw = _norm_class(T.loc[abbr, COL["cls"]])
        return _CLASS_MAP.get(raw, raw if raw else "other")
    return "other"

# Effective class used by the app
T["__class__"] = [_class_of(a) for a in T.index]


# Precompute polymeric-like flag (used to keep polyol selectors clean)
if "__is_polymeric__" not in T.columns:
        smi_col = COL.get("smiles")
        name_col = COL.get("name")
        mw_col = COL.get("mw")
        flags = []
        for ab in T.index:
            nm = (T.loc[ab, name_col] if (name_col and name_col in T.columns) else "")
            smi = (T.loc[ab, smi_col] if (smi_col and smi_col in T.columns) else "")
            mw  = (T.loc[ab, mw_col]  if (mw_col  and mw_col  in T.columns) else float("nan"))
mw = _to_float(mw)
mw = _to_float(mw)
            flags.append(bool(_is_polymeric_like(ab, nm, smi, mw)))
        T["__is_polymeric__"] = flags
        T["__is_polymeric__"] = False

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

    # Difunctional alcohols (Extenders)
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
    # NOTE: BDGE is a **difunctional aliphatic epoxy resin** (not a mono-epoxy reactive diluent).
    {"Abbrev": "BDGE",  "Role": "Epoxy resin (aliphatic)", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.0, "AHEW_g_eq": None},
    {"Abbrev": "ECH",   "Role": "Reactive diluent (mono-epoxy)", "NCO_%": None, "OH_mgKOH_g": None, "EEW_g_eq": 92.5, "AHEW_g_eq": None},
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

    Heuristic:
      - infer nominal Mn from tokens like PEG200, PPG1000, PTMEG2000, PCL1000, PCDL1000
      - assume f=2 unless triol hints
      - OH# ≈ 56100 * f / Mn
    """
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


def estimate_OH_number_from_mw(abbr: str, f: int = 2) -> Optional[float]:
    """Fallback OH# (mgKOH/g) from molecular weight for small multifunctional alcohols.
    OH# ≈ 56100 * f / MW
    Returns None if MW is missing/non-finite.
    """
        ab = str(abbr).strip().upper()
        mw_col = COL.get("mw")
        if (mw_col is None) or (mw_col not in T.columns) or (ab not in T.index):
            return None
        mw = T.loc[ab, mw_col]
mw = _to_float(mw) if (mw is not None and not pd.isna(mw)) else float("nan")
        if (not np.isfinite(mw)) or (mw <= 0):
            return None
        return float(56100.0 * float(f) / float(mw))
        return None
        return float(56100.0 * float(f) / float(mw))
        return None


def _fallback_oh_extender(ab: str) -> dict:
    remember = estimate_OH_number_from_mw(ab, f=2)
    return {"OH_mgKOH_g": remember} if remember is not None else {}
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
    if COL.get("name") and COL["name"] in T.columns:
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

def _mw_of(abbr: str) -> float:
    c = COL.get("mw")
    if not c or c not in T.columns:
        return float("nan")
    v = T.loc[abbr, c]
        return float(v)
        return float("nan")

# =========================
# EPOXY FUNCTIONALITY (SMILES-based; no RDKit)
# =========================

# =========================
# POLYMERIC FLAG (for "polyol" pool hygiene)
# =========================
_BIGSMILES_MARKERS = ("{", "}", "[*]", ":[*]", "*]", "[H]", "n}", "n}", "repeat", "rep")

def _is_polymeric_like(abbr: str, name: str, smiles_big: str, mw: float) -> bool:
    """Heuristic: identify polymeric entries (BigSMILES / repeat-unit) for polyol pools.

    Goal: keep the **polyol selector** restricted to polymeric polyols (PEG/PPG/PTMEG/PCL/PCDL, etc.)
    and avoid small-molecule diols/triols leaking into the polyol list.

    Criteria (any true -> polymeric):
      - BigSMILES / repeat markers in SMILES/BigSMILES
      - Abbrev encodes polymer families with Mn (PEG200, PPG1000, PTMEG2000, PCL1000, PCDL1000, PDMSxxx, PBDxxx)
      - Name contains polymer-family cues (polyether/polyester/polycarbonate diol, PEG/PPG/PTMEG, etc.)
      - High MW (>= 250 g/mol) + polyol-like naming (fallback)

    This is intentionally conservative: it is used to *filter pools*, not to change chemistry data.
    """
    a = str(abbr or "").upper()
    nm = str(name or "").lower()
    smi = str(smiles_big or "")

    if any(m in smi for m in _BIGSMILES_MARKERS):
        return True
    if re.search(r"\b(PEG|PPG|PTMEG|PCL|PCDL|PDMS|PBD)\s*\d{2,5}\b", a):
        return True
    if any(k in nm for k in [
        "polyether", "polyester", "polycarbonate", "polybutadiene", "polydimethylsiloxane",
        "polyol", "diol (mn", "polyethylene glycol", "polypropylene glycol", "polytetramethylene", "caprolactone",
        "polycarbonate diol", "polyether diol", "polyester diol", "ptmeg", "peg", "ppg"
    ]):
        # exclude obvious small-molecule glycols already caught elsewhere
        if not any(k in nm for k in ["ethylene glycol", "diethylene glycol", "triethylene glycol", "butanediol", "hexanediol", "propanediol", "glycerol"]):
            return True

    # MW-based fallback (only if name hints polyol/polymer)
    if np.isfinite(mw) and mw >= 250 and any(k in nm for k in ["poly", "glycol", "diol", "polyol"]):
        return True

    return False

def _count_epoxide_rings(smiles: str) -> int:
    """Count oxirane rings in SMILES with conservative regex.

    Handles typical epoxide fragments:
      - C1OC1, C2OC2, ...
      - C1CO1, C2CO2, ...
      - O1CC1, O2CC2, ...
    """
    if not smiles:
        return 0
    s = str(smiles).replace(" ", "")
    n = 0
    n += len(re.findall(r"C(\d)OC\1", s))
    n += len(re.findall(r"C(\d)CO\1", s))
    n += len(re.findall(r"O(\d)CC\1", s))
    n += len(re.findall(r"OC(\d)OC\1", s))
    return int(n)

def estimate_epoxy_function(abbr: str) -> int:
    """Estimate epoxy functionality (# epoxide groups).

    Priority:
      1) Count oxirane rings from SMILES/BigSMILES.
      2) Heuristic on names/abbrev (conservative).
    """
    ab = str(abbr).strip().upper()
    if ab not in T.index:
        return 0

    smi_col = COL.get("smiles")
    smi = None
    if smi_col and smi_col in T.columns:
        v = T.loc[ab, smi_col]
        if isinstance(v, str) and v.strip() and v.strip().lower() not in ("nan", "none"):
            smi = v.strip()
    if smi:
        cnt = _count_epoxide_rings(smi)
        if cnt > 0:
            return int(min(cnt, 6))

    name = _name_of(ab).lower()
    a = ab.lower()

    # strong cues (fallback only)
    if any(k in name for k in ["epoxy", "oxirane", "epoxide", "glycidyl"]):
        if "triglycidyl" in name:
            return 3
        if "tetraglycidyl" in name:
            return 4
        # conservative default if clearly epoxy but unknown
        return 1

    # Abbrev cues
    if ab in {"DGEBA", "DGEBF", "EPN", "ECN"}:
        return 2
    if "DGE" in ab and "MONO" not in ab:
        return 2

    return 0

if "__epoxy_fn__" not in T.columns:

# ------------------------------------------------------------------
# Effective class used by the app (ARTICLE-CONSISTENT)
# ------------------------------------------------------------------
# Priority:
# 1) Database 'Class' column
# 2) Enforce epoxy functionality:
#    - epoxy_fn >= 2  -> epoxy_resin (e.g. BDGE, aliphatic diepoxies)
#    - epoxy_fn == 1  -> reactive_diluent (mono-epoxies)
# ------------------------------------------------------------------
if "__class_eff__" not in T.columns:
    if "Class" in T.columns:
        T["__class_eff__"] = T["Class"].astype(str).str.lower()
    else:
        T["__class_eff__"] = "other"

if "__epoxy_fn__" in T.columns:
    T.loc[T["__epoxy_fn__"] >= 2, "__class_eff__"] = "epoxy_resin"
    T.loc[T["__epoxy_fn__"] == 1, "__class_eff__"] = "reactive_diluent"


        T["__epoxy_fn__"] = [estimate_epoxy_function(a) for a in T.index]
        T["__epoxy_fn__"] = pd.to_numeric(T["__epoxy_fn__"], errors="coerce").fillna(0).astype(int)

# --- Effective class override for epoxy systems ---
# Many databases label epoxies inconsistently; for the app (and the manuscript),
# we enforce a functionality-first split:
#   epoxy_fn >= 2  -> epoxy_resin
#   epoxy_fn == 1  -> reactive_diluent
# Everything else keeps the database-provided class.
if "__class_eff__" not in T.columns:
    T["__class_eff__"] = T["__class__"].astype(str)

# override only when epoxy functionality is detected
    fn = T["__epoxy_fn__"].astype(int)
    T.loc[fn >= 2, "__class_eff__"] = "epoxy_resin"
    T.loc[fn == 1, "__class_eff__"] = "reactive_diluent"
    # fail-safe: keep database class
    T["__class_eff__"] = T.get("__class__", "other").astype(str)
        T["__epoxy_fn__"] = 0

def estimate_EEW_from_db_or_mw(abbr: str) -> float:
    eq = equiv_lookup(abbr)
    if eq and (eq.get("EEW_g_eq") is not None) and (not (isinstance(eq.get("EEW_g_eq"), float) and np.isnan(eq.get("EEW_g_eq")))):
        re
def list_polyols(polymeric_only: bool = True) -> list[str]:
    """Return polyols; by default only polymeric (repeat-unit) entries."""
    opts = list_polyols(polymeric_only=True)
    if polymeric_only:
            opts = [i for i in opts if bool(T.loc[i, "__is_polymeric__"])]
            pass
    return list(dict.fromkeys(opts))

turn float(eq["EEW_g_eq"])

    # look for an EEW column if present
    eew_col = None
    for c in T.columns:
        if "eew" in str(c).lower():
            eew_col = c
            break
    if eew_col:
        v = T.loc[abbr, eew_col]
            if not pd.isna(v):
        return float(v)
            pass

    mw = _mw_of(abbr)
    fn = int(T.loc[abbr, "__epoxy_fn__"])
    if np.isfinite(mw) and fn > 0:
        return float(mw / fn)
    return float("nan")

# =========================
# POOLS (fixed for manuscript usage)
# =========================
def list_by_class(cls: str) -> list[str]:
    col = "__class_eff__" if "__class_eff__" in T.columns else "__class__"
    opts = [i for i in T.index if str(T.loc[i, col]) == cls]
    return list(dict.fromkeys(opts))

def _role_series() -> pd.Series:
    eqdf = get_equiv_df()
    if "Abbrev" in eqdf.columns and "Role" in eqdf.columns:
        s = eqdf.set_index("Abbrev")["Role"].astype(str).str.lower()
        return s
    return pd.Series(dtype=str)

ROLE = _role_series()

def _is_epoxy_resin(ab: str) -> bool:
    # resin if class says so OR epoxy_fn>=2 OR role says "epoxy resin"
        if T.loc[ab, "__class__"] == "epoxy_resin":
            return True
        if int(T.loc[ab, "__epoxy_fn__"]) >= 2:
            return True
        r = str(ROLE.get(ab, "")).strip().lower()
        if "epoxy resin" in r:
            return True
        pass
    return False

def _is_epoxy_hardener(ab: str) -> bool:
        if T.loc[ab, "__class__"] == "epoxy_hardener":
            return True
        r = str(ROLE.get(ab, "")).strip().lower()
        if "hardener" in r or "amine" in r or "curing" in r:
            return True
        pass
    return False

def _is_reactive_diluent(ab: str) -> bool:
    """
    Reactive diluent pool (for this study/app):
    - If equivalents library Role contains 'diluent' -> treat as reactive diluent (even if epoxy_fn>=2, e.g., BDGE).
    - Else if database Class == 'reactive_diluent' -> reactive diluent.
    - Else if epoxy functionality == 1 -> reactive diluent (fallback heuristic).
    """
        r = str(ROLE.get(ab, "")).strip().lower()
        if "diluent" in r:
            return True
        if str(T.loc[ab, "__class__"]) == "reactive_diluent":
            return True
        if int(T.loc[ab, "__epoxy_fn__"]) == 1:
            return True
        pass
    return False


def epoxy_pools():
    epoxy_candidates = [ab for ab in T.index if int(T.loc[ab, "__epoxy_fn__"]) > 0]
    ep_resins = sorted({ab for ab in epoxy_candidates if _is_epoxy_resin(ab)})
    ep_hards  = sorted({ab for ab in T.index if _is_epoxy_hardener(ab)})
    ep_dils   = sorted({ab for ab in epoxy_candidates if _is_reactive_diluent(ab)})
    return epoxy_candidates, ep_resins, ep_hards, ep_dils

# =========================
# UI HELPERS
# =========================
def select_abbr(title: str, options: list[str], key: str):
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

    out["Class_eff"] = str(T.loc[abbr, "__class__"])
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
        st.caption("Matching is done by **Abbrev** (case-insensitive). Edit values to auto-fill stoichiometry inputs.")
        df = get_equiv_df()
        df_edit = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="equiv_editor")
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
            eq = fallback(abbr) or {}
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

    # Candidate pools (from database classification)
    isos = list_by_class("isocyanate")
    polyols = list_polyols(polymeric_only=True)
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")
    acids = list_by_class("acid_anhydride")
    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    sil  = list_by_class("silane")

    epoxy_candidates, ep_resins, ep_hards, ep_dils = epoxy_pools()

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
        st.caption("Uses %NCO and OH number. Inputs auto-update if equivalents exist in the library; polyol OH has a transparent fallback estimator.")

        c1, c2, c3 = st.columns(3)
        with c1:
            iso = select_abbr("Isocyanate", isos, key="pu_iso")
        with c2:
            pol = select_abbr("Polyol", polyols, key="pu_pol")
        with c3:
            use_ext = st.checkbox("Include chain extender", value=True, key="pu_use_ext")

        ext_kind = "Diol"
        ext = None
        if use_ext:
            ext_kind = st.radio("Extender type", ["Diol", "Amine"], horizontal=True, key="pu_ext_kind")
            if ext_kind == "Amine":
                # Use epoxy amine hardeners as PU amine extenders (e.g., DETA, TETA, IPDA)
                ext_pool = sorted(set(list_by_class("epoxy_hardener_amine") + list_by_class("epoxy_hardener")))
            else:
                ext_pool = extenders
            ext = select_abbr("Chain extender", ext_pool, key="pu_ext")

        use_xl = st.checkbox("Include Crosslinker/triol", value=False, key="pu_use_xl")
        xl = None
        if use_xl:
            xl = select_abbr("Crosslinker/triol", crosslinkers, key="pu_xl")

        sync_from_library("pu_iso", {"NCO_%": "pu_nco"}, lock)
        sync_from_library("pu_pol", {"OH_mgKOH_g": "pu_oh_pol"}, lock, fallback=lambda ab: {"OH_mgKOH_g": estimate_OH_number_from_abbr(ab)})
        if use_ext and ext:
            if st.session_state.get("pu_ext_kind", "Diol") == "Amine":
                sync_from_library("pu_ext", {"AHEW_g_eq": "pu_ahew_ext"}, lock)
            else:
                sync_from_library("pu_ext", {"OH_mgKOH_g": "pu_oh_ext"}, lock, fallback=_fallback_oh_extender)
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
            if (use_ext and ext) and (st.session_state.get("pu_ext_kind", "Diol") == "Amine"):
                ahew_ext_default = float(st.session_state.get("pu_ahew_ext", 24.0))
                ahew_ext = st.number_input("Amine AHEW extender (g/eq)", min_value=5.0, max_value=500.0,
                                           value=ahew_ext_default, step=0.5, key="pu_ahew_ext", disabled=not (use_ext and ext))
                oh_ext = 0.0  # placeholder (not used)
            else:
                oh_ext_default = float(st.session_state.get("pu_oh_ext", 1200.0 if use_ext else 0.0))
                oh_ext = st.number_input("OH number extender (mgKOH/g)", min_value=0.0, max_value=3000.0,
                                         value=oh_ext_default, step=1.0, key="pu_oh_ext", disabled=not (use_ext and ext))
                ahew_ext = np.inf

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
        # Extender equivalent weight depends on chemistry:
        # - Diol: EW_OH = 56100 / OH#
        # - Amine: use AHEW (g/eq of active H)
        if use_ext and ext and (st.session_state.get("pu_ext_kind", "Diol") == "Amine"):
            EW_ext = float(st.session_state.get("pu_ahew_ext", ahew_ext if "ahew_ext" in locals() else np.inf))
        else:
            EW_ext = 56100.0 / max(oh_ext, 1e-12) if (use_ext and ext and oh_ext > 0) else np.inf
        EW_OH_xl  = 56100.0 / max(oh_xl, 1e-12) if (use_xl and xl and oh_xl > 0) else np.inf
        EW_NCO    = 4200.0  / max(nco,   1e-12)

        if iso and pol:
            if mass_mode == "Polyol-side mass (polyol+extender+crosslinker)":
                m_pol_side = target
                m_pol = m_pol_side * pol_frac
                m_ext = m_pol_side * ext_frac if (use_ext and ext) else 0.0
                m_xl  = m_pol_side * xl_frac if (use_xl and xl) else 0.0
                eq_OH = (m_pol / EW_OH_pol) + (m_ext / EW_ext if (use_ext and ext) else 0.0) + (m_xl / EW_OH_xl if (use_xl and xl) else 0.0)
                m_iso = idx * eq_OH * EW_NCO
                m_total = m_pol_side + m_iso
            else:
                term = (pol_frac / EW_OH_pol) \
                       + (ext_frac / EW_ext if (use_ext and ext) else 0.0) \
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
        st.caption("Reactive diluents are detected by epoxy functionality = 1 **excluding epoxy resins** (e.g., BDGE stays resin).")

        c1, c2, c3 = st.columns(3)
        with c1:
            resin = select_abbr("Resin", ep_resins, key="ep_resin")
        with c2:
            hard = select_abbr("Hardener", ep_hards, key="ep_hard")
        with c3:
            use_dil = st.checkbox("Include reactive diluent", value=False, key="ep_use_dil")

        # Diluents: default epoxy_fn==1, but allow relaxing to "any epoxy_fn>=1 not resin/hardener"
        strict_mono = st.checkbox("Reactive diluent filter: epoxy_fn == 1 (recommended)", value=True, key="ep_dil_strict")
        dil_pool = ep_dils if strict_mono else sorted({ab for ab in epoxy_candidates if (not _is_epoxy_resin(ab)) and (not _is_epoxy_hardener(ab))})
        dil = None
        if use_dil:
            dil = select_abbr("Reactive diluent", dil_pool, key="ep_dil")

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
            cand = ep_hards + (ep_dils if include_dil else [])
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
# HEATMAPS
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

    # Refresh pools (to include any library edits)
    ROLE = _role_series()
    epoxy_candidates, ep_resins, ep_hards, ep_dils = epoxy_pools()

    acids = list_by_class("acid_anhydride")
    isos = list_by_class("isocyanate")
    polyols = list_polyols(polymeric_only=True)
    extenders = list_by_class("extender")
    crosslinkers = list_by_class("crosslinker")
    mons = list_by_class("vinyl_monomer")
    solv = list_by_class("solvent_plasticizer")
    sil  = list_by_class("silane")

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
        st.caption("Polyols/diols/triols × acids/anhydrides (heatmap).")
        left = st.multiselect(
            "Polyols/diols/triols (rows)",
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
            fig = plot_heatmap(mat, f"POLYESTERS — Alcohols × Acids/Anhydrides ({metric})", cbar_label)
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
            polyols,
            default=_default(polyols, 18),
            format_func=label,
            key="hm_pu_polyols",
        )
        if pol_sel and isos_sel:
            mat = build_matrix(pol_sel, isos_sel)
            fig = plot_heatmap(mat, f"PU — Polyols × Isocyanates ({metric})", cbar_label)
            st.pyplot(fig)

        if show_ext and extenders:
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
            ep_dils_use,
            default=_default(ep_dils_use, 18),
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
# FIGURES (Manuscript)
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

    # Refresh epoxy pools for figures
    ROLE = _role_series()
    epoxy_candidates, ep_resins, ep_hards, ep_dils = epoxy_pools()

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
        A = st.multiselect(
            "Alcohols (polyols/diols/triols)",
            sorted(set(list_polyols(polymeric_only=True) + list_by_class("extender") + list_by_class("crosslinker"))),
            default=_def(sorted(set(list_polyols(polymeric_only=True) + list_by_class("extender") + list_by_class("crosslinker"))), 10),
            format_func=label
        )
        B = st.multiselect("Acids/anhydrides", list_by_class("acid_anhydride"), default=_def(list_by_class("acid_anhydride"), 10), format_func=label)
        dfp = make_pairs(A, B, tagA="alc", tagB="acid")

    elif fam == "Polyurethanes":
        iso_sel = st.multiselect("Isocyanates", list_by_class("isocyanate"), default=_def(list_by_class("isocyanate"), 8), format_func=label)
        pol_sel = st.multiselect("Polyols", list_polyols(polymeric_only=True), default=_def(list_polyols(polymeric_only=True), 10), format_func=label)
        alc_sel = st.multiselect(
            "Alcohols (extenders + crosslinkers)",
            sorted(set(list_by_class("extender") + list_by_class("crosslinker"))),
            default=_def(sorted(set(list_by_class("extender") + list_by_class("crosslinker"))), 10),
            format_func=label
        )
        df1 = make_pairs(pol_sel, iso_sel, tagA="polyol", tagB="iso")
        df2 = make_pairs([a for a in alc_sel if a in T.index], iso_sel, tagA="alc", tagB="iso")
        dfp = pd.concat([df1, df2], ignore_index=True) if (not df1.empty or not df2.empty) else pd.DataFrame()

    elif fam == "Epoxies":
    res_sel = st.multiselect("Resins", ep_resins, default=_def(ep_resins, 8), format_func=label)
    hard_sel = st.multiselect("Hardeners", ep_hards, default=_def(ep_hards, 10), format_func=label)

    fn_choice = st.selectbox(
        "Reactive diluents — epoxy functionality (figures)",
        ["All", 1, 2, 3, 4],
        index=0,
        key="fig_ep_fn_choice",
        help="Use 'All' to include every reactive diluent (Role/Class-based).",
    )
    ep_dils_use = list(ep_dils)
    if fn_choice != "All":
        ep_dils_use = [a for a in ep_dils_use if int(T.loc[a, "__epoxy_fn__"]) == int(fn_choice)]
    if not ep_dils_use:
        ep_dils_use = list(ep_dils)

    dil_sel = st.multiselect("Reactive diluents", ep_dils_use, default=_def(ep_dils_use, 12), format_func=label)

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
        target_pool = sorted(set(target_pool + _def(list_by_class("vinyl_monomer"), 10) + _def(list_polyols(polymeric_only=True), 10)))
        tar_sel = st.multiselect("Targets", target_pool, default=_def(target_pool, 12), format_func=label)
        dfp = make_pairs(sil_sel, tar_sel, tagA="sil", tagB="target")

    if dfp.empty:
        st.warning("No pairs available (check selections).")
        st.stop()

    st.pyplot(plot_scatter(dfp, "Ra", "Δδa", "pair", "FIGURE — Ra vs Δδa", "Ra (MPa$^{1/2}$)", "Δδa (MPa$^{1/2}$)"))
    st.pyplot(plot_scatter(dfp, "Ra", "Π", "pair", "FIGURE — Ra vs Π", "Ra (MPa$^{1/2}$)", "Π (dimensionless)"))

    st.markdown("#### Data used in figures")
    st.dataframe(dfp.sort_values("Ra").round({"Ra": 2, "Δδa": 2, "ΔσL": 2, "Π": 3}), use_container_width=True, hide_index=True)
