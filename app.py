
# ============================================================
# DiPEVa Formulator — CLEAN FINAL VERSION
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic / research use only — database-driven screening tool.")

HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "Database_clean_for_app.xlsx"

@st.cache_data
def load_db(p):
    df = pd.read_excel(p)
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_db(DB_FILE)
df["Abbrev"] = df["Abbrev"].astype(str).str.upper().str.strip()
df = df.drop_duplicates("Abbrev").set_index("Abbrev")
T = df.copy()

def list_by_class(cls):
    return sorted(T[T["Class_eff"] == cls].index.tolist())

def label(ab):
    if "Name" in T.columns and isinstance(T.loc[ab, "Name"], str):
        nm = T.loc[ab, "Name"].strip()
        if nm:
            return f"{ab} — {nm}"
    return ab

def Ra(a,b):
    return float(np.sqrt(4*(a["δD"]-b["δD"])**2 + (a["δP"]-b["δP"])**2 + (a["δH"]-b["δH"])**2))

def delta_a(a,b):
    if "δa" in T.columns and not pd.isna(a.get("δa")) and not pd.isna(b.get("δa")):
        return float(abs(a["δa"]-b["δa"]))
    return float(abs(np.sqrt(a["δP"]**2+a["δH"]**2)-np.sqrt(b["δP"]**2+b["δH"]**2)))

def Pi(da, ds, daM, dsM):
    nda = da/(daM+1e-12)
    nds = ds/(dsM+1e-12) if np.isfinite(ds) else 0
    return nda+nds

def score(ra,da,pi,raM,daM,piM):
    return 100*(1-(ra/raM+da/daM+pi/piM)/3)

epoxy_resins = list_by_class("epoxy_resin")+list_by_class("epoxy_resin_aliphatic")
epoxy_hardeners = list_by_class("epoxy_hardener")
reactive_diluents = list_by_class("reactive_diluent")
polyols = list_by_class("polyol_polymeric")
extenders = list_by_class("chain_extender_diol")
crosslinkers = list_by_class("crosslinker_triol")
acids = list_by_class("acid_diacid")+list_by_class("anhydride")

da_max = float(T["δa"].max()) if "δa" in T.columns else 1.0
ds_max = float(T["σL"].max()) if "σL" in T.columns else 1.0

tab1,tab2,tab3 = st.tabs(["Formulator","Heatmaps","Figures"])

with tab1:
    st.subheader("Pair screening")
    A = st.selectbox("A", T.index, format_func=label)
    B = st.selectbox("B", T.index, format_func=label)
    a,b = T.loc[A],T.loc[B]
    ra = Ra(a,b)
    da = delta_a(a,b)
    pi = Pi(da, np.nan, da_max, ds_max)
    st.metric("Ra", f"{ra:.2f}")
    st.metric("Δδa", f"{da:.2f}")
    st.metric("Π", f"{pi:.2f}")

with tab2:
    st.subheader("Epoxy heatmap")
    import matplotlib.pyplot as plt
    mat = pd.DataFrame(index=epoxy_hardeners+reactive_diluents, columns=epoxy_resins)
    for r in mat.index:
        for c in mat.columns:
            mat.loc[r,c]=Ra(T.loc[r],T.loc[c])
    fig,ax=plt.subplots()
    ax.imshow(mat.astype(float),aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns,rotation=90)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    st.pyplot(fig)

with tab3:
    st.subheader("Ra vs Δδa")
    rows=[]
    for r in epoxy_hardeners+reactive_diluents:
        for c in epoxy_resins:
            rows.append({"Ra":Ra(T.loc[r],T.loc[c]),"Δδa":delta_a(T.loc[r],T.loc[c])})
    dfp=pd.DataFrame(rows)
    fig,ax=plt.subplots()
    ax.scatter(dfp["Ra"],dfp["Δδa"])
    ax.set_xlabel("Ra")
    ax.set_ylabel("Δδa")
    st.pyplot(fig)
