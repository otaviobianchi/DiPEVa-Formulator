# ============================================================
# DiPEVa Formulator — FINAL, CLEAN, ARTICLE-ALIGNED
# Classification-driven (NO heuristics, NO inference)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="DiPEVa Formulator", layout="wide")
st.title("🧪 DiPEVa Formulator")
st.caption("Academic use only — classification-driven screening tool.")

DB_FILE = Path(__file__).resolve().parent / "Database_final_classified_strict_polymeric.xlsx"

# ============================================================
# LOAD DATABASE
# ============================================================
df = pd.read_excel(DB_FILE)
df.columns = df.columns.str.strip()

REQUIRED = {"Abbrev", "Class_eff", "δD", "δP", "δH"}
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Database missing required columns: {missing}")
    st.stop()

df["Abbrev"] = df["Abbrev"].astype(str).str.upper().str.strip()
df = df.drop_duplicates("Abbrev").set_index("Abbrev")

T = df.copy()

# ============================================================
# HELPERS
# ============================================================
def list_by_class(cls):
    return T[T["Class_eff"] == cls].index.tolist()

def label(ab):
    nm = T.loc[ab, "Name"] if "Name" in T.columns else ""
    return f"{ab} — {nm}" if isinstance(nm, str) and nm.strip() else ab

# ============================================================
# METRICS (ARTICLE)
# ============================================================
def Ra(a, b):
    return float(np.sqrt(
        4 * (a["δD"] - b["δD"])**2 +
            (a["δP"] - b["δP"])**2 +
            (a["δH"] - b["δH"])**2
    ))

def delta_a(a, b):
    if "δa" in T.columns:
        return float(abs(a["δa"] - b["δa"]))
    return float(abs(
        np.sqrt(a["δP"]**2 + a["δH"]**2) -
        np.sqrt(b["δP"]**2 + b["δH"]**2)
    ))

def delta_sigma(a, b):
    if "σL" in T.columns:
        return float(abs(a["σL"] - b["σL"]))
    return np.nan

def Pi(da, ds, da_max, ds_max):
    nda = da / (da_max + 1e-12)
    nds = ds / (ds_max + 1e-12) if np.isfinite(ds) else 0.0
    return nda + nds

def score(ra, da, pi, raM, daM, piM):
    return 100 * (1 - (ra/raM + da/daM + pi/piM) / 3)

# ============================================================
# POOLS — STRICTLY FROM Class_eff
# ============================================================
epoxy_resins = (
    list_by_class("epoxy_resin") +
    list_by_class("epoxy_resin_aliphatic")
)
reactive_diluents = list_by_class("reactive_diluent")
epoxy_hardeners = list_by_class("epoxy_hardener")

polyols = list_by_class("polyol_polymeric")
extenders = list_by_class("chain_extender_diol")
crosslinkers = list_by_class("crosslinker_triol")

acids = list_by_class("acid_diacid") + list_by_class("anhydride")

# ============================================================
# TABS
# ============================================================
tab_form, tab_maps, tab_figs = st.tabs(
    ["Formulator", "Heatmaps", "Figures"]
)

da_max = float(T["δa"].max()) if "δa" in T.columns else 1.0
ds_max = float(T["σL"].max()) if "σL" in T.columns else 1.0

# ============================================================
# FORMULATOR
# ============================================================
with tab_form:
    st.subheader("Formulator")

    mode = st.selectbox(
        "Mode",
        ["Pair screening", "Epoxy ranking", "Polyester screening"]
    )

    if mode == "Pair screening":
        A = st.selectbox("Component A", T.index, format_func=label)
        B = st.selectbox("Component B", T.index, format_func=label)

        a, b = T.loc[A], T.loc[B]
        ra = Ra(a, b)
        da = delta_a(a, b)
        ds = delta_sigma(a, b)
        pi = Pi(da, ds, da_max, ds_max)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ra", f"{ra:.2f}")
        c2.metric("Δδa", f"{da:.2f}")
        c3.metric("ΔσL", f"{ds:.2f}" if np.isfinite(ds) else "n/a")
        c4.metric("Π", f"{pi:.3f}")

    if mode == "Epoxy ranking":
        resin = st.selectbox("Epoxy resin", epoxy_resins, format_func=label)
        candidates = epoxy_hardeners + reactive_diluents

        rows = []
        for c in candidates:
            a, b = T.loc[c], T.loc[resin]
            ra = Ra(a, b)
            da = delta_a(a, b)
            ds = delta_sigma(a, b)
            pi = Pi(da, ds, da_max, ds_max)
            rows.append({
                "Candidate": c,
                "Class": T.loc[c, "Class_eff"],
                "Ra": ra,
                "Δδa": da,
                "Π": pi
            })

        df_rank = pd.DataFrame(rows)
        df_rank["Score"] = score(
            df_rank["Ra"],
            df_rank["Δδa"],
            df_rank["Π"],
            df_rank["Ra"].max(),
            df_rank["Δδa"].max(),
            df_rank["Π"].max()
        )

        st.dataframe(
            df_rank.sort_values("Score", ascending=False),
            use_container_width=True
        )

# ============================================================
# HEATMAPS
# ============================================================
with tab_maps:
    st.subheader("Heatmaps")

    family = st.selectbox(
        "Family",
        ["Epoxy", "Polyesters"]
    )

    def heatmap(rows, cols, title):
        mat = pd.DataFrame(index=rows, columns=cols)
        for r in rows:
            for c in cols:
                mat.loc[r, c] = Ra(T.loc[r], T.loc[c])
        fig, ax = plt.subplots(figsize=(0.6*len(cols)+3, 0.4*len(rows)+3))
        im = ax.imshow(mat.astype(float), aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Ra (MPa$^{1/2}$)")
        st.pyplot(fig)

    if family == "Epoxy":
        heatmap(
            epoxy_hardeners + reactive_diluents,
            epoxy_resins,
            "Epoxy — Hardeners & Reactive Diluents × Resins"
        )

    if family == "Polyesters":
        heatmap(
            polyols,
            acids,
            "Polyesters — Polyols × Acids / Anhydrides"
        )

# ============================================================
# FIGURES
# ============================================================
with tab_figs:
    st.subheader("Figures — Ra vs Δδa")

    fam = st.selectbox("Family", ["Epoxy", "Polyesters"])

    if fam == "Epoxy":
        A = epoxy_hardeners + reactive_diluents
        B = epoxy_resins
    else:
        A = polyols
        B = acids

    rows = []
    for a in A:
        for b in B:
            rows.append({
                "Ra": Ra(T.loc[a], T.loc[b]),
                "Δδa": delta_a(T.loc[a], T.loc[b])
            })

    dfp = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    ax.scatter(dfp["Ra"], dfp["Δδa"])
    ax.set_xlabel("Ra (MPa$^{1/2}$)")
    ax.set_ylabel("Δδa (MPa$^{1/2}$)")
    st.pyplot(fig)
