from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Not Suitable for Work — Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_FILE = Path(__file__).with_name("Not_Suitable_for_Work_MASTER_Drive.xlsx")

# -----------------------------
# Stile
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,.08);
    border: 1px solid rgba(128,128,128,.20);
    padding: 14px 16px;
    border-radius: 16px;
}
.small-note {opacity:.72; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Caricamento
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(path_or_buffer):
    survey = pd.read_excel(path_or_buffer, sheet_name="Survey_Data", engine="openpyxl")
    # Per un file caricato via uploader bisogna riavvolgere il buffer.
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    ml = pd.read_excel(path_or_buffer, sheet_name="ML_Output", engine="openpyxl", header=None)
    return survey, ml

def clean_survey(df):
    age_col = "Quanti anni hai ?"
    df = df[df[age_col].notna()].copy()
    return df

def ml_value(ml, key, default=None):
    for _, row in ml.iterrows():
        if str(row.iloc[0]).strip() == key:
            return row.iloc[1]
    return default

def compact_label(value, max_len=38):
    s = str(value)
    return s if len(s) <= max_len else s[:max_len-1] + "…"

def distribution_chart(df, column, title, horizontal=False):
    counts = (
        df[column].fillna("Non specificato")
        .astype(str)
        .value_counts()
        .rename_axis("Categoria")
        .reset_index(name="N")
    )
    if horizontal:
        counts = counts.sort_values("N", ascending=True)
        fig = px.bar(counts, x="N", y="Categoria", orientation="h", text="N", title=title)
        fig.update_yaxes(ticktext=[compact_label(x) for x in counts["Categoria"]],
                         tickvals=counts["Categoria"])
    else:
        fig = px.bar(counts, x="Categoria", y="N", text="N", title=title)
        fig.update_xaxes(tickangle=-25)
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=20))
    return fig

# -----------------------------
# Header
# -----------------------------
st.title("NOT SUITABLE FOR WORK")
st.caption("Dashboard interattiva • GDG Palermo • dati letti dal master Excel")

uploaded = st.sidebar.file_uploader(
    "Aggiorna il dataset",
    type=["xlsx"],
    help="Puoi caricare una versione aggiornata del master senza modificare il codice."
)

source = uploaded if uploaded is not None else DEFAULT_FILE
if uploaded is not None:
    uploaded.seek(0)

try:
    survey, ml = load_excel(source)
except Exception as e:
    st.error(f"Impossibile leggere il file Excel: {e}")
    st.stop()

df = clean_survey(survey)

# -----------------------------
# Colonne reali del master
# -----------------------------
COL_CONTEXT = "Contesto Lavorativo :"
COL_AGE = "Quanti anni hai ?"
COL_GENDER = "Genere"
COL_MODE = "Modalità lavorativa"
COL_SECTOR = "Settore lavorativo: (In quale settore lavori?)"
COL_SENIORITY = "Anzianità Lavorativa Complessiva"
COL_HOURS = "Ore lavorative settimanali"
COL_ROLE = "Ruolo e livello di responsabilità"
COL_RISK = "CLASSIFICAZIONE STANDARD"
COL_PROFILE = "ML_Profile"
COL_INDEX = "Risk_Index_Experimental"
COL_EXH = "Risk_Esaurimento"
COL_DET = "Risk_Distacco"
COL_REAL = "Risk_Bassa_Realizzazione"

# -----------------------------
# Filtri
# -----------------------------
st.sidebar.header("Filtri")
filter_cols = {
    "Genere": COL_GENDER,
    "Età": COL_AGE,
    "Settore": COL_SECTOR,
    "Modalità lavorativa": COL_MODE,
    "Contesto lavorativo": COL_CONTEXT,
    "Anzianità": COL_SENIORITY,
    "Ore settimanali": COL_HOURS,
    "Ruolo": COL_ROLE,
}

filtered = df.copy()
for label, col in filter_cols.items():
    opts = sorted(filtered[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect(label, opts, placeholder="Tutti")
    if selected:
        filtered = filtered[filtered[col].astype(str).isin(selected)]

if st.sidebar.button("Azzera filtri", use_container_width=True):
    st.rerun()

st.sidebar.caption(f"Risposte visualizzate: {len(filtered)} / {len(df)}")

# -----------------------------
# KPI
# -----------------------------
n = len(filtered)
risk_ma = filtered[COL_RISK].isin(["Medium_Risk", "High_Risk"]).mean() if n else 0
risk_low = (filtered[COL_RISK] == "Low_Risk").mean() if n else 0
risk_index = pd.to_numeric(filtered[COL_INDEX], errors="coerce").mean() if n else 0

cv_acc = ml_value(ml, "Cross-Validation Accuracy", None)
dominant = ml_value(ml, "Fattore_Dominante", "—")
importance = ml_value(ml, "Importanza_Dominante", None)
updated = ml_value(ml, "Timestamp_Aggiornamento", "—")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rispondenti", f"{n}")
k2.metric("Risk Index medio", f"{risk_index:.1%}" if pd.notna(risk_index) else "—")
k3.metric("Rischio medio-alto", f"{risk_ma:.1%}")
k4.metric("Accuracy CV", f"{float(cv_acc):.1%}" if cv_acc is not None and pd.notna(cv_acc) else "—")

st.caption(
    "I primi tre KPI rispettano i filtri attivi. "
    "Accuracy CV e metriche ML provengono dall'ultimo output della Random Forest."
)

# -----------------------------
# Overview
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Profilo del campione", "Random Forest"])

with tab1:
    c1, c2 = st.columns(2)

    risk_map = {
        "Low_Risk": "Rischio basso",
        "Medium_Risk": "Rischio medio",
        "High_Risk": "Rischio alto",
    }
    risk_counts = (
        filtered[COL_RISK].map(risk_map).fillna(filtered[COL_RISK])
        .value_counts()
        .rename_axis("Rischio")
        .reset_index(name="N")
    )
    fig = px.pie(risk_counts, names="Rischio", values="N", hole=.58,
                 title="Distribuzione del rischio")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=20))
    c1.plotly_chart(fig, use_container_width=True)

    profile_counts = (
        filtered[COL_PROFILE].fillna("Non specificato")
        .value_counts()
        .rename_axis("Profilo")
        .reset_index(name="N")
    )
    fig = px.bar(profile_counts, x="Profilo", y="N", text="N",
                 title="Distribuzione ML Profile")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=20))
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(distribution_chart(filtered, COL_SECTOR, "Rispondenti per settore", True),
                    use_container_width=True)
    c4.plotly_chart(distribution_chart(filtered, COL_MODE, "Modalità lavorativa", True),
                    use_container_width=True)

    st.subheader("Dimensioni di rischio")
    dims = pd.DataFrame({
        "Dimensione": ["Esaurimento emotivo", "Distacco", "Bassa realizzazione"],
        "Indice medio": [
            pd.to_numeric(filtered[COL_EXH], errors="coerce").mean(),
            pd.to_numeric(filtered[COL_DET], errors="coerce").mean(),
            pd.to_numeric(filtered[COL_REAL], errors="coerce").mean(),
        ]
    })
    fig = px.bar(dims, x="Dimensione", y="Indice medio", text_auto=".1%",
                 range_y=[0, 1], title="Indice medio per dimensione")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    a, b = st.columns(2)
    a.plotly_chart(distribution_chart(filtered, COL_AGE, "Età"), use_container_width=True)
    b.plotly_chart(distribution_chart(filtered, COL_GENDER, "Genere"), use_container_width=True)

    a, b = st.columns(2)
    a.plotly_chart(distribution_chart(filtered, COL_SENIORITY, "Anzianità lavorativa", True),
                   use_container_width=True)
    b.plotly_chart(distribution_chart(filtered, COL_HOURS, "Ore lavorative settimanali", True),
                   use_container_width=True)

    st.plotly_chart(distribution_chart(filtered, COL_ROLE, "Ruolo e livello di responsabilità", True),
                    use_container_width=True)

    with st.expander("Mostra dati filtrati"):
        visible_cols = [COL_AGE, COL_GENDER, COL_MODE, COL_SECTOR, COL_SENIORITY,
                        COL_HOURS, COL_ROLE, COL_INDEX, COL_RISK, COL_PROFILE]
        st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Random Forest")
    m1, m2, m3 = st.columns(3)
    full_acc = ml_value(ml, "Accuracy_Full", None)
    m1.metric("Accuracy training", f"{float(full_acc):.1%}" if full_acc is not None else "—")
    m2.metric("5-Fold Cross Validation", f"{float(cv_acc):.1%}" if cv_acc is not None else "—")
    m3.metric("Importanza dominante", f"{float(importance):.1%}" if importance is not None else "—")

    st.info(f"Fattore dominante: **{dominant}**")
    st.caption(f"Ultimo aggiornamento ML: {updated}")

    # Top feature: colonne H:I del foglio ML_Output
    features = ml.iloc[1:, [7, 8]].copy()
    features.columns = ["Feature", "Importanza"]
    features["Importanza"] = pd.to_numeric(features["Importanza"], errors="coerce")
    features = features.dropna(subset=["Feature", "Importanza"]).sort_values("Importanza", ascending=True)

    if not features.empty:
        fig = px.bar(features, x="Importanza", y="Feature", orientation="h",
                     text_auto=".1%", title="Top feature importance")
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=max(420, len(features)*38),
                          margin=dict(l=10, r=10, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Matrice di confusione — Cross Validation")
    # Nel master: D6:F9 circa; estrazione esplicita dai valori
    try:
        cm = pd.DataFrame(
            [[float(ml.iloc[7,4]), float(ml.iloc[7,5])],
             [float(ml.iloc[8,4]), float(ml.iloc[8,5])]],
            index=["Reale Basso", "Reale Medio-Alto"],
            columns=["Pred. Basso", "Pred. Medio-Alto"]
        )
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        labels=dict(x="Predizione", y="Classe reale", color="N"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Matrice di confusione CV non disponibile nel formato atteso.")

st.divider()
st.caption(
    "Fonte: Survey_Data + ML_Output del file Not_Suitable_for_Work_MASTER_Drive.xlsx. "
    "La dashboard non modifica il master Excel."
)
