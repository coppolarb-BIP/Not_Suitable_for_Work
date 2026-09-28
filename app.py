from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Not Suitable for Work — Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_FILE = Path(__file__).with_name(
    "Not_Suitable_for_Work_MASTER_Drive.xlsx"
)


# ============================================================
# STILE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.20);
        padding: 16px 18px;
        border-radius: 16px;
    }

    [data-testid="stMetricLabel"] {
        font-size: .95rem;
    }

    [data-testid="stMetricValue"] {
        font-weight: 650;
    }

    .small-note {
        opacity: .72;
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COLONNE DEL MASTER
# ============================================================

COL_CONTEXT = "Contesto Lavorativo :"
COL_AGE = "Quanti anni hai ?"
COL_GENDER = "Genere"
COL_MODE = "Modalità lavorativa"

COL_SECTOR = (
    "Settore lavorativo: (In quale settore lavori?)"
)

COL_SENIORITY = "Anzianità Lavorativa Complessiva"
COL_HOURS = "Ore lavorative settimanali"
COL_ROLE = "Ruolo e livello di responsabilità"

COL_RISK = "CLASSIFICAZIONE STANDARD"
COL_PROFILE = "ML_Profile"

COL_INDEX = "Risk_Index_Experimental"

COL_EXH = "Risk_Esaurimento"
COL_DET = "Risk_Distacco"
COL_REAL = "Risk_Bassa_Realizzazione"


# ============================================================
# FUNZIONI LETTURA
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(path_or_buffer):

    survey = pd.read_excel(
        path_or_buffer,
        sheet_name="Survey_Data",
        engine="openpyxl",
    )

    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)

    ml = pd.read_excel(
        path_or_buffer,
        sheet_name="ML_Output",
        engine="openpyxl",
        header=None,
    )

    return survey, ml


def clean_survey(df):

    df = df.copy()

    if COL_AGE in df.columns:
        df = df[df[COL_AGE].notna()].copy()

    return df


def ml_value(ml, key, default=None):

    if ml is None or ml.empty:
        return default

    for _, row in ml.iterrows():

        if len(row) < 2:
            continue

        current_key = str(row.iloc[0]).strip()

        if current_key == key:
            return row.iloc[1]

    return default


def ml_value_multi(ml, keys, default=None):

    for key in keys:

        value = ml_value(
            ml,
            key,
            None,
        )

        if value is not None and pd.notna(value):
            return value

    return default


def safe_float(value):

    try:

        if value is None or pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def format_percent(value, decimals=1):

    value = safe_float(value)

    if value is None:
        return "—"

    return f"{value:.{decimals}%}"


def format_pp(value, decimals=1):

    value = safe_float(value)

    if value is None:
        return "—"

    sign = "+" if value > 0 else ""

    return f"{sign}{value * 100:.{decimals}f} p.p."


def compact_label(value, max_len=42):

    s = str(value)

    if len(s) <= max_len:
        return s

    return s[:max_len - 1] + "…"


# ============================================================
# METRICHE DATASET
# ============================================================

def dataset_metrics(df):

    n = len(df)

    if n == 0:

        return {
            "n": 0,
            "risk_index": None,
            "risk_ma": None,
            "risk_low": None,
            "exhaustion": None,
            "detachment": None,
            "realization": None,
        }

    risk_index = pd.to_numeric(
        df[COL_INDEX],
        errors="coerce",
    ).mean()

    risk_ma = df[COL_RISK].isin(
        ["Medium_Risk", "High_Risk"]
    ).mean()

    risk_low = (
        df[COL_RISK] == "Low_Risk"
    ).mean()

    exhaustion = pd.to_numeric(
        df[COL_EXH],
        errors="coerce",
    ).mean()

    detachment = pd.to_numeric(
        df[COL_DET],
        errors="coerce",
    ).mean()

    realization = pd.to_numeric(
        df[COL_REAL],
        errors="coerce",
    ).mean()

    return {
        "n": n,
        "risk_index": risk_index,
        "risk_ma": risk_ma,
        "risk_low": risk_low,
        "exhaustion": exhaustion,
        "detachment": detachment,
        "realization": realization,
    }


def ml_metrics(ml):

    return {

        "training": ml_value_multi(
            ml,
            [
                "Accuracy_Full_Training",
                "Accuracy_Full",
            ],
        ),

        "cv": ml_value_multi(
            ml,
            [
                "Accuracy_CV",
                "Cross-Validation Accuracy",
            ],
        ),

        "dominant": ml_value_multi(
            ml,
            [
                "Fattore_Dominante",
            ],
            "—",
        ),

        "importance": ml_value_multi(
            ml,
            [
                "Importanza_Dominante",
            ],
        ),

        "updated": ml_value_multi(
            ml,
            [
                "Timestamp_Aggiornamento",
            ],
            "—",
        ),

        "respondents": ml_value_multi(
            ml,
            [
                "Rispondenti",
            ],
        ),

        "risk_low": ml_value_multi(
            ml,
            [
                "Rischio_Basso",
            ],
        ),

        "risk_ma": ml_value_multi(
            ml,
            [
                "Rischio_Medio_Alto",
            ],
        ),
    }


# ============================================================
# GRAFICI
# ============================================================

def distribution_chart(
    df,
    column,
    title,
    horizontal=False,
):

    if column not in df.columns or df.empty:
        return None

    counts = (
        df[column]
        .fillna("Non specificato")
        .astype(str)
        .value_counts()
        .rename_axis("Categoria")
        .reset_index(name="N")
    )

    if horizontal:

        counts = counts.sort_values(
            "N",
            ascending=True,
        )

        fig = px.bar(
            counts,
            x="N",
            y="Categoria",
            orientation="h",
            text="N",
            title=title,
        )

        fig.update_yaxes(
            ticktext=[
                compact_label(x)
                for x in counts["Categoria"]
            ],
            tickvals=counts["Categoria"],
        )

    else:

        fig = px.bar(
            counts,
            x="Categoria",
            y="N",
            text="N",
            title=title,
        )

        fig.update_xaxes(
            tickangle=-25
        )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=20,
        ),
    )

    return fig


def comparison_distribution(
    master_df,
    new_df,
    column,
    title,
):

    if (
        column not in master_df.columns
        or column not in new_df.columns
    ):
        return None

    master = (
        master_df[column]
        .fillna("Non specificato")
        .astype(str)
        .value_counts(normalize=True)
        .mul(100)
        .rename("MASTER")
    )

    new = (
        new_df[column]
        .fillna("Non specificato")
        .astype(str)
        .value_counts(normalize=True)
        .mul(100)
        .rename("CARICATO")
    )

    comp = pd.concat(
        [master, new],
        axis=1,
    ).fillna(0)

    comp = (
        comp
        .reset_index()
        .rename(
            columns={
                "index": "Categoria"
            }
        )
    )

    long_df = comp.melt(
        id_vars="Categoria",
        var_name="Dataset",
        value_name="Percentuale",
    )

    fig = px.bar(
        long_df,
        x="Categoria",
        y="Percentuale",
        color="Dataset",
        barmode="group",
        text_auto=".1f",
        title=title,
    )

    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
    )

    fig.update_yaxes(
        title="Percentuale",
        ticksuffix="%",
    )

    fig.update_xaxes(
        tickangle=-25
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=30,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("NOT SUITABLE FOR WORK")

st.caption(
    "Dashboard interattiva • GDG Palermo • "
    "analisi del campione e Random Forest"
)


# ============================================================
# CARICAMENTO MASTER BASELINE
# ============================================================

try:

    master_survey, master_ml = load_excel(
        DEFAULT_FILE
    )

except Exception as e:

    st.error(
        "Impossibile leggere il MASTER di riferimento: "
        f"{e}"
    )

    st.stop()


master_df = clean_survey(
    master_survey
)


# ============================================================
# UPLOAD NUOVO FILE
# ============================================================

uploaded = st.sidebar.file_uploader(
    "Carica dataset aggiornato",
    type=["xlsx"],
    help=(
        "Il MASTER rimane come baseline. "
        "Il file caricato viene analizzato e confrontato "
        "con il MASTER."
    ),
)


has_upload = uploaded is not None


if has_upload:

    try:

        uploaded.seek(0)

        current_survey, current_ml = load_excel(
            uploaded
        )

        current_df = clean_survey(
            current_survey
        )

        current_name = uploaded.name

    except Exception as e:

        st.error(
            "Impossibile leggere il file caricato: "
            f"{e}"
        )

        st.stop()

else:

    current_survey = master_survey
    current_ml = master_ml
    current_df = master_df.copy()

    current_name = DEFAULT_FILE.name


# ============================================================
# CONTROLLO COLONNE
# ============================================================

required_columns = [
    COL_AGE,
    COL_GENDER,
    COL_MODE,
    COL_SECTOR,
    COL_SENIORITY,
    COL_HOURS,
    COL_ROLE,
    COL_RISK,
    COL_PROFILE,
    COL_INDEX,
    COL_EXH,
    COL_DET,
    COL_REAL,
]


missing = [
    col
    for col in required_columns
    if col not in current_df.columns
]


if missing:

    st.error(
        "Nel dataset corrente mancano alcune "
        "colonne necessarie:"
    )

    st.write(missing)

    st.stop()


# ============================================================
# METRICHE ML
# ============================================================

master_ml_stats = ml_metrics(
    master_ml
)

current_ml_stats = ml_metrics(
    current_ml
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.divider()

if has_upload:

    st.sidebar.success(
        "Dataset caricato"
    )

    st.sidebar.caption(
        f"Corrente: {current_name}"
    )

    st.sidebar.caption(
        f"Baseline: {DEFAULT_FILE.name}"
    )

else:

    st.sidebar.info(
        "Stai visualizzando il MASTER."
    )


st.sidebar.divider()
st.sidebar.header("Filtri")


filter_cols = {

    "Genere":
        COL_GENDER,

    "Età":
        COL_AGE,

    "Settore":
        COL_SECTOR,

    "Modalità lavorativa":
        COL_MODE,

    "Contesto lavorativo":
        COL_CONTEXT,

    "Anzianità":
        COL_SENIORITY,

    "Ore settimanali":
        COL_HOURS,

    "Ruolo":
        COL_ROLE,
}


filtered = current_df.copy()


for label, col in filter_cols.items():

    if col not in filtered.columns:
        continue

    opts = sorted(
        filtered[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected = st.sidebar.multiselect(
        label,
        opts,
        placeholder="Tutti",
    )

    if selected:

        filtered = filtered[
            filtered[col]
            .astype(str)
            .isin(selected)
        ]


if st.sidebar.button(
    "Azzera filtri",
    use_container_width=True,
):

    st.rerun()


st.sidebar.caption(
    f"Risposte visualizzate: "
    f"{len(filtered)} / {len(current_df)}"
)


# ============================================================
# KPI CORRENTI
# ============================================================

current_filtered_stats = dataset_metrics(
    filtered
)


k1, k2, k3, k4 = st.columns(4)


k1.metric(
    "Rispondenti",
    current_filtered_stats["n"],
)


k2.metric(
    "Risk Index medio",
    format_percent(
        current_filtered_stats["risk_index"]
    ),
)


k3.metric(
    "Rischio medio-alto",
    format_percent(
        current_filtered_stats["risk_ma"]
    ),
)


k4.metric(
    "Accuracy CV",
    format_percent(
        current_ml_stats["cv"]
    ),
)


if has_upload:

    st.caption(
        f"Dataset corrente: {current_name}. "
        "I primi tre KPI rispettano i filtri attivi. "
        "Accuracy CV proviene dall'ultimo output "
        "Random Forest del file caricato."
    )

else:

    st.caption(
        "Dataset corrente: MASTER. "
        "Carica un nuovo Excel dalla barra laterale "
        "per attivare il confronto."
    )


# ============================================================
# TAB
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Profilo del campione",
        "Random Forest",
        "Confronto dataset",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader("Quadro generale")

    c1, c2 = st.columns(2)


    risk_map = {
        "Low_Risk": "Rischio basso",
        "Medium_Risk": "Rischio medio",
        "High_Risk": "Rischio alto",
    }


    risk_counts = (
        filtered[COL_RISK]
        .map(risk_map)
        .fillna(filtered[COL_RISK])
        .value_counts()
        .rename_axis("Rischio")
        .reset_index(name="N")
    )


    if not risk_counts.empty:

        fig = px.pie(
            risk_counts,
            names="Rischio",
            values="N",
            hole=.58,
            title="Distribuzione del rischio",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=20,
            ),
        )

        c1.plotly_chart(
            fig,
            use_container_width=True,
        )


    profile_counts = (
        filtered[COL_PROFILE]
        .fillna("Non specificato")
        .value_counts()
        .rename_axis("Profilo")
        .reset_index(name="N")
    )


    if not profile_counts.empty:

        fig = px.bar(
            profile_counts,
            x="Profilo",
            y="N",
            text="N",
            title="Distribuzione ML Profile",
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=20,
            ),
        )

        c2.plotly_chart(
            fig,
            use_container_width=True,
        )


    c3, c4 = st.columns(2)


    fig = distribution_chart(
        filtered,
        COL_SECTOR,
        "Rispondenti per settore",
        True,
    )

    if fig is not None:

        c3.plotly_chart(
            fig,
            use_container_width=True,
        )


    fig = distribution_chart(
        filtered,
        COL_MODE,
        "Modalità lavorativa",
        True,
    )

    if fig is not None:

        c4.plotly_chart(
            fig,
            use_container_width=True,
        )


    st.subheader(
        "Dimensioni di rischio"
    )


    dims = pd.DataFrame(
        {
            "Dimensione": [
                "Esaurimento emotivo",
                "Distacco",
                "Bassa realizzazione",
            ],

            "Indice medio": [
                current_filtered_stats[
                    "exhaustion"
                ],
                current_filtered_stats[
                    "detachment"
                ],
                current_filtered_stats[
                    "realization"
                ],
            ],
        }
    )


    fig = px.bar(
        dims,
        x="Dimensione",
        y="Indice medio",
        text_auto=".1%",
        range_y=[0, 1],
        title="Indice medio per dimensione",
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# TAB 2 — PROFILO CAMPIONE
# ============================================================

with tab2:

    st.subheader(
        "Profilo del campione"
    )


    a, b = st.columns(2)


    fig = distribution_chart(
        filtered,
        COL_AGE,
        "Età",
    )

    if fig is not None:

        a.plotly_chart(
            fig,
            use_container_width=True,
        )


    fig = distribution_chart(
        filtered,
        COL_GENDER,
        "Genere",
    )

    if fig is not None:

        b.plotly_chart(
            fig,
            use_container_width=True,
        )


    a, b = st.columns(2)


    fig = distribution_chart(
        filtered,
        COL_SENIORITY,
        "Anzianità lavorativa",
        True,
    )

    if fig is not None:

        a.plotly_chart(
            fig,
            use_container_width=True,
        )


    fig = distribution_chart(
        filtered,
        COL_HOURS,
        "Ore lavorative settimanali",
        True,
    )

    if fig is not None:

        b.plotly_chart(
            fig,
            use_container_width=True,
        )


    fig = distribution_chart(
        filtered,
        COL_ROLE,
        "Ruolo e livello di responsabilità",
        True,
    )

    if fig is not None:

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with st.expander(
        "Mostra dati filtrati"
    ):

        visible_cols = [
            COL_AGE,
            COL_GENDER,
            COL_MODE,
            COL_SECTOR,
            COL_SENIORITY,
            COL_HOURS,
            COL_ROLE,
            COL_INDEX,
            COL_RISK,
            COL_PROFILE,
        ]

        st.dataframe(
            filtered[visible_cols],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 3 — RANDOM FOREST
# ============================================================

with tab3:

    st.subheader(
        "Performance Random Forest"
    )

    st.caption(
        "Le metriche provengono dal foglio ML_Output "
        "del dataset corrente."
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "Accuracy training",
        format_percent(
            current_ml_stats["training"]
        ),
    )


    m2.metric(
        "5-Fold Cross Validation",
        format_percent(
            current_ml_stats["cv"]
        ),
    )


    m3.metric(
        "Importanza dominante",
        format_percent(
            current_ml_stats["importance"]
        ),
    )


    st.info(
        "Fattore dominante: "
        f"**{current_ml_stats['dominant']}**"
    )


    st.caption(
        "Ultimo aggiornamento ML: "
        f"{current_ml_stats['updated']}"
    )


    # --------------------------------------------------------
    # TRAINING VS CV
    # --------------------------------------------------------

    performance_rows = []


    if safe_float(
        current_ml_stats["training"]
    ) is not None:

        performance_rows.append(
            {
                "Metrica": "Training",
                "Accuracy": float(
                    current_ml_stats["training"]
                ),
            }
        )


    if safe_float(
        current_ml_stats["cv"]
    ) is not None:

        performance_rows.append(
            {
                "Metrica": "5-Fold CV",
                "Accuracy": float(
                    current_ml_stats["cv"]
                ),
            }
        )


    if performance_rows:

        performance_df = pd.DataFrame(
            performance_rows
        )

        fig = px.bar(
            performance_df,
            x="Metrica",
            y="Accuracy",
            text_auto=".1%",
            range_y=[0, 1],
            title=(
                "Training accuracy vs "
                "Cross Validation"
            ),
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        fig.update_layout(
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "#### Feature importance"
    )


    try:

        features = current_ml.iloc[
            1:,
            [7, 8]
        ].copy()

        features.columns = [
            "Feature",
            "Importanza",
        ]

        features["Feature"] = (
            features["Feature"]
            .astype(str)
            .str.strip()
        )

        features["Importanza"] = pd.to_numeric(
            features["Importanza"],
            errors="coerce",
        )

        features = features.dropna(
            subset=["Importanza"]
        )

        features = features[
            ~features["Feature"].isin(
                ["", "nan", "None"]
            )
        ]

        features = features.sort_values(
            "Importanza",
            ascending=True,
        )


        if not features.empty:

            fig = px.bar(
                features,
                x="Importanza",
                y="Feature",
                orientation="h",
                text_auto=".1%",
                title=(
                    "Importanza delle variabili "
                    "nel modello"
                ),
            )

            fig.update_xaxes(
                tickformat=".0%"
            )

            fig.update_layout(
                height=max(
                    420,
                    len(features) * 42,
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "Feature importance non disponibile."
            )

    except Exception as e:

        st.warning(
            "Impossibile leggere la feature importance: "
            f"{e}"
        )


    # --------------------------------------------------------
    # MATRICE CONFUSIONE TRAINING
    # --------------------------------------------------------

    st.markdown(
        "#### Matrice di confusione — Training"
    )


    try:

        cm_training = pd.DataFrame(

            [
                [
                    float(current_ml.iloc[3, 4]),
                    float(current_ml.iloc[3, 5]),
                ],
                [
                    float(current_ml.iloc[4, 4]),
                    float(current_ml.iloc[4, 5]),
                ],
            ],

            index=[
                "Reale Basso",
                "Reale Medio-Alto",
            ],

            columns=[
                "Pred. Basso",
                "Pred. Medio-Alto",
            ],
        )


        fig = px.imshow(
            cm_training,
            text_auto=True,
            aspect="auto",
            labels=dict(
                x="Predizione",
                y="Classe reale",
                color="N",
            ),
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    except Exception:

        st.info(
            "Matrice di confusione training "
            "non disponibile nel formato atteso."
        )


    # --------------------------------------------------------
    # MATRICE CONFUSIONE CV
    # --------------------------------------------------------

    st.markdown(
        "#### Matrice di confusione — Cross Validation"
    )


    try:

        cm_cv = pd.DataFrame(

            [
                [
                    float(current_ml.iloc[7, 4]),
                    float(current_ml.iloc[7, 5]),
                ],
                [
                    float(current_ml.iloc[8, 4]),
                    float(current_ml.iloc[8, 5]),
                ],
            ],

            index=[
                "Reale Basso",
                "Reale Medio-Alto",
            ],

            columns=[
                "Pred. Basso",
                "Pred. Medio-Alto",
            ],
        )


        fig = px.imshow(
            cm_cv,
            text_auto=True,
            aspect="auto",
            labels=dict(
                x="Predizione",
                y="Classe reale",
                color="N",
            ),
        )

        fig.update_layout(
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    except Exception:

        st.warning(
            "Matrice di confusione CV "
            "non disponibile nel formato atteso."
        )


# ============================================================
# TAB 4 — CONFRONTO DATASET
# ============================================================

with tab4:

    st.subheader(
        "Confronto MASTER vs dataset caricato"
    )


    if not has_upload:

        st.info(
            "Carica un nuovo file Excel dalla barra "
            "laterale per confrontarlo con il MASTER."
        )

    else:

        master_stats = dataset_metrics(
            master_df
        )

        new_stats = dataset_metrics(
            current_df
        )


        # ====================================================
        # DIMENSIONE CAMPIONE
        # ====================================================

        st.markdown(
            "### Dimensione del campione"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "MASTER",
            master_stats["n"],
        )


        c2.metric(
            "CARICATO",
            new_stats["n"],
        )


        delta_n = (
            new_stats["n"]
            - master_stats["n"]
        )


        c3.metric(
            "Variazione",
            f"{delta_n:+d}",
        )


        # ====================================================
        # KPI COMPARATIVI
        # ====================================================

        st.markdown(
            "### Indicatori principali"
        )


        comparison_rows = [

            {
                "Indicatore":
                    "Risk Index medio",

                "MASTER":
                    master_stats["risk_index"],

                "CARICATO":
                    new_stats["risk_index"],
            },

            {
                "Indicatore":
                    "Rischio medio-alto",

                "MASTER":
                    master_stats["risk_ma"],

                "CARICATO":
                    new_stats["risk_ma"],
            },

            {
                "Indicatore":
                    "Rischio basso",

                "MASTER":
                    master_stats["risk_low"],

                "CARICATO":
                    new_stats["risk_low"],
            },

            {
                "Indicatore":
                    "Esaurimento emotivo",

                "MASTER":
                    master_stats["exhaustion"],

                "CARICATO":
                    new_stats["exhaustion"],
            },

            {
                "Indicatore":
                    "Distacco",

                "MASTER":
                    master_stats["detachment"],

                "CARICATO":
                    new_stats["detachment"],
            },

            {
                "Indicatore":
                    "Bassa realizzazione",

                "MASTER":
                    master_stats["realization"],

                "CARICATO":
                    new_stats["realization"],
            },
        ]


        comparison_df = pd.DataFrame(
            comparison_rows
        )


        comparison_df["Delta"] = (
            comparison_df["CARICATO"]
            - comparison_df["MASTER"]
        )


        display_df = comparison_df.copy()


        display_df["MASTER"] = (
            display_df["MASTER"]
            .apply(format_percent)
        )


        display_df["CARICATO"] = (
            display_df["CARICATO"]
            .apply(format_percent)
        )


        display_df["Delta"] = (
            comparison_df["Delta"]
            .apply(format_pp)
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # GRAFICO KPI
        # ====================================================

        chart_df = comparison_df.melt(
            id_vars="Indicatore",
            value_vars=[
                "MASTER",
                "CARICATO",
            ],
            var_name="Dataset",
            value_name="Valore",
        )


        fig = px.bar(
            chart_df,
            x="Indicatore",
            y="Valore",
            color="Dataset",
            barmode="group",
            text_auto=".1%",
            title=(
                "Confronto degli indicatori "
                "di rischio"
            ),
        )


        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1],
        )


        fig.update_xaxes(
            tickangle=-20
        )


        fig.update_layout(
            height=450,
            margin=dict(
                l=10,
                r=10,
                t=60,
                b=30,
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        st.markdown(
            "### Confronto Random Forest"
        )


        ml_rows = [

            {
                "Metrica":
                    "Accuracy training",

                "MASTER":
                    safe_float(
                        master_ml_stats["training"]
                    ),

                "CARICATO":
                    safe_float(
                        current_ml_stats["training"]
                    ),
            },

            {
                "Metrica":
                    "Accuracy CV",

                "MASTER":
                    safe_float(
                        master_ml_stats["cv"]
                    ),

                "CARICATO":
                    safe_float(
                        current_ml_stats["cv"]
                    ),
            },

            {
                "Metrica":
                    "Importanza dominante",

                "MASTER":
                    safe_float(
                        master_ml_stats["importance"]
                    ),

                "CARICATO":
                    safe_float(
                        current_ml_stats["importance"]
                    ),
            },
        ]


        ml_compare = pd.DataFrame(
            ml_rows
        )


        ml_compare["Delta"] = (
            ml_compare["CARICATO"]
            - ml_compare["MASTER"]
        )


        ml_display = ml_compare.copy()


        ml_display["MASTER"] = (
            ml_display["MASTER"]
            .apply(format_percent)
        )


        ml_display["CARICATO"] = (
            ml_display["CARICATO"]
            .apply(format_percent)
        )


        ml_display["Delta"] = (
            ml_compare["Delta"]
            .apply(format_pp)
        )


        st.dataframe(
            ml_display,
            use_container_width=True,
            hide_index=True,
        )


        ml_long = ml_compare.melt(
            id_vars="Metrica",
            value_vars=[
                "MASTER",
                "CARICATO",
            ],
            var_name="Dataset",
            value_name="Valore",
        )


        fig = px.bar(
            ml_long,
            x="Metrica",
            y="Valore",
            color="Dataset",
            barmode="group",
            text_auto=".1%",
            title=(
                "Performance Random Forest: "
                "MASTER vs CARICATO"
            ),
        )


        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1],
        )


        fig.update_layout(
            height=400
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # FATTORE DOMINANTE
        # ====================================================

        st.markdown(
            "### Fattore dominante"
        )


        d1, d2 = st.columns(2)


        d1.info(
            "MASTER\n\n"
            f"**{master_ml_stats['dominant']}**\n\n"
            "Importanza: "
            f"**{format_percent(master_ml_stats['importance'])}**"
        )


        d2.info(
            "CARICATO\n\n"
            f"**{current_ml_stats['dominant']}**\n\n"
            "Importanza: "
            f"**{format_percent(current_ml_stats['importance'])}**"
        )


        # ====================================================
        # DISTRIBUZIONE RISCHIO
        # ====================================================

        st.markdown(
            "### Distribuzione del rischio"
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_RISK,
            "Distribuzione del rischio: MASTER vs CARICATO",
        )


        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # ML PROFILE
        # ====================================================

        st.markdown(
            "### Profili ML"
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_PROFILE,
            "ML Profile: MASTER vs CARICATO",
        )


        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # PROFILO CAMPIONE
        # ====================================================

        st.markdown(
            "### Confronto del campione"
        )


        comparison_variables = [

            (
                COL_GENDER,
                "Genere",
            ),

            (
                COL_AGE,
                "Età",
            ),

            (
                COL_MODE,
                "Modalità lavorativa",
            ),

            (
                COL_SECTOR,
                "Settore lavorativo",
            ),

            (
                COL_SENIORITY,
                "Anzianità lavorativa",
            ),

            (
                COL_HOURS,
                "Ore lavorative settimanali",
            ),

            (
                COL_ROLE,
                "Ruolo e responsabilità",
            ),
        ]


        for column, title in comparison_variables:

            fig = comparison_distribution(
                master_df,
                current_df,
                column,
                f"{title}: MASTER vs CARICATO",
            )

            if fig is not None:

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


        # ====================================================
        # RIEPILOGO AUTOMATICO
        # ====================================================

        st.markdown(
            "### Riepilogo delle variazioni"
        )


        if delta_n > 0:

            st.write(
                f"Il dataset caricato contiene "
                f"**{delta_n} rispondenti in più** "
                f"rispetto al MASTER."
            )

        elif delta_n < 0:

            st.write(
                f"Il dataset caricato contiene "
                f"**{abs(delta_n)} rispondenti in meno** "
                f"rispetto al MASTER."
            )

        else:

            st.write(
                "Il numero complessivo di rispondenti "
                "è invariato."
            )


        risk_delta = (
            new_stats["risk_ma"]
            - master_stats["risk_ma"]
        )


        index_delta = (
            new_stats["risk_index"]
            - master_stats["risk_index"]
        )


        st.write(
            "Variazione del **Risk Index medio**: "
            f"**{format_pp(index_delta)}**."
        )


        st.write(
            "Variazione della quota di "
            "**rischio medio-alto**: "
            f"**{format_pp(risk_delta)}**."
        )


        cv_master = safe_float(
            master_ml_stats["cv"]
        )

        cv_new = safe_float(
            current_ml_stats["cv"]
        )


        if (
            cv_master is not None
            and cv_new is not None
        ):

            st.write(
                "Variazione della "
                "**Cross Validation Accuracy**: "
                f"**{format_pp(cv_new - cv_master)}**."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fonte: Survey_Data + ML_Output. "
    "Il MASTER costituisce la baseline di riferimento. "
    "Il file caricato viene analizzato separatamente "
    "e confrontato con la baseline. "
    "La dashboard non modifica i file Excel."
)
