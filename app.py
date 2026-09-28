from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURAZIONE PAGINA
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
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.20);
        padding: 15px 17px;
        border-radius: 16px;
    }

    [data-testid="stMetricValue"] {
        font-weight: 650;
    }

    [data-testid="stMetricLabel"] {
        font-size: .92rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COLONNE SURVEY_DATA
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


RISK_LABELS = {
    "Low_Risk": "Rischio basso",
    "Medium_Risk": "Rischio medio",
    "High_Risk": "Rischio alto",
}


# ============================================================
# LETTURA EXCEL
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


# ============================================================
# PULIZIA DATASET
# ============================================================

def clean_survey(df):
    """
    IMPORTANTE:
    mantiene tutti i rispondenti.

    Non elimina più le righe con età mancante.
    Elimina esclusivamente righe completamente vuote.
    """

    df = df.copy()

    df = df.dropna(
        how="all"
    ).copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


# ============================================================
# FUNZIONI GENERALI
# ============================================================

def safe_float(value):

    try:

        if value is None or pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def format_percent(
    value,
    decimals=1,
):

    value = safe_float(value)

    if value is None:
        return "—"

    return f"{value:.{decimals}%}"


def format_pp(
    value,
    decimals=1,
):

    value = safe_float(value)

    if value is None:
        return "—"

    sign = "+" if value > 0 else ""

    return (
        f"{sign}"
        f"{value * 100:.{decimals}f} p.p."
    )


# ============================================================
# LETTURA ML_OUTPUT
# ============================================================

def ml_value(
    ml,
    key,
    default=None,
):

    if ml is None or ml.empty:
        return default

    for _, row in ml.iterrows():

        if len(row) < 2:
            continue

        current_key = str(
            row.iloc[0]
        ).strip()

        if current_key == key:

            value = row.iloc[1]

            if pd.isna(value):
                return default

            return value

    return default


def ml_value_multi(
    ml,
    keys,
    default=None,
):

    for key in keys:

        value = ml_value(
            ml,
            key,
            None,
        )

        if (
            value is not None
            and pd.notna(value)
        ):
            return value

    return default


# ============================================================
# METRICHE ML
# ============================================================

def get_ml_metrics(ml):

    return {

        "training":
            ml_value_multi(
                ml,
                [
                    "Accuracy_Full",
                    "Accuracy_Full_Training",
                ],
            ),

        "cv":
            ml_value_multi(
                ml,
                [
                    "Accuracy_CV",
                    "Cross-Validation Accuracy",
                ],
            ),

        "dominant":
            ml_value(
                ml,
                "Fattore_Dominante",
                "—",
            ),

        "importance":
            ml_value(
                ml,
                "Importanza_Dominante",
            ),

        "updated":
            ml_value(
                ml,
                "Timestamp_Aggiornamento",
                "—",
            ),

        "respondents":
            ml_value(
                ml,
                "Rispondenti",
            ),

        "risk_low":
            ml_value(
                ml,
                "Rischio_Basso",
            ),

        "risk_ma":
            ml_value(
                ml,
                "Rischio_Medio_Alto",
            ),

        "training_errors":
            ml_value(
                ml,
                "Errori_Training",
            ),
    }


# ============================================================
# METRICHE DATASET
# ============================================================

def dataset_metrics(df):
    """
    Il numero dei rispondenti viene preso direttamente
    dalle righe effettive di Survey_Data.
    """

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

    # --------------------------------------------------------
    # Risk Index
    # --------------------------------------------------------

    if COL_INDEX in df.columns:

        risk_index = pd.to_numeric(
            df[COL_INDEX],
            errors="coerce",
        ).mean()

    else:
        risk_index = None


    # --------------------------------------------------------
    # Rischio medio-alto / basso
    # --------------------------------------------------------

    if COL_RISK in df.columns:

        valid_risk = df[COL_RISK].notna()

        if valid_risk.any():

            risk_ma = (
                df.loc[
                    valid_risk,
                    COL_RISK
                ]
                .isin([
                    "Medium_Risk",
                    "High_Risk",
                ])
                .mean()
            )

            risk_low = (
                df.loc[
                    valid_risk,
                    COL_RISK
                ]
                .eq("Low_Risk")
                .mean()
            )

        else:

            risk_ma = None
            risk_low = None

    else:

        risk_ma = None
        risk_low = None


    # --------------------------------------------------------
    # Dimensioni
    # --------------------------------------------------------

    if COL_EXH in df.columns:

        exhaustion = pd.to_numeric(
            df[COL_EXH],
            errors="coerce",
        ).mean()

    else:
        exhaustion = None


    if COL_DET in df.columns:

        detachment = pd.to_numeric(
            df[COL_DET],
            errors="coerce",
        ).mean()

    else:
        detachment = None


    if COL_REAL in df.columns:

        realization = pd.to_numeric(
            df[COL_REAL],
            errors="coerce",
        ).mean()

    else:
        realization = None


    return {

        "n":
            n,

        "risk_index":
            risk_index,

        "risk_ma":
            risk_ma,

        "risk_low":
            risk_low,

        "exhaustion":
            exhaustion,

        "detachment":
            detachment,

        "realization":
            realization,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(ml):

    try:

        features = ml.iloc[
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
                [
                    "",
                    "nan",
                    "None",
                ]
            )
        ]

        features = (
            features
            .sort_values(
                "Importanza",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return features

    except Exception:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importanza",
            ]
        )


# ============================================================
# MATRICE DI CONFUSIONE
# ============================================================

def get_confusion_matrix(
    ml,
    matrix_type="cv",
):

    try:

        if matrix_type == "training":

            values = [

                [
                    float(
                        ml.iloc[2, 4]
                    ),
                    float(
                        ml.iloc[2, 5]
                    ),
                ],

                [
                    float(
                        ml.iloc[3, 4]
                    ),
                    float(
                        ml.iloc[3, 5]
                    ),
                ],
            ]

        else:

            values = [

                [
                    float(
                        ml.iloc[7, 4]
                    ),
                    float(
                        ml.iloc[7, 5]
                    ),
                ],

                [
                    float(
                        ml.iloc[8, 4]
                    ),
                    float(
                        ml.iloc[8, 5]
                    ),
                ],
            ]

        return pd.DataFrame(
            values,

            index=[
                "Reale Basso",
                "Reale Medio-Alto",
            ],

            columns=[
                "Pred. Basso",
                "Pred. Medio-Alto",
            ],
        )

    except Exception:

        return None


# ============================================================
# GRAFICO DISTRIBUZIONE
# ============================================================

def distribution_chart(
    df,
    column,
    title,
    horizontal=False,
):

    if (
        column not in df.columns
        or df.empty
    ):
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

        height = max(
            390,
            len(counts) * 42,
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

        height = 390

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=40,
        ),
    )

    return fig


# ============================================================
# GRAFICO CONFRONTO
# ============================================================

def comparison_distribution(
    master_df,
    new_df,
    column,
    title,
    category_map=None,
    horizontal=False,
):

    if (
        column not in master_df.columns
        or column not in new_df.columns
    ):
        return None

    master_series = (
        master_df[column]
        .fillna("Non specificato")
        .astype(str)
    )

    new_series = (
        new_df[column]
        .fillna("Non specificato")
        .astype(str)
    )

    if category_map:

        master_series = (
            master_series.replace(
                category_map
            )
        )

        new_series = (
            new_series.replace(
                category_map
            )
        )

    master_pct = (
        master_series
        .value_counts(
            normalize=True
        )
        .mul(100)
    )

    new_pct = (
        new_series
        .value_counts(
            normalize=True
        )
        .mul(100)
    )

    categories = sorted(
        set(master_pct.index)
        .union(
            set(new_pct.index)
        )
    )

    # IMPORTANTE:
    # creiamo Categoria esplicitamente.
    # Evita il precedente KeyError.

    comp = pd.DataFrame({

        "Categoria":
            categories,

        "MASTER":
            [
                float(
                    master_pct.get(
                        x,
                        0,
                    )
                )
                for x in categories
            ],

        "CARICATO":
            [
                float(
                    new_pct.get(
                        x,
                        0,
                    )
                )
                for x in categories
            ],
    })

    long_df = comp.melt(
        id_vars=[
            "Categoria"
        ],

        value_vars=[
            "MASTER",
            "CARICATO",
        ],

        var_name="Dataset",

        value_name="Percentuale",
    )

    if horizontal:

        fig = px.bar(
            long_df,
            x="Percentuale",
            y="Categoria",
            color="Dataset",
            barmode="group",
            orientation="h",
            text="Percentuale",
            title=title,
        )

        fig.update_traces(
            texttemplate="%{x:.1f}%"
        )

        fig.update_xaxes(
            ticksuffix="%"
        )

        height = max(
            400,
            len(categories) * 50,
        )

    else:

        fig = px.bar(
            long_df,
            x="Categoria",
            y="Percentuale",
            color="Dataset",
            barmode="group",
            text="Percentuale",
            title=title,
        )

        fig.update_traces(
            texttemplate="%{y:.1f}%"
        )

        fig.update_yaxes(
            ticksuffix="%"
        )

        fig.update_xaxes(
            tickangle=-20
        )

        height = 430

    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=50,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title(
    "NOT SUITABLE FOR WORK"
)

st.caption(
    "Dashboard interattiva • GDG Palermo • "
    "MASTER baseline + confronto dataset"
)


# ============================================================
# CARICAMENTO MASTER
# ============================================================

try:

    master_survey_raw, master_ml = (
        load_excel(
            DEFAULT_FILE
        )
    )

except Exception as e:

    st.error(
        "Impossibile leggere il MASTER: "
        f"{e}"
    )

    st.stop()


# ============================================================
# PULIZIA MASTER
# ============================================================

master_df = clean_survey(
    master_survey_raw
)


master_stats = dataset_metrics(
    master_df
)


master_ml_stats = get_ml_metrics(
    master_ml
)


# ============================================================
# UPLOAD DATASET
# ============================================================

uploaded = st.sidebar.file_uploader(
    "Carica dataset aggiornato",
    type=["xlsx"],
    help=(
        "Il MASTER rimane la baseline. "
        "Il nuovo file viene confrontato "
        "con il MASTER."
    ),
)


has_upload = (
    uploaded is not None
)


if has_upload:

    try:

        uploaded.seek(0)

        current_survey_raw, current_ml = (
            load_excel(
                uploaded
            )
        )

        current_df = clean_survey(
            current_survey_raw
        )

        current_name = (
            uploaded.name
        )

    except Exception as e:

        st.error(
            "Errore nel file caricato: "
            f"{e}"
        )

        st.stop()

else:

    current_survey_raw = (
        master_survey_raw.copy()
    )

    current_df = (
        master_df.copy()
    )

    current_ml = (
        master_ml.copy()
    )

    current_name = (
        DEFAULT_FILE.name
    )


current_ml_stats = get_ml_metrics(
    current_ml
)


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

    if col
    not in current_df.columns
]


if missing:

    st.error(
        "Il dataset caricato non è "
        "compatibile con il MASTER."
    )

    st.write(
        "Colonne mancanti:",
        missing,
    )

    st.stop()


# ============================================================
# SIDEBAR — CONTROLLO RIGHE
# ============================================================

st.sidebar.divider()


st.sidebar.subheader(
    "Controllo dataset"
)


st.sidebar.metric(
    "Righe MASTER Excel",
    len(master_survey_raw),
)


st.sidebar.metric(
    "Rispondenti MASTER",
    len(master_df),
)


if has_upload:

    st.sidebar.metric(
        "Righe file caricato",
        len(current_survey_raw),
    )

    st.sidebar.metric(
        "Rispondenti caricati",
        len(current_df),
    )


# ============================================================
# CONTROLLO ETÀ MANCANTI
# ============================================================

if COL_AGE in master_df.columns:

    master_missing_age = (
        master_df[
            COL_AGE
        ]
        .isna()
        .sum()
    )

    if master_missing_age > 0:

        st.sidebar.caption(
            f"Età mancanti nel MASTER: "
            f"{master_missing_age}. "
            "Queste righe NON vengono eliminate."
        )


# ============================================================
# STATO DATASET
# ============================================================

st.sidebar.divider()


if has_upload:

    st.sidebar.success(
        "Confronto attivo"
    )

    st.sidebar.caption(
        f"MASTER: "
        f"{len(master_df)} rispondenti"
    )

    st.sidebar.caption(
        f"CARICATO: "
        f"{len(current_df)} rispondenti"
    )

else:

    st.sidebar.info(
        "Visualizzazione MASTER"
    )


# ============================================================
# FILTRI
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "Filtri"
)


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


filtered = (
    current_df.copy()
)


for label, col in filter_cols.items():

    if col not in filtered.columns:
        continue

    options = sorted(
        filtered[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected = (
        st.sidebar.multiselect(
            label,
            options,
            placeholder="Tutti",
        )
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
    "Risposte visualizzate: "
    f"{len(filtered)} / "
    f"{len(current_df)}"
)


# ============================================================
# KPI HEADER
# ============================================================

filtered_stats = dataset_metrics(
    filtered
)


k1, k2, k3, k4 = (
    st.columns(4)
)


k1.metric(
    "Rispondenti",
    filtered_stats["n"],
)


k2.metric(
    "Risk Index medio",
    format_percent(
        filtered_stats[
            "risk_index"
        ]
    ),
)


k3.metric(
    "Rischio medio-alto",
    format_percent(
        filtered_stats[
            "risk_ma"
        ]
    ),
)


k4.metric(
    "Accuracy CV",
    format_percent(
        current_ml_stats[
            "cv"
        ]
    ),
)


st.caption(
    "I primi tre KPI rispettano i filtri attivi. "
    "Il numero dei rispondenti deriva direttamente "
    "dalle righe di Survey_Data."
)


# ============================================================
# TAB
# ============================================================

tab1, tab2, tab3, tab4 = (
    st.tabs(
        [
            "Overview",
            "Profilo del campione",
            "Random Forest",
            "Confronto MASTER",
        ]
    )
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Quadro generale"
    )


    # --------------------------------------------------------
    # RISCHIO + ML PROFILE
    # --------------------------------------------------------

    c1, c2 = (
        st.columns(2)
    )


    risk_counts = (
        filtered[COL_RISK]
        .replace(
            RISK_LABELS
        )
        .fillna(
            "Non specificato"
        )
        .value_counts()
        .rename_axis(
            "Rischio"
        )
        .reset_index(
            name="N"
        )
    )


    if not risk_counts.empty:

        fig = px.pie(
            risk_counts,
            names="Rischio",
            values="N",
            hole=.58,
            title=(
                "Distribuzione "
                "del rischio"
            ),
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig.update_layout(
            height=390
        )

        c1.plotly_chart(
            fig,
            use_container_width=True,
        )


    profile_counts = (
        filtered[COL_PROFILE]
        .fillna(
            "Non specificato"
        )
        .value_counts()
        .rename_axis(
            "Profilo"
        )
        .reset_index(
            name="N"
        )
    )


    if not profile_counts.empty:

        fig = px.bar(
            profile_counts,
            x="Profilo",
            y="N",
            text="N",
            title=(
                "Distribuzione "
                "ML Profile"
            ),
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=390
        )

        c2.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # SETTORE + MODALITÀ
    # --------------------------------------------------------

    c3, c4 = (
        st.columns(2)
    )


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


    # --------------------------------------------------------
    # DIMENSIONI RISCHIO
    # --------------------------------------------------------

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
                filtered_stats[
                    "exhaustion"
                ],
                filtered_stats[
                    "detachment"
                ],
                filtered_stats[
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
        range_y=[
            0,
            1,
        ],
        title=(
            "Indice medio "
            "per dimensione"
        ),
    )


    fig.update_yaxes(
        tickformat=".0%"
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


    a, b = (
        st.columns(2)
    )


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


    a, b = (
        st.columns(2)
    )


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
            filtered[
                visible_cols
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 3 — RANDOM FOREST
# ============================================================

with tab3:

    st.subheader(
        "Random Forest"
    )


    m1, m2, m3, m4 = (
        st.columns(4)
    )


    m1.metric(
        "Accuracy training",
        format_percent(
            current_ml_stats[
                "training"
            ]
        ),
    )


    m2.metric(
        "5-Fold CV",
        format_percent(
            current_ml_stats[
                "cv"
            ]
        ),
    )


    m3.metric(
        "Importanza dominante",
        format_percent(
            current_ml_stats[
                "importance"
            ]
        ),
    )


    respondents_ml = safe_float(
        current_ml_stats[
            "respondents"
        ]
    )


    m4.metric(
        "Campione ML",
        (
            int(respondents_ml)
            if respondents_ml is not None
            else "—"
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

    perf_rows = []


    training_value = safe_float(
        current_ml_stats[
            "training"
        ]
    )


    cv_value = safe_float(
        current_ml_stats[
            "cv"
        ]
    )


    if training_value is not None:

        perf_rows.append(
            {
                "Metrica":
                    "Training",

                "Accuracy":
                    training_value,
            }
        )


    if cv_value is not None:

        perf_rows.append(
            {
                "Metrica":
                    "5-Fold CV",

                "Accuracy":
                    cv_value,
            }
        )


    if perf_rows:

        perf = pd.DataFrame(
            perf_rows
        )


        fig = px.bar(
            perf,
            x="Metrica",
            y="Accuracy",
            text_auto=".1%",
            range_y=[
                0,
                1,
            ],
            title=(
                "Training vs "
                "Cross Validation"
            ),
        )


        fig.update_yaxes(
            tickformat=".0%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # MATRICE TRAINING
    # --------------------------------------------------------

    st.markdown(
        "#### Matrice di confusione — Training"
    )


    cm_training = get_confusion_matrix(
        current_ml,
        "training",
    )


    if cm_training is not None:

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

    else:

        st.warning(
            "Matrice training "
            "non disponibile."
        )


    # --------------------------------------------------------
    # MATRICE CV
    # --------------------------------------------------------

    st.markdown(
        "#### Matrice di confusione — 5-Fold CV"
    )


    cm_cv = get_confusion_matrix(
        current_ml,
        "cv",
    )


    if cm_cv is not None:

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

    else:

        st.warning(
            "Matrice CV "
            "non disponibile."
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "#### Feature importance"
    )


    features = get_feature_importance(
        current_ml
    )


    if not features.empty:

        features_plot = (
            features
            .head(15)
            .sort_values(
                "Importanza",
                ascending=True,
            )
        )


        fig = px.bar(
            features_plot,
            x="Importanza",
            y="Feature",
            orientation="h",
            text_auto=".1%",
            title=(
                "Top feature "
                "del modello"
            ),
        )


        fig.update_xaxes(
            tickformat=".0%"
        )


        fig.update_layout(
            height=max(
                450,
                len(features_plot)
                * 40,
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# TAB 4 — CONFRONTO MASTER
# ============================================================

with tab4:

    st.subheader(
        "Confronto con il MASTER di partenza"
    )


    # ========================================================
    # NESSUN FILE CARICATO
    # ========================================================

    if not has_upload:

        st.info(
            "Il MASTER costituisce la baseline "
            "di partenza. Carica un nuovo Excel "
            "per attivare il confronto."
        )


        st.markdown(
            "### Baseline iniziale"
        )


        b1, b2, b3, b4 = (
            st.columns(4)
        )


        b1.metric(
            "Rispondenti",
            master_stats["n"],
        )


        b2.metric(
            "Accuracy training",
            format_percent(
                master_ml_stats[
                    "training"
                ]
            ),
        )


        b3.metric(
            "Accuracy CV",
            format_percent(
                master_ml_stats[
                    "cv"
                ]
            ),
        )


        b4.metric(
            "Importanza dominante",
            format_percent(
                master_ml_stats[
                    "importance"
                ]
            ),
        )


        st.info(
            "Fattore dominante della baseline: "
            f"**{master_ml_stats['dominant']}**"
        )


    # ========================================================
    # FILE CARICATO
    # ========================================================

    else:

        new_stats = dataset_metrics(
            current_df
        )


        # ====================================================
        # DIMENSIONE CAMPIONE
        # ====================================================

        st.markdown(
            "### Dimensione del campione"
        )


        delta_n = (
            new_stats["n"]
            - master_stats["n"]
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        c1.metric(
            "MASTER",
            master_stats[
                "n"
            ],
        )


        c2.metric(
            "CARICATO",
            new_stats[
                "n"
            ],
        )


        c3.metric(
            "Variazione",
            f"{delta_n:+d}",
        )


        # ====================================================
        # INDICATORI DI RISCHIO
        # ====================================================

        st.markdown(
            "### Indicatori di rischio"
        )


        comparison = pd.DataFrame(
            [
                {
                    "Indicatore":
                        "Risk Index medio",

                    "MASTER":
                        master_stats[
                            "risk_index"
                        ],

                    "CARICATO":
                        new_stats[
                            "risk_index"
                        ],
                },

                {
                    "Indicatore":
                        "Rischio medio-alto",

                    "MASTER":
                        master_stats[
                            "risk_ma"
                        ],

                    "CARICATO":
                        new_stats[
                            "risk_ma"
                        ],
                },

                {
                    "Indicatore":
                        "Rischio basso",

                    "MASTER":
                        master_stats[
                            "risk_low"
                        ],

                    "CARICATO":
                        new_stats[
                            "risk_low"
                        ],
                },

                {
                    "Indicatore":
                        "Esaurimento emotivo",

                    "MASTER":
                        master_stats[
                            "exhaustion"
                        ],

                    "CARICATO":
                        new_stats[
                            "exhaustion"
                        ],
                },

                {
                    "Indicatore":
                        "Distacco",

                    "MASTER":
                        master_stats[
                            "detachment"
                        ],

                    "CARICATO":
                        new_stats[
                            "detachment"
                        ],
                },

                {
                    "Indicatore":
                        "Bassa realizzazione",

                    "MASTER":
                        master_stats[
                            "realization"
                        ],

                    "CARICATO":
                        new_stats[
                            "realization"
                        ],
                },
            ]
        )


        comparison[
            "Delta"
        ] = (
            comparison[
                "CARICATO"
            ]
            - comparison[
                "MASTER"
            ]
        )


        display = (
            comparison.copy()
        )


        display[
            "MASTER"
        ] = (
            display[
                "MASTER"
            ]
            .apply(
                format_percent
            )
        )


        display[
            "CARICATO"
        ] = (
            display[
                "CARICATO"
            ]
            .apply(
                format_percent
            )
        )


        display[
            "Delta"
        ] = (
            comparison[
                "Delta"
            ]
            .apply(
                format_pp
            )
        )


        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------
        # GRAFICO INDICATORI
        # ----------------------------------------------------

        long_df = (
            comparison.melt(
                id_vars=[
                    "Indicatore"
                ],

                value_vars=[
                    "MASTER",
                    "CARICATO",
                ],

                var_name="Dataset",

                value_name="Valore",
            )
        )


        fig = px.bar(
            long_df,
            x="Indicatore",
            y="Valore",
            color="Dataset",
            barmode="group",
            text_auto=".1%",
            title=(
                "Indicatori di rischio: "
                "MASTER vs CARICATO"
            ),
        )


        fig.update_yaxes(
            tickformat=".0%",
            range=[
                0,
                1,
            ],
        )


        fig.update_xaxes(
            tickangle=-20
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        st.markdown(
            "### Random Forest"
        )


        rf_compare = pd.DataFrame(
            [
                {
                    "Metrica":
                        "Accuracy training",

                    "MASTER":
                        safe_float(
                            master_ml_stats[
                                "training"
                            ]
                        ),

                    "CARICATO":
                        safe_float(
                            current_ml_stats[
                                "training"
                            ]
                        ),
                },

                {
                    "Metrica":
                        "Accuracy CV",

                    "MASTER":
                        safe_float(
                            master_ml_stats[
                                "cv"
                            ]
                        ),

                    "CARICATO":
                        safe_float(
                            current_ml_stats[
                                "cv"
                            ]
                        ),
                },

                {
                    "Metrica":
                        "Importanza dominante",

                    "MASTER":
                        safe_float(
                            master_ml_stats[
                                "importance"
                            ]
                        ),

                    "CARICATO":
                        safe_float(
                            current_ml_stats[
                                "importance"
                            ]
                        ),
                },
            ]
        )


        rf_compare[
            "Delta"
        ] = (
            rf_compare[
                "CARICATO"
            ]
            - rf_compare[
                "MASTER"
            ]
        )


        rf_display = (
            rf_compare.copy()
        )


        rf_display[
            "MASTER"
        ] = (
            rf_display[
                "MASTER"
            ]
            .apply(
                format_percent
            )
        )


        rf_display[
            "CARICATO"
        ] = (
            rf_display[
                "CARICATO"
            ]
            .apply(
                format_percent
            )
        )


        rf_display[
            "Delta"
        ] = (
            rf_compare[
                "Delta"
            ]
            .apply(
                format_pp
            )
        )


        st.dataframe(
            rf_display,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # FATTORE DOMINANTE
        # ====================================================

        st.markdown(
            "### Fattore dominante"
        )


        f1, f2 = (
            st.columns(2)
        )


        f1.info(
            "MASTER\n\n"
            f"**{master_ml_stats['dominant']}**\n\n"
            "Importanza: "
            f"**{format_percent(master_ml_stats['importance'])}**"
        )


        f2.info(
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
            "MASTER vs CARICATO",
            category_map=RISK_LABELS,
        )


        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # CLUSTER / ML PROFILE
        # ====================================================

        st.markdown(
            "### Cluster / ML Profile"
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_PROFILE,
            (
                "Distribuzione ML Profile: "
                "MASTER vs CARICATO"
            ),
        )


        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # COMPOSIZIONE CAMPIONE
        # ====================================================

        st.markdown(
            "### Composizione del campione"
        )


        row1a, row1b = (
            st.columns(2)
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_GENDER,
            "Genere",
        )


        if fig is not None:

            row1a.plotly_chart(
                fig,
                use_container_width=True,
            )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_MODE,
            "Modalità lavorativa",
        )


        if fig is not None:

            row1b.plotly_chart(
                fig,
                use_container_width=True,
            )


        row2a, row2b = (
            st.columns(2)
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_AGE,
            "Età",
        )


        if fig is not None:

            row2a.plotly_chart(
                fig,
                use_container_width=True,
            )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_SENIORITY,
            "Anzianità lavorativa",
            horizontal=True,
        )


        if fig is not None:

            row2b.plotly_chart(
                fig,
                use_container_width=True,
            )


        row3a, row3b = (
            st.columns(2)
        )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_HOURS,
            "Ore settimanali",
            horizontal=True,
        )


        if fig is not None:

            row3a.plotly_chart(
                fig,
                use_container_width=True,
            )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_SECTOR,
            "Settore lavorativo",
            horizontal=True,
        )


        if fig is not None:

            row3b.plotly_chart(
                fig,
                use_container_width=True,
            )


        fig = comparison_distribution(
            master_df,
            current_df,
            COL_ROLE,
            "Ruolo e responsabilità",
            horizontal=True,
        )


        if fig is not None:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        st.markdown(
            "### Evoluzione delle feature"
        )


        master_features = (
            get_feature_importance(
                master_ml
            )
        )


        new_features = (
            get_feature_importance(
                current_ml
            )
        )


        if (
            not master_features.empty
            and not new_features.empty
        ):

            feature_compare = (
                pd.merge(
                    master_features,
                    new_features,
                    on="Feature",
                    how="outer",
                    suffixes=(
                        "_MASTER",
                        "_CARICATO",
                    ),
                )
                .fillna(0)
            )


            feature_compare[
                "Delta"
            ] = (
                feature_compare[
                    "Importanza_CARICATO"
                ]
                - feature_compare[
                    "Importanza_MASTER"
                ]
            )


            feature_compare = (
                feature_compare
                .sort_values(
                    "Importanza_CARICATO",
                    ascending=False,
                )
            )


            top_features = (
                feature_compare
                .head(15)
                .copy()
            )


            feature_long = (
                top_features.melt(
                    id_vars=[
                        "Feature"
                    ],

                    value_vars=[
                        "Importanza_MASTER",
                        "Importanza_CARICATO",
                    ],

                    var_name="Dataset",

                    value_name="Importanza",
                )
            )


            feature_long[
                "Dataset"
            ] = (
                feature_long[
                    "Dataset"
                ]
                .replace(
                    {
                        "Importanza_MASTER":
                            "MASTER",

                        "Importanza_CARICATO":
                            "CARICATO",
                    }
                )
            )


            fig = px.bar(
                feature_long,
                x="Importanza",
                y="Feature",
                color="Dataset",
                barmode="group",
                orientation="h",
                text_auto=".1%",
                title=(
                    "Feature importance: "
                    "MASTER vs CARICATO"
                ),
            )


            fig.update_xaxes(
                tickformat=".0%"
            )


            fig.update_layout(
                height=max(
                    500,
                    len(top_features)
                    * 45,
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ====================================================
        # RIEPILOGO
        # ====================================================

        st.markdown(
            "### Variazioni principali"
        )


        if delta_n > 0:

            st.write(
                "Il dataset caricato contiene "
                f"**{delta_n} rispondenti in più** "
                "rispetto al MASTER."
            )

        elif delta_n < 0:

            st.write(
                "Il dataset caricato contiene "
                f"**{abs(delta_n)} rispondenti in meno** "
                "rispetto al MASTER."
            )

        else:

            st.write(
                "Il numero di rispondenti "
                "è **invariato**."
            )


        # ----------------------------------------------------
        # Risk Index
        # ----------------------------------------------------

        master_index = safe_float(
            master_stats[
                "risk_index"
            ]
        )

        new_index = safe_float(
            new_stats[
                "risk_index"
            ]
        )


        if (
            master_index is not None
            and new_index is not None
        ):

            st.write(
                "Risk Index medio: "
                f"**{format_pp(new_index - master_index)}**."
            )


        # ----------------------------------------------------
        # Rischio medio-alto
        # ----------------------------------------------------

        master_risk = safe_float(
            master_stats[
                "risk_ma"
            ]
        )

        new_risk = safe_float(
            new_stats[
                "risk_ma"
            ]
        )


        if (
            master_risk is not None
            and new_risk is not None
        ):

            st.write(
                "Quota rischio medio-alto: "
                f"**{format_pp(new_risk - master_risk)}**."
            )


        # ----------------------------------------------------
        # CV
        # ----------------------------------------------------

        master_cv = safe_float(
            master_ml_stats[
                "cv"
            ]
        )

        new_cv = safe_float(
            current_ml_stats[
                "cv"
            ]
        )


        if (
            master_cv is not None
            and new_cv is not None
        ):

            st.write(
                "Accuracy CV: "
                f"**{format_pp(new_cv - master_cv)}**."
            )


        # ----------------------------------------------------
        # Fattore dominante
        # ----------------------------------------------------

        if (
            str(
                master_ml_stats[
                    "dominant"
                ]
            )
            ==
            str(
                current_ml_stats[
                    "dominant"
                ]
            )
        ):

            st.write(
                "Fattore dominante: "
                "**invariato** — "
                f"{current_ml_stats['dominant']}."
            )

        else:

            st.write(
                "Fattore dominante: da "
                f"**{master_ml_stats['dominant']}** "
                "a "
                f"**{current_ml_stats['dominant']}**."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Fonte: Survey_Data + ML_Output. "
    "Il MASTER costituisce la baseline fissa. "
    "Le righe con valori mancanti in singole variabili "
    "non vengono eliminate dal conteggio dei rispondenti."
)
