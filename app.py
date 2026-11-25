"""
Dashboard California Housing Prices
Jalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="Eksplorasi & Prediksi California Housing Prices",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).parent / "housing(3).csv"
TARGET = "median_house_value"
NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income"
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Identitas Kelompok
GROUP_NAME = "Kelompok 2"
CLASS_NAME = "05TPLP010"
GROUP_MEMBERS = [
    ("Noval Athilah Akbar", "241011401404"),
    ("Reysa Eka Saputra", "241011401989"),
    ("Rizka Rasya Nayara", "241011402333"),
]
LECTURER = "Farizi Ilham, S.Kom., M.Kom."
COURSE_TITLE = "Machine Learning"
STUDY_PROGRAM = "Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Pamulang"

# Palet Biru Navy - Hijau untuk Grafik
NAVY_GREEN_PALETTE = [
    "#1f3a5a",  # biru navy utama
    "#3f7d64",  # hijau sage/forest utama
    "#3b6998",  # biru navy medium
    "#68a68c",  # hijau mint/sage lembut
    "#152840",  # navy gelap
    "#2a5e49",  # hijau tua
    "#7da3c7",  # biru navy muda
    "#9bcbb6",  # hijau muda
]

# Tema Warna Dinamis Sidebar & Aksen per Menu (Biru Navy & Hijau)
MENU_THEMES = {
    "Beranda": {
        "sb_top": "#d8e7f5",
        "sb_bot": "#dff1e8",
        "sb_border": "#bfd4e6",
        "accent_navy": "#1f3a5a",
        "accent_green": "#3f7d64",
        "active_bg": "linear-gradient(90deg, #1f3a5a 0%, #3f7d64 100%)",
        "app_bg": "linear-gradient(160deg, #f2f7fb 0%, #f1f9f5 55%, #eef5fa 100%)",
        "label_color": "#265443",
    },
    "Tentang Dataset": {
        "sb_top": "#cfe2f3",
        "sb_bot": "#d4ece6",
        "sb_border": "#b0cadf",
        "accent_navy": "#193659",
        "accent_green": "#2b7a68",
        "active_bg": "linear-gradient(90deg, #193659 0%, #2b7a68 100%)",
        "app_bg": "linear-gradient(160deg, #eef6fc 0%, #edf8f5 55%, #eaf3fa 100%)",
        "label_color": "#1f594c",
    },
    "Eksplorasi Data (EDA)": {
        "sb_top": "#cbdcf2",
        "sb_bot": "#d1ede1",
        "sb_border": "#a9c4dc",
        "accent_navy": "#1b3358",
        "accent_green": "#2e7d52",
        "active_bg": "linear-gradient(90deg, #1b3358 0%, #2e7d52 100%)",
        "app_bg": "linear-gradient(160deg, #edf4fc 0%, #ecf9f2 55%, #e9f2fa 100%)",
        "label_color": "#225c3c",
    },
    "Preprocessing": {
        "sb_top": "#d3e4ed",
        "sb_bot": "#d7f0e6",
        "sb_border": "#b3ccd9",
        "accent_navy": "#1d3f57",
        "accent_green": "#368068",
        "active_bg": "linear-gradient(90deg, #1d3f57 0%, #368068 100%)",
        "app_bg": "linear-gradient(160deg, #eff7fa 0%, #eef9f4 55%, #ebf5f9 100%)",
        "label_color": "#255e4c",
    },
    "Reduksi Dimensi (PCA)": {
        "sb_top": "#c7dcf0",
        "sb_bot": "#cbe7dc",
        "sb_border": "#a3c0da",
        "accent_navy": "#152f4d",
        "accent_green": "#266b52",
        "active_bg": "linear-gradient(90deg, #152f4d 0%, #266b52 100%)",
        "app_bg": "linear-gradient(160deg, #ebf3fa 0%, #eaf6f1 55%, #e6f0f8 100%)",
        "label_color": "#1c4f3c",
    },
    "Modelling & Evaluasi": {
        "sb_top": "#c4d7eb",
        "sb_bot": "#c7e5d6",
        "sb_border": "#9ebbd6",
        "accent_navy": "#132844",
        "accent_green": "#236449",
        "active_bg": "linear-gradient(90deg, #132844 0%, #236449 100%)",
        "app_bg": "linear-gradient(160deg, #eaf1f8 0%, #e8f5ef 55%, #e4eef7 100%)",
        "label_color": "#194a36",
    },
    "Prediksi Satu Rumah": {
        "sb_top": "#d0e5f2",
        "sb_bot": "#d2efe8",
        "sb_border": "#abc9dc",
        "accent_navy": "#183b5c",
        "accent_green": "#2f7e6c",
        "active_bg": "linear-gradient(90deg, #183b5c 0%, #2f7e6c 100%)",
        "app_bg": "linear-gradient(160deg, #eef6fb 0%, #ecf8f5 55%, #e8f3f9 100%)",
        "label_color": "#215c4e",
    },
    "Prediksi Massal (CSV)": {
        "sb_top": "#cde1ee",
        "sb_bot": "#cdebe0",
        "sb_border": "#a7c4d6",
        "accent_navy": "#1b3952",
        "accent_green": "#2d775b",
        "active_bg": "linear-gradient(90deg, #1b3952 0%, #2d775b 100%)",
        "app_bg": "linear-gradient(160deg, #edf5fa 0%, #ecf8f3 55%, #e7f2f8 100%)",
        "label_color": "#205743",
    },
    "Kesimpulan & Saran": {
        "sb_top": "#cadbe7",
        "sb_bot": "#d0e8da",
        "sb_border": "#a6bed0",
        "accent_navy": "#173147",
        "accent_green": "#336e4f",
        "active_bg": "linear-gradient(90deg, #173147 0%, #336e4f 100%)",
        "app_bg": "linear-gradient(160deg, #edf3f8 0%, #edf7f1 55%, #e8eff5 100%)",
        "label_color": "#25523b",
    },
}

NAV_SECTIONS = {
    "Pengantar": [
        "Beranda",
        "Tentang Dataset",
        "Eksplorasi Data (EDA)",
    ],
    "Pengolahan & Model": [
        "Preprocessing",
        "Reduksi Dimensi (PCA)",
        "Modelling & Evaluasi",
    ],
    "Penerapan": [
        "Prediksi Satu Rumah",
        "Prediksi Massal (CSV)",
        "Kesimpulan & Saran",
    ],
}
ALL_PAGES = [page for pages in NAV_SECTIONS.values() for page in pages]

if "active_menu" not in st.session_state or st.session_state.active_menu not in ALL_PAGES:
    st.session_state.active_menu = "Beranda"

menu = st.session_state.active_menu
theme = MENU_THEMES.get(menu, MENU_THEMES["Beranda"])

# =========================================================
# STYLE DINAMIS (BERUBAH WARNA SAAT PINDAH MENU)
# =========================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background: {theme["app_bg"]};
        color: #182b3e;
        transition: background 0.3s ease;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {theme["sb_top"]} 0%, {theme["sb_bot"]} 100%);
        border-right: 1px solid {theme["sb_border"]};
        transition: background 0.3s ease, border-color 0.3s ease;
    }}
    [data-testid="stSidebar"] * {{
        color: #182b3e;
    }}

    .sidebar-brand {{
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(6px);
        border: 1px solid {theme["sb_border"]};
        border-left: 4px solid {theme["accent_navy"]};
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(24, 43, 62, 0.07);
    }}
    .sidebar-brand-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {theme["accent_navy"]};
        margin: 0 0 3px 0;
    }}
    .sidebar-brand-sub {{
        font-size: 0.77rem;
        color: {theme["label_color"]};
        font-weight: 600;
        margin: 0;
    }}

    .sidebar-group-label {{
        font-size: 0.71rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {theme["label_color"]};
        margin: 12px 0 6px 2px;
    }}

    /* Tombol menu sidebar tanpa bulet-bulet */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background: rgba(255, 255, 255, 0.56);
        color: #182b3e;
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 4px;
        font-size: 0.88rem;
        font-weight: 500;
        transition: all 0.18s ease;
        box-shadow: none;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.94);
        border-color: {theme["accent_green"]};
        color: {theme["accent_navy"]};
        transform: translateX(2px);
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {theme["active_bg"]};
        color: #ffffff;
        border: 1px solid transparent;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(24, 43, 62, 0.16);
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        opacity: 0.96;
        color: #ffffff;
        transform: none;
    }}

    .hero {{
        background: linear-gradient(120deg, {theme["sb_top"]} 0%, {theme["sb_bot"]} 62%, #f4f9fc 100%);
        border: 1px solid {theme["sb_border"]};
        border-radius: 20px;
        padding: 30px 34px;
        margin-bottom: 22px;
        box-shadow: 0 10px 26px rgba(24, 43, 62, 0.08);
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid {theme["accent_green"]};
        color: {theme["accent_green"]};
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 11px;
    }}
    .hero h1 {{
        color: {theme["accent_navy"]};
        font-size: 2.3rem;
        line-height: 1.15;
        margin: 0 0 8px 0;
        font-weight: 700;
    }}
    .hero p {{
        color: #425b70;
        margin: 0;
        font-size: 1.02rem;
    }}

    .card-grid-4 {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        align-items: stretch;
        margin: 8px 0 6px 0;
    }}
    .card-grid-3 {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        align-items: stretch;
        margin: 8px 0 14px 0;
    }}
    .card-grid-2 {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        align-items: stretch;
        margin: 8px 0 14px 0;
    }}
    @media (max-width: 1024px) {{
        .card-grid-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .card-grid-3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
        .card-grid-4, .card-grid-3, .card-grid-2 {{ grid-template-columns: 1fr; }}
    }}

    .info-card {{
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid {theme["sb_border"]};
        border-radius: 16px;
        padding: 18px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 6px 18px rgba(24, 43, 62, 0.05);
        transition: transform 0.16s ease, border-color 0.16s ease;
    }}
    .info-card:hover {{
        transform: translateY(-2px);
        border-color: {theme["accent_green"]};
    }}
    .info-card h4 {{
        margin: 10px 0 6px 0;
        color: {theme["accent_navy"]};
        font-size: 1.03rem;
        font-weight: 700;
        line-height: 1.3;
    }}
    .info-card p {{
        margin: 0;
        font-size: 0.86rem;
        color: #4c6479;
        line-height: 1.48;
    }}
    .card-footer-tag {{
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px dashed #d6e3ee;
        font-size: 0.78rem;
        font-weight: 600;
        color: {theme["accent_green"]};
    }}

    .step-badge {{
        display: inline-block;
        width: fit-content;
        background: #dce8f4;
        border: 1px solid #b6cde3;
        color: {theme["accent_navy"]};
        font-weight: 700;
        font-size: 0.73rem;
        padding: 3px 10px;
        border-radius: 999px;
    }}
    .step-badge-green {{
        display: inline-block;
        width: fit-content;
        background: #dcf0e7;
        border: 1px solid #aed7c4;
        color: {theme["accent_green"]};
        font-weight: 700;
        font-size: 0.73rem;
        padding: 3px 10px;
        border-radius: 999px;
    }}

    .academic-banner {{
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid {theme["sb_border"]};
        border-left: 4px solid {theme["accent_navy"]};
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px 28px;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(24, 43, 62, 0.04);
    }}
    .academic-item {{
        font-size: 0.88rem;
        color: #2b4257;
    }}
    .academic-item strong {{
        color: {theme["accent_navy"]};
    }}

    .insight-box {{
        background: linear-gradient(90deg, {theme["sb_top"]}, {theme["sb_bot"]});
        border-left: 4px solid {theme["accent_green"]};
        border-top: 1px solid {theme["sb_border"]};
        border-right: 1px solid {theme["sb_border"]};
        border-bottom: 1px solid {theme["sb_border"]};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
        color: #1b3044;
        line-height: 1.55;
    }}
    .insight-box strong {{
        color: {theme["accent_navy"]};
    }}

    .result-banner {{
        background: linear-gradient(120deg, {theme["sb_top"]} 0%, {theme["sb_bot"]} 100%);
        border: 1px solid {theme["accent_green"]};
        border-radius: 16px;
        padding: 20px 24px;
        margin-top: 12px;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid {theme["sb_border"]};
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(24, 43, 62, 0.05);
    }}
    div[data-testid="stMetric"] label {{
        color: #4c657b;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {theme["sb_border"]};
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.85);
    }}

    /* Tombol utama di area konten (bukan sidebar) */
    .stMain .stButton > button[kind="primary"],
    .stMain .stFormSubmitButton > button[kind="primary"] {{
        background: {theme["active_bg"]};
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }}
    .stMain .stButton > button[kind="primary"]:hover,
    .stMain .stFormSubmitButton > button[kind="primary"]:hover {{
        opacity: 0.94;
        color: #ffffff;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER GRAFIK (SELARAS BIRU NAVY - HIJAU)
# =========================================================
def style_fig(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.82)",
        plot_bgcolor="rgba(249,252,254,0.92)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#182b3e"),
        colorway=NAVY_GREEN_PALETTE,
        margin=dict(l=24, r=24, t=54 if title else 36, b=28),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cfe0ec",
            borderwidth=1
        )
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color="#152940")))
    fig.update_xaxes(showgrid=True, gridcolor="#e2ecf3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2ecf3", zeroline=False)
    return fig

# =========================================================
# DATA & MODEL
# =========================================================
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        return None
    return data

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

@st.cache_resource
def train_models(data):
    clean = data[FEATURES + [TARGET]].copy()
    X = clean[FEATURES]
    y = clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model_definitions = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.08, max_depth=3,
            random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }

    fitted = {}
    rows = []
    predictions = {}
    for name, estimator in model_definitions.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("regressor", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    return fitted, results, best_name, X_test, y_test, predictions

@st.cache_data
def pca_analysis(data):
    X = data[NUMERIC_FEATURES].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(scaled)
    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)
    n90 = int(np.argmax(cumulative >= 0.90) + 1)
    n95 = int(np.argmax(cumulative >= 0.95) + 1)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=NUMERIC_FEATURES,
        columns=[f"PC{i}" for i in range(1, len(NUMERIC_FEATURES) + 1)]
    )
    return individual, cumulative, n90, n95, pca, loadings

def get_feature_importance(pipeline):
    reg = pipeline.named_steps.get("regressor")
    prep = pipeline.named_steps.get("preprocessor")
    if reg is None or prep is None:
        return None
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        return None
    clean_names = [
        n.replace("numeric__", "").replace("categorical__", "")
        for n in feature_names
    ]
    if hasattr(reg, "feature_importances_"):
        values = reg.feature_importances_
        label = "Feature Importance"
    elif hasattr(reg, "coef_"):
        values = np.abs(np.ravel(reg.coef_))
        label = "|Koefisien Regresi|"
    else:
        return None
    imp_df = pd.DataFrame({"Fitur": clean_names, label: values})
    return imp_df.sort_values(label, ascending=False).reset_index(drop=True), label

def money(value):
    return "US$ " + f"{value:,.0f}"

# =========================================================
# SIDEBAR NAVIGATION (TOMBOL TANPA BULET-BULET) & DATA UPLOAD
# =========================================================
with st.sidebar:
    st.markdown(f"""<div class="sidebar-brand"><p class="sidebar-brand-title">🏡 California Housing Prices</p><p class="sidebar-brand-sub">{GROUP_NAME} • Kelas {CLASS_NAME}</p></div>""", unsafe_allow_html=True)

    for section_name, pages in NAV_SECTIONS.items():
        st.markdown(f"<div class='sidebar-group-label'>{section_name}</div>", unsafe_allow_html=True)
        for page_name in pages:
            is_active = (menu == page_name)
            btn_label = f"▸ {page_name}" if is_active else page_name
            if st.button(
                btn_label,
                key=f"nav_btn_{page_name}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state.active_menu = page_name
                st.rerun()

    st.divider()
    st.markdown("<div class='sidebar-group-label'>Sumber Dataset</div>", unsafe_allow_html=True)
    uploaded_data = st.file_uploader(
        "Ganti dataset CSV (opsional)",
        type=["csv"],
        help="Jika tidak mengunggah file, aplikasi memakai housing(3).csv yang ada di folder proyek."
    )

    st.divider()
    st.caption(f"{COURSE_TITLE} • {CLASS_NAME}")

# =========================================================
# LOAD DATA
# =========================================================
df = load_data(uploaded_data)
if df is None:
    st.error("Dataset tidak ditemukan. Letakkan file housing(3).csv satu folder dengan app.py atau unggah CSV melalui sidebar.")
    st.stop()

missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_columns:
    st.error(f"Kolom dataset belum sesuai. Kolom yang tidak ditemukan: {missing_columns}")
    st.stop()

# =========================================================
# PAGES
# =========================================================
if menu == "Beranda":
    st.markdown(f"""<div class="hero"><div class="hero-badge">🏡 California Housing Prices • {GROUP_NAME} ({CLASS_NAME})</div><h1>Prediksi Harga Rumah California</h1><p>Penjelasan dataset, pola karakteristik wilayah, reduksi dimensi, dan model prediksi nilai median rumah</p></div>""", unsafe_allow_html=True)

    st.write(
        "Website ini menjelaskan dataset **California Housing Prices** secara lengkap dan "
        "terstruktur—mulai dari makna tiap variabel wilayah dan hunian, sebaran nilai rumah, "
        "hubungan pendapatan serta kedekatan dengan laut, proses preprocessing dan PCA, "
        "perbandingan model regresi, hingga simulasi prediksi harga rumah."
    )
    st.write(
        "Setiap baris pada dataset ini merepresentasikan **kelompok blok/wilayah** di "
        "California (bukan satu unit rumah tunggal), sehingga analisisnya berfokus pada "
        "pola karakteristik wilayah terhadap **`median_house_value`**."
    )

    models, results, best_name, X_test, y_test, preds = train_models(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Observasi", f"{len(df):,}")
    c2.metric("Jumlah Fitur Input", f"{len(FEATURES)}")
    c3.metric("Model Terbaik (RMSE)", best_name)
    c4.metric("Skor R² Terbaik", f"{results.iloc[0]['R²']:.4f}")

    st.divider()
    st.subheader("Cakupan Pembahasan Dataset")
    st.markdown("""<div class="card-grid-4"><div class="info-card"><div><span class="step-badge">Bagian 01</span><h4>Struktur &amp; EDA</h4><p>Memahami definisi tiap kolom, kelengkapan data, distribusi harga rumah, dan korelasi antarvariabel.</p></div><div class="card-footer-tag">Eksplorasi &amp; Visualisasi</div></div><div class="info-card"><div><span class="step-badge-green">Bagian 02</span><h4>Preprocessing &amp; PCA</h4><p>Menangani data kosong, standarisasi fitur numerik, encoding kategori, dan analisis variansi PCA.</p></div><div class="card-footer-tag">Pipeline &amp; Reduksi Dimensi</div></div><div class="info-card"><div><span class="step-badge">Bagian 03</span><h4>Model Regresi</h4><p>Melatih dan membandingkan enam algoritma regresi menggunakan metrik MAE, RMSE, dan skor R².</p></div><div class="card-footer-tag">Evaluasi &amp; Komparasi</div></div><div class="info-card"><div><span class="step-badge-green">Bagian 04</span><h4>Simulasi Prediksi</h4><p>Menguji estimasi nilai median rumah untuk satu wilayah maupun unggahan file CSV secara massal.</p></div><div class="card-footer-tag">Penerapan Model</div></div></div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader(f"Identitas {GROUP_NAME} — Kelas {CLASS_NAME}")

    member_cards_html = "".join([
        f'<div class="info-card">'
        f'<div>'
        f'<span class="{"step-badge-green" if idx % 2 == 0 else "step-badge"}">Anggota {idx:02d}</span>'
        f'<h4>{name}</h4>'
        f'<p>{STUDY_PROGRAM}</p>'
        f'</div>'
        f'<div class="card-footer-tag">NIM: {nim}</div>'
        f'</div>'
        for idx, (name, nim) in enumerate(GROUP_MEMBERS, start=1)
    ])
    st.markdown(f'<div class="card-grid-3">{member_cards_html}</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="academic-banner"><div class="academic-item"><strong>Kelompok / Kelas:</strong> {GROUP_NAME} ({CLASS_NAME})</div><div class="academic-item"><strong>Mata Kuliah:</strong> {COURSE_TITLE}</div><div class="academic-item"><strong>Dosen Pengampu:</strong> {LECTURER}</div></div>""", unsafe_allow_html=True)

    with st.expander("Penjelasan mendalam mengenai sifat dataset California Housing Prices"):
        st.write(
            "Dataset **California Housing Prices** berasal dari data sensus wilayah di negara "
            "bagian California. Satu observasi (baris) menggambarkan agregat suatu blok/area "
            "geografis, bukan rumah individual—sehingga kolom seperti `total_rooms`, "
            "`total_bedrooms`, `population`, dan `households` merupakan akumulasi pada "
            "wilayah tersebut."
        )
        st.write(
            "Target **`median_house_value`** menyatakan nilai median rumah dalam dolar AS (US$) "
            "pada blok tersebut. Pada dataset California Housing klasik, nilai target memiliki "
            "batas atas (*capping*) di sekitar **US$ 500.001**, artinya blok dengan nilai "
            "median di atas ambang tersebut dicatat pada nilai batas atas yang sama."
        )

elif menu == "Tentang Dataset":
    st.title("Tentang Dataset")
    st.write(
        "Dataset **California Housing Prices** merekam informasi lokasi geografis, "
        "demografi penduduk, ukuran hunian, usia bangunan, kedekatan ke pesisir, dan "
        "tingkat pendapatan rumah tangga untuk memperkirakan **`median_house_value`**."
    )

    a, b, c, d = st.columns(4)
    a.metric("Jumlah Baris", f"{len(df):,}")
    b.metric("Jumlah Kolom", len(df.columns))
    c.metric("Total Missing Value", f"{int(df.isna().sum().sum()):,}")
    d.metric("Median Harga Rumah", money(df[TARGET].median()))

    st.markdown("""<div class="insight-box"><strong>Penjelasan jenis masalah &amp; struktur data:</strong> Karena target <code>median_house_value</code> merupakan variabel numerik kontinu (nilai uang), permasalahan ini diselesaikan dengan pendekatan <strong>regresi</strong>. Dataset memiliki <strong>8 fitur numerik</strong> (menggambarkan koordinat, usia rumah, jumlah ruang/kamar, populasi, rumah tangga, dan pendapatan) serta <strong>1 fitur kategorikal</strong> (<code>ocean_proximity</code>) yang merangkum posisi wilayah terhadap laut.</div>""", unsafe_allow_html=True)

    st.subheader("Cuplikan Dataset (20 Baris Awal)")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Kamus Variabel & Makna Teknisnya")
    descriptions = {
        "longitude": "Koordinat garis bujur wilayah; nilai lebih negatif menunjukkan posisi lebih ke arah barat California.",
        "latitude": "Koordinat garis lintang wilayah; nilai lebih besar menunjukkan posisi lebih ke arah utara California.",
        "housing_median_age": "Median usia bangunan rumah (dalam tahun) pada suatu blok wilayah; mencerminkan apakah kawasan tergolong baru atau lama.",
        "total_rooms": "Jumlah total seluruh ruangan (kamar tidur, ruang tamu, dapur, dll.) pada seluruh rumah di wilayah tersebut.",
        "total_bedrooms": "Jumlah total kamar tidur pada seluruh rumah di wilayah tersebut; sering memiliki nilai kosong (missing) pada sebagian baris.",
        "population": "Jumlah total penduduk yang berdomisili di wilayah blok tersebut.",
        "households": "Jumlah total rumah tangga (kepala keluarga) yang menghuni wilayah tersebut.",
        "median_income": "Median pendapatan rumah tangga di wilayah tersebut, diukur dalam satuan puluhan ribu dolar AS (misalnya 8.0 ≈ US$ 80.000).",
        "ocean_proximity": "Kategori kedekatan wilayah terhadap laut/pesisir (misalnya INLAND, NEAR OCEAN, NEAR BAY, <1H OCEAN, ISLAND).",
        "median_house_value": "Variabel target: nilai median rumah (dalam dolar AS) pada blok wilayah tersebut."
    }
    desc_df = pd.DataFrame({
        "Variabel": list(descriptions.keys()),
        "Peran": ["Target" if k == TARGET else ("Kategorikal" if k in CATEGORICAL_FEATURES else "Numerik") for k in descriptions],
        "Tipe Data": [str(df[k].dtype) for k in descriptions],
        "Keterangan": list(descriptions.values())
    })
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.subheader("Ringkasan Statistik Deskriptif")
    st.write(
        "Tabel berikut menampilkan ringkasan statistik seperti rata-rata (`mean`), simpangan "
        "baku (`std`), nilai minimum/maksimum, serta kuartil (`25%`, `50%`, `75%`) untuk "
        "melihat rentang skala tiap variabel sebelum dinormalisasi."
    )
    st.dataframe(df.describe(include="all").T, use_container_width=True)

elif menu == "Eksplorasi Data (EDA)":
    st.title("Eksplorasi Data (EDA)")
    st.write(
        "Exploratory Data Analysis (EDA) bertujuan memeriksa kelengkapan data, bentuk "
        "sebaran nilai target, kekuatan hubungan linear antarvariabel numerik, perbedaan "
        "harga antar kategori lokasi, serta keberadaan nilai ekstrem (*outlier*)."
    )

    st.subheader("1. Pemeriksaan Missing Value")
    st.write(
        "Pada dataset California Housing, kekurangan data umumnya muncul pada kolom "
        "`total_bedrooms` karena pendataan jumlah kamar tidur per blok tidak selalu lengkap "
        "pada seluruh wilayah. Mengetahui jumlah dan persentasenya membantu menentukan "
        "strategi imputasi yang tepat."
    )
    missing = pd.DataFrame({
        "Jumlah Missing": df.isna().sum(),
        "Persentase (%)": (df.isna().mean() * 100).round(2)
    }).sort_values("Jumlah Missing", ascending=False)
    st.dataframe(missing, use_container_width=True)
    fig_missing = px.bar(
        missing.reset_index(), x="index", y="Jumlah Missing",
        labels={"index": "Variabel"}
    )
    fig_missing.update_traces(marker_color="#1f3a5a")
    st.plotly_chart(style_fig(fig_missing, "Jumlah Missing Value per Variabel"), use_container_width=True)

    st.subheader("2. Distribusi Target (`median_house_value`)")
    st.write(
        "Histogram dan boxplot berikut memperlihatkan penyebaran `median_house_value` di "
        "seluruh wilayah. Perhatikan apakah distribusinya condong ke kanan (*right-skewed*) "
        "dan apakah terdapat penumpukan observasi pada nilai maksimum tertentu."
    )
    fig_target = px.histogram(
        df, x=TARGET, nbins=50, marginal="box"
    )
    fig_target.update_traces(marker_color="#3f7d64", selector=dict(type="histogram"))
    fig_target.update_traces(marker_color="#1f3a5a", selector=dict(type="box"))
    st.plotly_chart(style_fig(fig_target, "Distribusi Median House Value"), use_container_width=True)
    st.caption(
        "Lonjakan pada batas atas histogram menunjukkan adanya pembatasan nilai maksimum "
        "(capping) pada variabel target di dataset ini, yang membuat model cenderung "
        "tidak memprediksi nilai jauh di atas batas capping tersebut."
    )

    st.subheader("3. Korelasi Variabel Numerik")
    st.write(
        "Matriks korelasi Pearson mengukur kekuatan dan arah hubungan **linear** antar "
        "pasangan variabel numerik (rentang `-1` hingga `+1`). Nilai mendekati `+1` berarti "
        "cenderung naik bersama, mendekati `-1` berarti berlawanan arah, dan mendekati `0` "
        "berarti hubungan linearnya lemah (bukan berarti tidak ada hubungan non-linear)."
    )
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#1f3a5a", "#f4f8fb", "#3f7d64"],
        zmin=-1, zmax=1
    )
    st.plotly_chart(style_fig(fig_corr, "Matriks Korelasi Pearson (Variabel Numerik)"), use_container_width=True)

    top_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    strongest_feat = top_corr.index[0]
    strongest_val = top_corr.iloc[0]
    st.markdown(f"""<div class="insight-box"><strong>Temuan korelasi terhadap target:</strong> Variabel numerik dengan korelasi linear terkuat terhadap <code>{TARGET}</code> adalah <strong>{strongest_feat}</strong> (r = <strong>{strongest_val:.2f}</strong>). Hal ini menegaskan bahwa tingkat pendapatan rumah tangga merupakan salah satu indikator utama daya beli dan nilai properti di suatu wilayah, meskipun faktor lokasi geografis dan kategori pesisir juga sangat berperan.</div>""", unsafe_allow_html=True)

    st.subheader("4. Hubungan Pendapatan dan Harga Rumah")
    st.write(
        "Scatter plot di bawah memvisualisasikan hubungan antara `median_income` dan "
        "`median_house_value` dengan pewarnaan berdasarkan `ocean_proximity`. Terlihat tren "
        "kenaikan harga seiring naiknya pendapatan, sekaligus garis horizontal batas atas "
        "nilai rumah pada dataset."
    )
    sample_n = min(5000, len(df))
    fig_income = px.scatter(
        df.sample(sample_n, random_state=42),
        x="median_income", y=TARGET, color="ocean_proximity",
        opacity=0.65
    )
    st.plotly_chart(
        style_fig(fig_income, f"Median Income vs Median House Value (Sampel {sample_n:,} Baris)"),
        use_container_width=True
    )

    st.subheader("5. Harga Rumah Berdasarkan Ocean Proximity")
    st.write(
        "Perbandingan statistik dan boxplot berdasarkan `ocean_proximity` menunjukkan "
        "bagaimana kedekatan wilayah terhadap laut atau teluk memengaruhi tingkat dan "
        "sebaran harga rumah dibandingkan wilayah pedalaman (`INLAND`)."
    )
    grouped = df.groupby("ocean_proximity", dropna=False)[TARGET].agg(
        Jumlah="count", Rata_rata="mean", Median="median"
    ).reset_index().sort_values("Median", ascending=False)
    st.dataframe(
        grouped.style.format({"Jumlah": "{:,.0f}", "Rata_rata": "{:,.2f}", "Median": "{:,.2f}"}),
        use_container_width=True, hide_index=True
    )
    fig_ocean = px.box(
        df, x="ocean_proximity", y=TARGET, color="ocean_proximity", points=False
    )
    st.plotly_chart(
        style_fig(fig_ocean, "Sebaran Harga Rumah Berdasarkan Ocean Proximity"),
        use_container_width=True
    )

    st.subheader("6. Deteksi Outlier dengan Metode IQR")
    st.write(
        "Metode *Interquartile Range* (IQR = Q3 − Q1) menandai nilai yang berada di bawah "
        "`Q1 − 1,5 × IQR` atau di atas `Q3 + 1,5 × IQR` sebagai *outlier* statistik. Pada "
        "data wilayah, banyaknya *outlier* pada fitur jumlah ruangan atau populasi sering kali "
        "disebabkan oleh perbedaan ukuran/kepadatan blok antarwilayah."
    )
    outlier_rows = []
    for col in NUMERIC_FEATURES + [TARGET]:
        series = df[col].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        outlier_rows.append({
            "Variabel": col,
            "Jumlah Outlier": count,
            "Persentase (%)": round(count / len(df) * 100, 2)
        })
    outlier_df = pd.DataFrame(outlier_rows).sort_values("Jumlah Outlier", ascending=False)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    st.info(
        "Nilai di luar rentang 1,5 × IQR tidak langsung dibuang karena pada data wilayah "
        "nilai ekstrem bisa mencerminkan kondisi geografis atau demografis yang memang ada "
        "(misalnya kawasan padat penduduk atau blok dengan banyak unit rumah)."
    )

elif menu == "Preprocessing":
    st.title("Preprocessing")
    st.write(
        "Model machine learning memerlukan input numerik yang bersih dan terstruktur. "
        "Pada proyek ini, seluruh tahapan pembersihan dan transformasi fitur disatukan ke "
        "dalam **`Pipeline`** dan **`ColumnTransformer`** agar urutan prosesnya baku "
        "dan dapat direplikasi pada data uji maupun input prediksi baru."
    )

    st.subheader("1. Ringkasan Kualitas Kolom")
    st.write(
        "Tabel berikut merangkum tipe data, jumlah nilai kosong, dan jumlah nilai unik "
        "di setiap kolom dataset yang dimuat saat ini."
    )
    quality = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Missing Value": df.isna().sum().values,
        "Nilai Unik": df.nunique().values
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.subheader("2. Strategi Transformasi Tiap Tipe Fitur")
    st.markdown("""<div class="card-grid-2"><div class="info-card"><div><span class="step-badge">Fitur Numerik (8 Kolom)</span><h4>Imputasi Median + StandardScaler</h4><p>Nilai kosong diisi dengan <strong>median</strong> data training karena lebih tahan terhadap nilai ekstrem/<em>outlier</em> dibanding rata-rata (<em>mean</em>). Selanjutnya <strong>StandardScaler</strong> menstandarisasi tiap fitur agar memiliki rata-rata mendekati 0 dan standar deviasi 1, sehingga fitur berskala besar seperti <code>population</code> atau <code>total_rooms</code> tidak mendominasi proses belajar model.</p></div><div class="card-footer-tag">SimpleImputer(median) → StandardScaler()</div></div><div class="info-card"><div><span class="step-badge-green">Fitur Kategorikal (1 Kolom)</span><h4>Imputasi Modus + OneHotEncoder</h4><p>Jika terdapat nilai kosong pada <code>ocean_proximity</code>, diisi menggunakan kategori paling sering muncul (<strong>most_frequent</strong>). Kemudian <strong>OneHotEncoder</strong> mengubah tiap kategori menjadi kolom indikator biner (0/1) tanpa mengasumsikan urutan angka buatan, dengan parameter <code>handle_unknown='ignore'</code> agar aman saat menerima kategori baru.</p></div><div class="card-footer-tag">SimpleImputer(most_frequent) → OneHotEncoder()</div></div></div>""", unsafe_allow_html=True)

    st.subheader("3. Pembagian Data Latih & Data Uji")
    st.write(
        "Dataset dibagi menjadi **80% data training** (untuk melatih model dan mempelajari "
        "parameter preprocessing) dan **20% data testing** (untuk mengukur performa model "
        "pada data yang belum pernah dilihat saat pelatihan)."
    )
    train_count = int(len(df) * 0.8)
    test_count = len(df) - train_count
    x, y = st.columns(2)
    x.metric("Data Training (80%)", f"{train_count:,} baris")
    y.metric("Data Testing (20%)", f"{test_count:,} baris")
    st.caption(
        "Pembagian dilakukan menggunakan `train_test_split(..., test_size=0.20, random_state=42)` "
        "agar pembagian data bersifat reproducible (konsisten saat dijalankan ulang)."
    )

    st.subheader("4. Cuplikan Kode Pipeline")
    st.code("""
ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), NUMERIC_FEATURES),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), CATEGORICAL_FEATURES)
])
""", language="python")

    st.markdown("""<div class="insight-box"><strong>Mengapa sangat penting menggunakan Pipeline?</strong> Jika imputasi atau scaling dihitung pada seluruh dataset sebelum dibagi menjadi data latih dan data uji, informasi dari data uji dapat ikut memengaruhi proses (<em>data leakage</em>). Dengan <code>Pipeline</code>, metode <code>fit</code> hanya berjalan pada data training, sedangkan <code>transform</code> diterapkan pada data testing dan data prediksi baru.</div>""", unsafe_allow_html=True)

elif menu == "Reduksi Dimensi (PCA)":
    st.title("Reduksi Dimensi (PCA)")
    st.write(
        "*Principal Component Analysis* (PCA) adalah teknik reduksi dimensi linear yang "
        "mengubah variabel-variabel numerik yang saling berkorelasi menjadi kombinasi linear "
        "baru yang saling ortogonal (tidak saling berkorelasi), disebut **Principal Components** "
        "(PC1, PC2, ...), yang diurutkan dari variansi terbesar ke terkecil."
    )
    st.write(
        "Sebelum PCA dijalankan, 8 fitur numerik distandardisasi terlebih dahulu karena PCA "
        "sensitif terhadap perbedaan skala antarvariabel."
    )
    individual, cumulative, n90, n95, pca, loadings = pca_analysis(df)
    k1, k2, k3 = st.columns(3)
    k1.metric("Fitur Numerik Awal", len(NUMERIC_FEATURES))
    k2.metric("Komponen untuk ≥ 90% Variansi", f"{n90} Komponen")
    k3.metric("Komponen untuk ≥ 95% Variansi", f"{n95} Komponen")

    pc_df = pd.DataFrame({
        "Komponen": [f"PC{i}" for i in range(1, len(individual) + 1)],
        "Explained Variance": individual,
        "Cumulative Variance": cumulative
    })
    fig_pca = go.Figure()
    fig_pca.add_bar(
        x=pc_df["Komponen"], y=pc_df["Explained Variance"],
        name="Variansi per Komponen", marker_color="#1f3a5a"
    )
    fig_pca.add_scatter(
        x=pc_df["Komponen"], y=pc_df["Cumulative Variance"],
        name="Variansi Kumulatif", mode="lines+markers",
        line=dict(color="#3f7d64", width=3),
        marker=dict(size=8, color="#2a5e49")
    )
    fig_pca.add_hline(y=0.90, line_dash="dash", line_color="#1f3a5a", annotation_text="Ambang 90%")
    fig_pca.add_hline(y=0.95, line_dash="dot", line_color="#3f7d64", annotation_text="Ambang 95%")
    fig_pca.update_layout(
        yaxis_title="Proporsi Variansi", xaxis_title="Principal Component"
    )
    st.plotly_chart(style_fig(fig_pca, "Scree Plot & Variansi Kumulatif PCA"), use_container_width=True)

    st.subheader("Tabel Variansi per Komponen")
    st.write(
        "`Explained Variance` menunjukkan proporsi keragaman data yang ditangkap oleh satu "
        "komponen tertentu, sedangkan `Cumulative Variance` menunjukkan total akumulasi variansi "
        "dari PC1 hingga komponen ke-k."
    )
    st.dataframe(
        pc_df.style.format({
            "Explained Variance": "{:.4f}",
            "Cumulative Variance": "{:.2%}"
        }),
        use_container_width=True, hide_index=True
    )

    with st.expander("Lihat Bobot / Loading Tiap Fitur pada Komponen Utama"):
        st.write(
            "Nilai *loading* menunjukkan seberapa besar kontribusi tiap fitur asli terhadap "
            "pembentukan suatu komponen utama (tanda positif/negatif menunjukkan arah bobot)."
        )
        st.dataframe(loadings.round(3), use_container_width=True)

    st.markdown(f"""<div class="insight-box"><strong>Interpretasi hasil PCA:</strong> Sebanyak <strong>{n90} komponen</strong> sudah mampu mempertahankan <strong>{cumulative[n90-1]*100:.2f}%</strong> variansi data numerik, dan <strong>{n95} komponen</strong> mempertahankan <strong>{cumulative[n95-1]*100:.2f}%</strong>. Meski PCA berguna untuk meringkas dimensi dan melihat struktur variansi, model regresi pada halaman evaluasi tetap memakai fitur asli + One-Hot Encoding agar hubungan tiap fitur (seperti pendapatan dan lokasi) tetap langsung terbaca.</div>""", unsafe_allow_html=True)

elif menu == "Modelling & Evaluasi":
    st.title("Modelling & Evaluasi")
    st.write(
        "Pada tahap ini, enam algoritma regresi dilatih pada data training yang identik dan "
        "diuji pada 20% data testing untuk memperkirakan `median_house_value`. Performa "
        "dibandingkan menggunakan tiga metrik utama: **MAE**, **RMSE**, dan **R²**."
    )
    st.write(
        "- **MAE (*Mean Absolute Error*)**: rata-rata selisih absolut antara nilai aktual dan "
        "prediksi dalam satuan dolar AS; mudah dimaknai sebagai rata-rata kesalahan prediksi.\n"
        "- **RMSE (*Root Mean Squared Error*)**: akar dari rata-rata kuadrat kesalahan; "
        "memberi bobot lebih besar pada kesalahan prediksi yang cukup jauh dari nilai aktual.\n"
        "- **R² (*Coefficient of Determination*)**: proporsi variasi `median_house_value` yang "
        "dapat dijelaskan oleh model (semakin mendekati 1 semakin baik pada data uji)."
    )
    with st.spinner("Melatih dan mengevaluasi model..."):
        models, results, best_name, X_test, y_test, preds = train_models(df)

    st.subheader("1. Tabel Perbandingan Seluruh Model")
    st.dataframe(
        results.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    fig_compare = px.bar(
        results.sort_values("RMSE"), x="Model", y="RMSE",
        color="Model", text_auto=".2s"
    )
    st.plotly_chart(
        style_fig(fig_compare, "Perbandingan RMSE Antar Model (Semakin Rendah Semakin Baik)"),
        use_container_width=True
    )

    st.subheader("2. Performa Model Terbaik")
    best_row = results.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Model Terbaik", best_name)
    b.metric("MAE", money(best_row["MAE"]))
    c.metric("RMSE", money(best_row["RMSE"]))
    d.metric("R²", f"{best_row['R²']:.4f}")

    st.subheader("3. Visualisasi Prediksi vs Nilai Aktual")
    st.write(
        "Titik-titik pada grafik di bawah membandingkan nilai aktual (`Aktual`) pada sumbu X "
        "dengan hasil prediksi model (`Prediksi`) pada sumbu Y untuk 150 data uji pertama. "
        "Garis putus-putus diagonal menggambarkan kondisi prediksi sempurna (`Prediksi = Aktual`)."
    )
    best_pred = preds[best_name]
    compare = pd.DataFrame({
        "Aktual": y_test.to_numpy()[:150],
        "Prediksi": best_pred[:150]
    })
    fig_actual = px.scatter(
        compare, x="Aktual", y="Prediksi",
        opacity=0.72
    )
    fig_actual.update_traces(marker=dict(color="#1f3a5a", size=7, line=dict(width=1, color="#ffffff")))
    low = float(min(compare.min()))
    high = float(max(compare.max()))
    fig_actual.add_shape(
        type="line", x0=low, y0=low, x1=high, y1=high,
        line=dict(dash="dash", color="#3f7d64", width=2)
    )
    st.plotly_chart(
        style_fig(fig_actual, f"Nilai Aktual vs Prediksi ({best_name} — 150 Sampel Data Uji Pertama)"),
        use_container_width=True
    )

    imp_result = get_feature_importance(models[best_name])
    if imp_result is not None:
        imp_df, imp_label = imp_result
        st.subheader(f"4. Kontribusi Fitur pada {best_name}")
        st.write(
            "Grafik berikut memperlihatkan fitur-fitur yang paling berkontribusi dalam "
            "membentuk prediksi pada model terbaik, membantu menjelaskan faktor mana yang "
            "paling dominan memengaruhi estimasi harga rumah."
        )
        fig_imp = px.bar(
            imp_df.head(10).sort_values(imp_label),
            x=imp_label, y="Fitur", orientation="h"
        )
        fig_imp.update_traces(marker_color="#3f7d64")
        st.plotly_chart(
            style_fig(fig_imp, f"Top 10 Fitur Paling Berpengaruh ({imp_label})"),
            use_container_width=True
        )

    st.caption(
        "Model berbasis pohon/ensemble (seperti Random Forest, Extra Trees, atau Gradient Boosting) "
        "umumnya mengungguli model linear karena mampu menangkap pola non-linear dan interaksi "
        "antar lokasi geografis, kepadatan wilayah, serta pendapatan."
    )

elif menu == "Prediksi Satu Rumah":
    st.title("Prediksi Satu Rumah")
    st.write(
        "Masukkan nilai fitur karakteristik wilayah dan hunian di bawah ini untuk memperoleh "
        "estimasi `median_house_value` menggunakan model terbaik yang telah dilatih. Anda juga "
        "dapat menekan tombol contoh cepat untuk membandingkan skenario wilayah berbeda."
    )
    with st.spinner("Menyiapkan model prediksi..."):
        models, results, best_name, *_ = train_models(df)
    st.caption(f"Model yang digunakan: **{best_name}** (RMSE pada data uji: {money(results.iloc[0]['RMSE'])})")

    presets = {
        "Contoh 1 — Area Teluk / Pendapatan Tinggi": {
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
            "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
            "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
        },
        "Contoh 2 — Area Pedalaman / Menengah": {
            "longitude": -119.78, "latitude": 36.74, "housing_median_age": 24.0,
            "total_rooms": 2450.0, "total_bedrooms": 480.0, "population": 1350.0,
            "households": 450.0, "median_income": 3.4500, "ocean_proximity": "INLAND"
        },
    }
    ocean_options = sorted(df["ocean_proximity"].dropna().astype(str).unique().tolist())

    if "pred_inputs" not in st.session_state:
        st.session_state.pred_inputs = {
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 25.0,
            "total_rooms": 2000.0, "total_bedrooms": 400.0, "population": 1000.0,
            "households": 350.0, "median_income": 4.0,
            "ocean_proximity": ocean_options[0] if ocean_options else "NEAR BAY"
        }

    p_col1, p_col2, _ = st.columns([2, 2, 3])
    if p_col1.button("🏡 Contoh 1 (Pesisir/Tinggi)", use_container_width=True):
        st.session_state.pred_inputs.update(presets["Contoh 1 — Area Teluk / Pendapatan Tinggi"])
        st.rerun()
    if p_col2.button("🏡 Contoh 2 (Pedalaman/Menengah)", use_container_width=True):
        st.session_state.pred_inputs.update(presets["Contoh 2 — Area Pedalaman / Menengah"])
        st.rerun()

    cur = st.session_state.pred_inputs
    default_ocean = cur["ocean_proximity"] if cur["ocean_proximity"] in ocean_options else ocean_options[0]

    with st.form("single_prediction"):
        c1, c2 = st.columns(2)
        with c1:
            longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=float(cur["longitude"]), step=0.01)
            latitude = st.number_input("Latitude", min_value=32.0, max_value=43.0, value=float(cur["latitude"]), step=0.01)
            housing_median_age = st.number_input("Housing Median Age (tahun)", min_value=0.0, max_value=100.0, value=float(cur["housing_median_age"]))
            total_rooms = st.number_input("Total Rooms", min_value=1.0, value=float(cur["total_rooms"]))
        with c2:
            total_bedrooms = st.number_input("Total Bedrooms", min_value=0.0, value=float(cur["total_bedrooms"]))
            population = st.number_input("Population", min_value=1.0, value=float(cur["population"]))
            households = st.number_input("Households", min_value=1.0, value=float(cur["households"]))
            median_income = st.number_input("Median Income (puluhan ribu USD)", min_value=0.0, value=float(cur["median_income"]), step=0.1)
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ocean_options,
            index=ocean_options.index(default_ocean)
        )
        submitted = st.form_submit_button("Prediksi Harga Rumah", type="primary")

    if submitted:
        st.session_state.pred_inputs = {
            "longitude": longitude, "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
            "population": population, "households": households,
            "median_income": median_income, "ocean_proximity": ocean_proximity
        }
        input_df = pd.DataFrame([st.session_state.pred_inputs])
        estimate = float(models[best_name].predict(input_df)[0])
        st.success("Prediksi berhasil dihitung.")
        st.markdown(f"""<div class="result-banner"><span class="step-badge-green">Hasil Estimasi • {best_name}</span><h2 style="margin: 8px 0 6px 0; color: #152940;">{money(estimate)}</h2><small style="color: #365066;">Lokasi ({longitude:.2f}, {latitude:.2f}) • Kategori <strong>{ocean_proximity}</strong> • Usia bangunan <strong>{housing_median_age:.0f} thn</strong> • Median income <strong>{median_income:.2f}</strong> (×US$ 10.000)</small></div>""", unsafe_allow_html=True)
        st.caption(
            "Hasil merupakan estimasi median nilai rumah pada karakteristik wilayah yang "
            "diinput berdasarkan pola dataset historis, bukan harga transaksi pasar aktual "
            "atau penilaian appraisal resmi."
        )

elif menu == "Prediksi Massal (CSV)":
    st.title("Prediksi Massal (CSV)")
    st.write(
        "Unggah file CSV yang berisi banyak baris data wilayah/properti untuk menghasilkan "
        "kolom prediksi `predicted_median_house_value` secara otomatis dan mengunduh hasilnya."
    )
    st.info(
        "Pastikan file CSV memiliki 8 kolom numerik (`longitude`, `latitude`, "
        "`housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, "
        "`households`, `median_income`) dan kolom `ocean_proximity`. Kolom target "
        "`median_house_value` tidak perlu disertakan."
    )

    template = pd.DataFrame([
        {
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
            "total_rooms": 880, "total_bedrooms": 129, "population": 322,
            "households": 126, "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        },
        {
            "longitude": -119.78, "latitude": 36.74, "housing_median_age": 24,
            "total_rooms": 2450, "total_bedrooms": 480, "population": 1350,
            "households": 450, "median_income": 3.4500,
            "ocean_proximity": "INLAND"
        }
    ])
    st.download_button(
        "Unduh Contoh Template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_prediksi_housing.csv",
        mime="text/csv"
    )

    batch_file = st.file_uploader("Pilih file CSV untuk diprediksi", type=["csv"], key="batch_csv")
    if batch_file is not None:
        batch = pd.read_csv(batch_file)
        missing = [col for col in FEATURES if col not in batch.columns]
        if missing:
            st.error(f"Kolom berikut belum tersedia pada file CSV: {missing}")
        elif len(batch) == 0:
            st.warning("File CSV tidak memiliki baris data.")
        else:
            with st.spinner("Memproses prediksi massal..."):
                models, results, best_name, *_ = train_models(df)
                batch_features = batch[FEATURES].copy()
                batch["predicted_median_house_value"] = models[best_name].predict(batch_features)

            st.success(f"Prediksi selesai untuk {len(batch):,} baris menggunakan model **{best_name}**.")
            s1, s2, s3 = st.columns(3)
            s1.metric("Jumlah Data Diprediksi", f"{len(batch):,}")
            s2.metric("Rata-rata Prediksi", money(batch["predicted_median_house_value"].mean()))
            s3.metric("Median Prediksi", money(batch["predicted_median_house_value"].median()))

            st.dataframe(batch.head(50), use_container_width=True)
            st.download_button(
                "Unduh Hasil Prediksi (CSV)",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_housing.csv",
                mime="text/csv"
            )

elif menu == "Kesimpulan & Saran":
    st.title("Kesimpulan & Saran")
    st.subheader("Kesimpulan")
    st.write(
        "Dataset **California Housing Prices** memperlihatkan bahwa nilai median rumah "
        "pada suatu blok wilayah berkaitan erat dengan tingkat pendapatan rumah tangga "
        "(`median_income`), letak geografis (`longitude`, `latitude`), serta kedekatan "
        "dengan pesisir/perairan (`ocean_proximity`), sekaligus memiliki ciri khas berupa "
        "batas atas nilai (*capping*) pada distribusi target."
    )
    st.write(
        "Penerapan `Pipeline` dan `ColumnTransformer` memastikan proses imputasi nilai "
        "kosong (median untuk numerik, modus untuk kategorikal), standarisasi fitur "
        "numerik, dan *one-hot encoding* fitur kategorikal berlangsung konsisten serta "
        "menghindari kebocoran informasi (*data leakage*) dari data uji."
    )
    st.write(
        "Analisis PCA menunjukkan bahwa sebagian besar variansi fitur numerik dapat diringkas "
        "dalam beberapa komponen utama, sementara pada pemodelan regresi penggunaan fitur "
        "asli beserta encoding kategori membantu model menangkap pola spesifik tiap wilayah."
    )

    with st.spinner("Memuat hasil evaluasi model..."):
        models, results, best_name, *_ = train_models(df)
    best = results.iloc[0]

    st.markdown(f"""<div class="insight-box"><strong>Ringkasan evaluasi model:</strong> Pada pembagian data saat ini, model terbaik adalah <strong>{best_name}</strong> dengan <strong>MAE {money(best['MAE'])}</strong>, <strong>RMSE {money(best['RMSE'])}</strong>, dan <strong>R² {best['R²']:.4f}</strong>. Angka ini menunjukkan tingkat ketepatan serta margin kesalahan estimasi model pada data uji yang dimuat.</div>""", unsafe_allow_html=True)

    st.subheader("Saran Pengembangan")
    st.markdown("""
1. **Pembaruan & pengayaan data:** Menambahkan data sensus atau transaksi properti yang lebih mutakhir serta fitur tambahan (seperti jarak ke pusat kota, akses transportasi, atau kondisi ekonomi lokal) agar relevansi model tetap terjaga.
2. **Rekayasa fitur (*feature engineering*):** Menguji fitur rasio yang lebih bermakna secara domain—misalnya rata-rata ruangan per rumah tangga, proporsi kamar tidur terhadap total ruangan, atau kepadatan penduduk per rumah tangga.
3. **Hyperparameter tuning & cross-validation:** Melakukan pencarian hyperparameter (*GridSearchCV* atau *RandomizedSearchCV*) disertai *k-fold cross-validation* untuk memastikan performa model tidak bergantung pada satu pembagian data saja.
4. **Kajian efek capping & nilai ekstrem:** Menganalisis pengaruh batas atas harga rumah pada dataset terhadap bias prediksi di segmen properti bernilai tinggi.
5. **Pemanfaatan hasil estimasi secara proporsional:** Menggunakan hasil prediksi model sebagai alat bantu analisis awal atau estimasi kasar, bukan pengganti survei lapangan maupun penilaian properti resmi.
    """)
    st.caption("Seluruh metrik evaluasi diperbarui secara otomatis mengikuti dataset yang sedang aktif.")