import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="Smart Machine | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "SVM": BASE_DIR / "svm_model.pkl",
    "Naive Bayes": BASE_DIR / "naive_bayes_model.pkl",
}

THRESHOLD = 0.30


# =========================================================
# 2. DESAIN APLIKASI
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f7fb;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #101d32 0%,
                #172b46 100%
            );
        }

        [data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
        }

        [data-testid="stSidebar"] * {
            color: #f1f5f9;
        }

        [data-testid="stSidebar"] hr {
            margin: 8px 0 12px;
            border-color: #33465f;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 4px 18px;
        }

        .sidebar-logo {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 44px;
            height: 44px;
            border-radius: 12px;
            background: #23466c;
            font-size: 23px;
        }

        .sidebar-title {
            color: #ffffff;
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 0.5px;
            line-height: 1.4;
        }

        .sidebar-subtitle {
            margin-top: 2px;
            color: #a9bdd5;
            font-size: 11px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            display: none;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: -8px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 10px 12px;
            border-radius: 9px;
        }

        .sidebar-note {
            margin-top: 28px;
            padding: 12px 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.06);
            color: #b8c7d9;
            font-size: 11px;
            line-height: 1.7;
        }

        .sidebar-status {
            color: #86efac;
        }

        .hero {
            margin-bottom: 24px;
            padding: 28px 30px;
            border-radius: 18px;
            background: linear-gradient(120deg, #10213d, #1e4976);
            color: #ffffff;
        }

        .hero h1 {
            margin-bottom: 8px;
            color: #ffffff;
            font-size: 2.1rem;
        }

        .hero p {
            margin-bottom: 6px;
            color: #dbeafe;
            font-size: 1rem;
        }

        .section-title {
            margin: 12px 0 16px;
            color: #172b4d;
            font-size: 1.25rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            padding: 18px;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stForm"] {
            padding: 22px;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            min-height: 44px;
            border-radius: 9px;
            font-weight: 600;
        }

        .status-safe,
        .status-risk {
            padding: 20px;
            border-radius: 14px;
        }

        .status-safe {
            border: 1px solid #a7f3d0;
            background: #ecfdf3;
            color: #166534;
        }

        .status-risk {
            border: 1px solid #fecdd3;
            background: #fff1f2;
            color: #9f1239;
        }

        .status-safe h3,
        .status-risk h3 {
            margin-top: 0;
        }

        .footer {
            margin-top: 36px;
            padding-top: 18px;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.8rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MEMUAT MODEL MACHINE LEARNING
# =========================================================
@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.is_file():
            raise FileNotFoundError(
                f"File model {model_name} tidak ditemukan: "
                f"{model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:
    models = load_models()

except Exception as error:
    st.error("Model gagal dimuat.")
    st.error(
        "Periksa lokasi file model dan kompatibilitas "
        "versi library scikit-learn."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 4. FUNGSI BANTU
# =========================================================
def show_section_title(title):
    """Menampilkan judul bagian."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def create_input_data(
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
    product_type,
):
    """Menyusun input sesuai fitur saat pelatihan model."""
    return pd.DataFrame(
        [
            {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": product_type,
            }
        ]
    )


def predict_failure(selected_model, input_data):
    """Menghasilkan probabilitas dan kelas prediksi."""
    if not hasattr(selected_model, "predict_proba"):
        raise ValueError(
            "Model tidak mendukung predict_proba()."
        )

    probabilities = selected_model.predict_proba(input_data)[0]
    classes = list(selected_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Model tidak memiliki kelas target kegagalan (1)."
        )

    failure_probability = float(
        probabilities[classes.index(1)]
    )

    prediction = int(failure_probability >= THRESHOLD)

    return failure_probability, prediction


# =========================================================
# 5. SIDEBAR / NAVIGASI
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">⚙️</div>
            <div>
                <div class="sidebar-title">SMART MACHINE</div>
                <div class="sidebar-subtitle">
                    AI Predictive Maintenance
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("MENU UTAMA")

    page = st.radio(
        "Navigasi",
        options=[
            "Prediksi Mesin",
            "Evaluasi Model",
            "Informasi Sistem",
        ],
        label_visibility="collapsed",
        format_func=lambda item: {
            "Prediksi Mesin": "⚙️  Prediksi Mesin",
            "Evaluasi Model": "📊  Evaluasi Model",
            "Informasi Sistem": "ℹ️  Informasi Sistem",
        }[item],
    )

    st.markdown(
        """
        <div class="sidebar-note">
            <span class="sidebar-status">●</span>
            Sistem monitoring berbasis AI
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. HEADER UTAMA
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>Smart Machine</h1>
        <p>
            <b>
                Intelligent Machine Monitoring &
                Predictive Maintenance
            </b>
        </p>
        <p>
            Analisis data operasional untuk membantu
            mengidentifikasi indikasi potensi kegagalan mesin.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. HALAMAN PREDIKSI MESIN
# =========================================================
if page == "Prediksi Mesin":

    show_section_title("Input Kondisi Operasional")

    st.write(
        "Masukkan parameter operasional mesin dan pilih "
        "algoritma untuk memperoleh estimasi potensi kegagalan."
    )

    with st.form("form_prediksi"):

        st.markdown("#### Pengaturan Analisis")

        available_models = list(models.keys())

        selected_algorithm = st.selectbox(
            "Pilih Algoritma Machine Learning",
            options=available_models,
            index=0,
            help="Pilih model yang akan digunakan untuk prediksi.",
        )

        st.markdown("#### Parameter Mesin")

        col1, col2, col3 = st.columns(3)

        with col1:
            air_temp = st.number_input(
                "Suhu udara (K)",
                min_value=295.0,
                max_value=305.0,
                value=300.0,
                step=0.1,
                help="Suhu lingkungan di sekitar mesin.",
            )

            process_temp = st.number_input(
                "Suhu proses (K)",
                min_value=305.0,
                max_value=315.0,
                value=310.0,
                step=0.1,
                help="Suhu proses operasional mesin.",
            )

        with col2:
            rotational_speed = st.number_input(
                "Kecepatan putaran (RPM)",
                min_value=1000,
                max_value=3000,
                value=1500,
                step=10,
            )

            torque = st.number_input(
                "Torsi (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.5,
            )

        with col3:
            tool_wear = st.number_input(
                "Keausan alat (menit)",
                min_value=0,
                max_value=300,
                value=100,
                step=1,
            )

            product_type = st.selectbox(
                "Jenis produk",
                options=["L", "M", "H"],
                index=1,
                format_func=lambda value: {
                    "L": "L — Low",
                    "M": "M — Medium",
                    "H": "H — High",
                }[value],
            )

        submitted = st.form_submit_button(
            "🔍 Analisis Kondisi Mesin",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_data = create_input_data(
            air_temp=air_temp,
            process_temp=process_temp,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            product_type=product_type,
        )

        try:
            selected_model = models[selected_algorithm]

            probability, prediction = predict_failure(
                selected_model,
                input_data,
            )

            st.divider()
            show_section_title("Hasil Analisis")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Probabilitas kegagalan",
                    f"{probability:.2%}",
                )

            with metric2:
                st.metric(
                    "Ambang klasifikasi",
                    f"{THRESHOLD:.0%}",
                )

            with metric3:
                st.metric(
                    "Jenis produk",
                    product_type,
                )

            left, right = st.columns([1.4, 1])

            with left:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="status-risk">
                            <h3>⚠️ Potensi Kegagalan Terdeteksi</h3>
                            Skor probabilitas mencapai atau
                            melampaui ambang klasifikasi
                            {THRESHOLD:.0%}.
                            <br><br>
                            <b>Rekomendasi:</b> lakukan pemeriksaan
                            teknis terhadap kondisi mesin dan
                            parameter operasional.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown(
                        """
                        <div class="status-safe">
                            <h3>
                                ✓ Tidak Terindikasi Potensi Kegagalan
                            </h3>
                            Skor probabilitas berada di bawah
                            ambang klasifikasi yang ditentukan.
                            <br><br>
                            <b>Rekomendasi:</b> tetap lakukan
                            pemantauan dan perawatan rutin sesuai
                            prosedur.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with right:
                st.markdown("#### Indikator probabilitas")
                st.progress(probability)

                st.caption(
                    f"Skor model: {probability:.4f} "
                    f"({probability:.2%})"
                )

                if prediction == 1:
                    st.warning(
                        "Skor berada pada atau di atas ambang."
                    )
                else:
                    st.success("Skor berada di bawah ambang.")

            st.caption(
                f"Algoritma yang digunakan: {selected_algorithm}"
            )

            st.caption(
                "Hasil merupakan estimasi model machine learning, "
                "bukan jaminan kondisi aktual mesin. Probabilitas "
                "model belum tentu terkalibrasi. Lakukan pemeriksaan "
                "teknis jika ditemukan indikasi risiko."
            )

        except Exception as error:
            st.error("Prediksi gagal dilakukan.")
            st.exception(error)

    else:
        st.info(
            "Isi parameter mesin, pilih algoritma, lalu tekan "
            "**Analisis Kondisi Mesin** untuk melihat hasil prediksi."
        )

# =========================================================
# 8. HALAMAN EVALUASI MODEL
# =========================================================
elif page == "Evaluasi Model":

    show_section_title("Evaluasi & Perbandingan Algoritma")

    st.write(
        "Perbandingan performa Support Vector Machine (SVM) "
        "dan Gaussian Naive Bayes berdasarkan pengujian pada "
        "2.000 data uji dengan ambang klasifikasi 30%."
    )

    # Metrik hasil pengujian dari Google Colab
    comparison = pd.DataFrame(
        {
            "Model": ["SVM", "Naive Bayes"],
            "Accuracy": [0.9690, 0.9565],
            "Precision": [0.5341, 0.2889],
            "Recall": [0.6912, 0.1912],
            "F1-score": [0.6026, 0.2301],
            "ROC-AUC": [0.9655, 0.8468],
        }
    ).set_index("Model")

    # Ringkasan metrik utama
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Accuracy SVM",
            f"{comparison.loc['SVM', 'Accuracy']:.2%}",
        )

    with metric2:
        st.metric(
            "F1-score SVM",
            f"{comparison.loc['SVM', 'F1-score']:.2%}",
        )

    with metric3:
        st.metric(
            "Recall SVM",
            f"{comparison.loc['SVM', 'Recall']:.2%}",
        )

    st.success(
        "SVM memiliki F1-score tertinggi, yaitu 60,26%, "
        "sehingga menjadi model terbaik berdasarkan metrik ini."
    )

    # Tabel perbandingan
    st.markdown("#### Tabel Perbandingan Metrik")

    st.dataframe(
        comparison.style.format("{:.2%}"),
        use_container_width=True,
    )

    # Grafik perbandingan
    st.markdown("#### Visualisasi Metrik")

    metric_options = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    selected_metrics = st.multiselect(
        "Pilih metrik yang ingin dibandingkan",
        options=metric_options,
        default=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ],
    )

    if selected_metrics:
        chart_data = comparison[selected_metrics].T

        st.bar_chart(
            chart_data,
            y_label="Skor",
        )
    else:
        st.info(
            "Pilih minimal satu metrik untuk menampilkan grafik."
        )

    # Confusion matrix
    st.markdown("#### Confusion Matrix")

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**Support Vector Machine (SVM)**")

        svm_cm = pd.DataFrame(
            [[1891, 41], [21, 47]],
            index=[
                "Aktual: Tidak Gagal",
                "Aktual: Gagal",
            ],
            columns=[
                "Prediksi: Tidak Gagal",
                "Prediksi: Gagal",
            ],
        )

        st.dataframe(
            svm_cm,
            use_container_width=True,
        )

    with cm_col2:
        st.markdown("**Gaussian Naive Bayes**")

        nb_cm = pd.DataFrame(
            [[1900, 32], [55, 13]],
            index=[
                "Aktual: Tidak Gagal",
                "Aktual: Gagal",
            ],
            columns=[
                "Prediksi: Tidak Gagal",
                "Prediksi: Gagal",
            ],
        )

        st.dataframe(
            nb_cm,
            use_container_width=True,
        )

    st.caption(
        "Baris menunjukkan kelas aktual, sedangkan kolom "
        "menunjukkan kelas hasil prediksi."
    )

    # Interpretasi hasil
    st.markdown("#### Interpretasi Hasil")

    st.markdown(
        """
        - **Accuracy:** SVM memperoleh 96,90%, sedangkan
          Naive Bayes memperoleh 95,65%.
        - **Precision:** SVM memperoleh 53,41%, lebih tinggi
          daripada Naive Bayes sebesar 28,89%.
        - **Recall:** SVM memperoleh 69,12%, sedangkan
          Naive Bayes memperoleh 19,12%. SVM mendeteksi
          lebih banyak kasus kegagalan aktual.
        - **F1-score:** SVM memperoleh 60,26%, sedangkan
          Naive Bayes memperoleh 23,01%. SVM lebih seimbang
          dalam mengidentifikasi kelas kegagalan.
        - **ROC-AUC:** SVM memperoleh 96,55%, sedangkan
          Naive Bayes memperoleh 84,68%.
        """
    )

    st.markdown("#### Kesimpulan")

    st.write(
        "Berdasarkan hasil evaluasi, SVM menunjukkan performa "
        "yang lebih baik daripada Naive Bayes pada seluruh "
        "metrik yang dibandingkan. SVM juga mendeteksi 47 dari "
        "68 kasus kegagalan aktual, sedangkan Naive Bayes "
        "mendeteksi 13 dari 68 kasus. Oleh karena itu, SVM "
        "lebih direkomendasikan untuk prototipe ini apabila "
        "tujuan utamanya adalah mendeteksi potensi kegagalan mesin."
    )

    st.warning(
        "Data kegagalan jauh lebih sedikit daripada data "
        "tidak gagal. Karena itu, accuracy saja tidak cukup "
        "untuk menilai model. Recall dan F1-score kelas Gagal "
        "juga perlu diperhatikan."
    )

# =========================================================
# 9. HALAMAN INFORMASI SISTEM
# =========================================================
elif page == "Informasi Sistem":

    show_section_title("Informasi Sistem")

    st.write(
        "Smart Machine — AI Predictive Maintenance merupakan "
        "prototipe aplikasi machine learning untuk membantu "
        "mengidentifikasi indikasi potensi kegagalan mesin "
        "berdasarkan parameter operasional."
    )

    st.markdown("### Tujuan Sistem")

    st.write(
        "Menganalisis data operasional mesin dan memberikan "
        "indikasi potensi kegagalan sebagai pendukung pemantauan "
        "serta perawatan mesin."
    )

    st.markdown("### Dataset")

    st.markdown("**AI4I 2020 Predictive Maintenance Dataset**")

    st.write(
        "Dataset digunakan untuk melatih dan mengevaluasi model "
        "klasifikasi kondisi mesin."
    )

    st.markdown("### Parameter Input")

    features = pd.DataFrame(
        {
            "Parameter": [
                "Suhu udara",
                "Suhu proses",
                "Kecepatan putaran",
                "Torsi",
                "Keausan alat",
                "Jenis produk",
            ],
            "Keterangan": [
                "Suhu lingkungan mesin dalam Kelvin (K).",
                "Suhu proses operasional dalam Kelvin (K).",
                "Kecepatan putaran mesin dalam RPM.",
                "Torsi mesin dalam Newton-meter (Nm).",
                "Durasi keausan alat dalam menit.",
                "Kategori produk L, M, atau H.",
            ],
        }
    )

    st.dataframe(
        features,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Algoritma yang Tersedia")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Support Vector Machine (SVM)")

        st.write(
            "SVM mempelajari batas pemisah antarkelas berdasarkan "
            "pola pada data. Model yang digunakan dalam aplikasi "
            "menerapkan kernel RBF dan mendukung estimasi probabilitas."
        )

    with col2:
        st.markdown("#### Gaussian Naive Bayes")

        st.write(
            "Gaussian Naive Bayes menggunakan teorema Bayes "
            "dengan asumsi bahwa fitur mengikuti distribusi "
            "Gaussian pada setiap kelas."
        )

    st.markdown("### Aturan Klasifikasi")

    st.write(
        f"Ambang klasifikasi ditetapkan sebesar {THRESHOLD:.0%}. "
        "Jika skor probabilitas kegagalan mencapai atau melampaui "
        "ambang tersebut, hasil diklasifikasikan sebagai "
        "terindikasi potensi kegagalan."
    )

    st.markdown("### Keterbatasan")

    st.warning(
        "Hasil merupakan estimasi model, bukan jaminan kondisi "
        "aktual mesin. Kinerja model dipengaruhi oleh kualitas "
        "data dan kesesuaian model dengan kondisi mesin nyata. "
        "Probabilitas belum tentu terkalibrasi. Pemeriksaan teknis "
        "tetap diperlukan sebelum mengambil keputusan perawatan."
    )


# =========================================================
# 10. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        Smart Machine — AI Predictive Maintenance<br>
        Prototipe machine learning untuk analisis indikasi
        kegagalan mesin
    </div>
    """,
    unsafe_allow_html=True,
)
