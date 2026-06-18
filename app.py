import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AIDS Survival Prediction",
    layout="wide"
)

st.title("AIDS Survival Prediction")
st.markdown(
    "Prediksi survival pasien menggunakan model Random Survival Forest."
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    rsf_model = joblib.load("rsf_model.pkl")
    cox_model = joblib.load("cox_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")

    return rsf_model, cox_model, scaler, feature_cols


try:
    rsf_model, cox_model, scaler, feature_cols = load_model()

    st.success(
        f"Model berhasil dimuat ({len(feature_cols)} fitur)."
    )

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================================
# SIDEBAR INPUT
# ==========================================
st.sidebar.header("Input Parameter Pasien")


def get_value(option):
    return int(option.split(" - ")[0])


offtrt = st.sidebar.selectbox(
    "Off Treatment",
    [
        "0 - Tidak",
        "1 - Ya"
    ]
)

symptom = st.sidebar.selectbox(
    "Symptom",
    [
        "0 - Tidak Bergejala",
        "1 - Bergejala"
    ]
)

z30 = st.sidebar.selectbox(
    "Z30 (Penggunaan ZDV 30 hari sebelum test)",
    [
        "0 - Tidak",
        "1 - Ya"
    ]
)

drugs = st.sidebar.selectbox(
    "Riwayat IV Drug",
    [
        "0 - Tidak",
        "1 - Ya"
    ]
)

oprior = st.sidebar.selectbox(
    "ARV Non-ZDV Sebelumnya",
    [
        "0 - Tidak",
        "1 - Ya"
    ]
)

trt = st.sidebar.selectbox(
    "Treatment Group",
    [
        "0 - (ZDV only)",
        "1 - (ZDV + ddl)",
        "2 - (ZDV + Zal)",
        "3 - (ddl only)"
    ]
)

cd40 = st.sidebar.number_input(
    "CD4 Baseline (cd40)",
    min_value=0.0,
    value=350.0
)

cd420 = st.sidebar.number_input(
    "CD4 Week 20 (cd420)",
    min_value=0.0,
    value=350.0
)

# ==========================================
# INPUT DATAFRAME
# ==========================================
input_data = {
    "offtrt": get_value(offtrt),
    "symptom": get_value(symptom),
    "z30": get_value(z30),
    "drugs": get_value(drugs),
    "oprior": get_value(oprior),
    "trt": get_value(trt),
    "cd40": cd40,
    "cd420": cd420
}

# ==========================================
# PREDICTION
# ==========================================
if st.sidebar.button(
    "Prediksi",
    type="primary"
):

    try:

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_cols]

        input_scaled = scaler.transform(input_df)

        cox_risk_score = float(
            cox_model.predict_partial_hazard(
                input_df
            ).iloc[0]
        )

        surv_func = rsf_model.predict_survival_function(
            input_scaled
        )[0]

        surv_func_cox = cox_model.predict_survival_function(
            input_df
        )
        
        def get_rsf_prob(surv_function, day):
            try:
                return float(surv_function(day))
            except:
                return float(surv_function.y[-1])


        def get_cox_prob(surv_df, day):
            available_times = surv_df.index[
                surv_df.index <= day
            ]

            if len(available_times) == 0:
                return float(
                    surv_df.iloc[0, 0]
                )

            return float(
                surv_df.loc[
                    available_times[-1]
                ].iloc[0]
            )
        # ==========================================
        # MEDIAN SURVIVAL TIME
        # ==========================================
        median_time = None

        for time, prob in zip(
            surv_func.x,
            surv_func.y
        ):
            if prob <= 0.5:
                median_time = time
                break

        # ==========================================
        # SURVIVAL PROBABILITIES
        # ==========================================
        def get_survival_prob(surv_function, day):
            try:
                return float(surv_function(day))
            except:
                return float(surv_function.y[-1])

        # RSF
        rsf_1y = get_rsf_prob(
            surv_func,
            365
        )

        rsf_2y = get_rsf_prob(
            surv_func,
            730
        )

        rsf_3y = get_rsf_prob(
            surv_func,
            1095
        )

        # Cox
        cox_1y = get_cox_prob(
            surv_func_cox,
            365
        )

        cox_2y = get_cox_prob(
            surv_func_cox,
            730
        )

        cox_3y = get_cox_prob(
            surv_func_cox,
            1095
        )

        # Ensemble
        ens_1y = (rsf_1y + cox_1y) / 2
        ens_2y = (rsf_2y + cox_2y) / 2
        ens_3y = (rsf_3y + cox_3y) / 2

        # ==========================================
        # RISK CATEGORY
        # ==========================================
        if ens_1y >= 0.85:
            risk_category = "RENDAH"
            icon = "🟢"

        elif ens_1y >= 0.70:
            risk_category = "SEDANG"
            icon = "🟡"

        else:
            risk_category = "TINGGI"
            icon = "🔴"

        st.divider()

        st.subheader("Probabilitas Survival")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "1 Tahun",
                f"{ens_1y*100:.2f}%"
            )

        with col2:
            st.metric(
                "2 Tahun",
                f"{ens_2y*100:.2f}%"
            )

        with col3:
            st.metric(
                "3 Tahun",
                f"{ens_3y*100:.2f}%"
            )

        st.divider()

        # ==========================================
        # SURVIVAL CURVE
        # ==========================================
        st.subheader("Kurva Survival")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=surv_func.x,
                y=surv_func.y,
                mode="lines",
                name="RSF"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=surv_func_cox.index,
                y=surv_func_cox.iloc[:, 0],
                mode="lines",
                name="Cox PH"
            )
        )

        fig.add_vline(
            x=365,
            line_dash="dash"
        )

        fig.add_vline(
            x=730,
            line_dash="dash"
        )

        fig.add_vline(
            x=1095,
            line_dash="dash"
        )

        fig.update_layout(
            xaxis_title="Waktu (Hari)",
            yaxis_title="Probabilitas Survival",
            yaxis=dict(range=[0, 1]),
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # SUMMARY
        # ==========================================
        st.divider()

        st.subheader("Interpretasi")

        st.write(
            f"""
            Berdasarkan model Random Survival Forest:

            - Probabilitas survival 1 tahun: **{ens_1y*100:.2f}%**
            - Probabilitas survival 2 tahun: **{ens_2y*100:.2f}%**
            - Probabilitas survival 3 tahun: **{ens_3y*100:.2f}%**
            - Hazard Ratio (Cox PH): **{cox_risk_score:.4f}**
            - Kategori Risiko: **{risk_category}**

            Pasien diperkirakan memiliki peluang bertahan hidup
            sebesar **{ens_1y*100:.2f}%** pada tahun pertama,
            **{ens_2y*100:.2f}%** pada tahun kedua,
            dan **{ens_3y*100:.2f}%** pada tahun ketiga observasi.
            """
        )

    except Exception as e:
        st.error(
            f"Prediksi gagal: {e}"
        )