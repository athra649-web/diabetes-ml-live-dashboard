
import streamlit as st
import pandas as pd
import joblib
import gspread

from datetime import datetime, timezone
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

SPREADSHEET_ID = "1f4_F7eF4_Qc4_4ferXlvtmPpJXu6cob_WjGkzi4cd-M"

@st.cache_resource
def load_models():
    return joblib.load("diabetes_5_models.joblib")

@st.cache_resource
def connect_google_sheets():
    credentials = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets"
        ]
    )

    client = gspread.authorize(credentials)
    return client.open_by_key(
        SPREADSHEET_ID
    ).worksheet("Results")

def save_prediction_to_sheets(participant, predictions):
    worksheet = connect_google_sheets()

    result_dict = {
        item["Model"]: item["Prediction"]
        for item in predictions
    }

    worksheet.append_row([
        datetime.now(timezone.utc).isoformat(),
        participant,
        result_dict.get("KNN", ""),
        result_dict.get("BNB", ""),
        result_dict.get("DT", ""),
        result_dict.get("LR", ""),
        result_dict.get("SVM", "")
    ], value_input_option="RAW")

def load_live_results():
    records = connect_google_sheets().get_all_records()

    columns = [
        "Timestamp", "Participant",
        "KNN", "BNB", "DT", "LR", "SVM"
    ]

    return pd.DataFrame(records, columns=columns)

# Navigation
st.sidebar.title("AI Diabetes Challenge")

page = st.sidebar.radio(
    "Navigation",
    ["Student Prediction", "Live Dashboard"]
)

st.sidebar.caption(
    "Educational demonstration only"
)

if page == "Student Prediction":

    st.title("AI Diabetes Prediction Challenge")
    st.write("Compare five machine learning models.")

    st.info(
        "Use fictional patient data only. "
        "This application is not a diagnostic tool."
    )

    with st.form("patient_form"):

        name = st.text_input(
            "Participant Nickname",
            placeholder="Student01"
        )

        st.subheader("Patient Features")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies", 0, 17, 0
            )
            glucose = st.number_input(
                "Glucose (mg/dL)", 1, 300, 120
            )
            bp = st.number_input(
                "Blood Pressure (mmHg)", 1, 200, 72
            )
            skin = st.number_input(
                "Skin Thickness (mm)", 1, 100, 29
            )

        with col2:
            insulin = st.number_input(
                "Insulin (mu U/mL)", 1, 900, 125
            )
            bmi = st.number_input(
                "BMI", 1.0, 70.0, 28.5
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                0.0, 3.0, 0.47
            )
            age = st.number_input(
                "Age", 21, 100, 30
            )

        submitted = st.form_submit_button(
            "Predict Diabetes",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not name.strip():
            st.warning("Please enter a nickname.")

        else:
            package = load_models()

            patient = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age
            }

            input_df = pd.DataFrame(
                [patient],
                columns=package["features"]
            )

            processed = input_df.copy()

            zero_columns = [
                "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI"
            ]

            processed[zero_columns] = (
                processed[zero_columns].replace(
                    0, float("nan")
                )
            )

            imputed = package["imputer"].transform(processed)
            scaled = package["scaler"].transform(imputed)

            predictions = []

            for model_name, model in package["models"].items():

                prediction = int(model.predict(scaled)[0])
                score = float(
                    model.predict_proba(scaled)[0, 1]
                )

                predictions.append({
                    "Model": model_name,
                    "Prediction": (
                        "Positive" if prediction else "Negative"
                    ),
                    "Model Score": round(score * 100, 2)
                })

            st.success(
                f"Prediction completed for {name.strip()}!"
            )

            st.subheader("Five Models Prediction Results")

            st.dataframe(
                pd.DataFrame(predictions),
                hide_index=True,
                use_container_width=True
            )

            try:
                save_prediction_to_sheets(
                    name.strip(), predictions
                )
                st.success("Results saved to Live Dashboard!")

            except Exception as e:
                st.warning("Results could not be saved.")
                st.error(f"Error type: {type(e).__name__}")

            st.caption(
                "Model scores are not calibrated clinical "
                "risk probabilities or diagnoses."
            )

else:

    @st.fragment(run_every="5s")
    def show_dashboard():

        st.title("Live Diabetes Prediction Dashboard")
        st.caption("Live classroom results | Fictional cases")

        try:
            df = load_live_results()

            if df.empty:
                st.info("Waiting for submissions...")
                return

            model_columns = ["KNN", "BNB", "DT", "LR", "SVM"]

            positive = (
                df[model_columns] == "Positive"
            ).sum().sum()

            negative = (
                df[model_columns] == "Negative"
            ).sum().sum()

            c1, c2, c3 = st.columns(3)

            c1.metric("Submissions", len(df))
            c2.metric("Positive Predictions", int(positive))
            c3.metric("Negative Predictions", int(negative))

            st.subheader("Live Student Predictions")

            st.dataframe(
                df[["Participant"] + model_columns],
                hide_index=True,
                use_container_width=True
            )

            st.subheader("Model Prediction Comparison")

            chart_data = pd.DataFrame({
                "Positive": (
                    df[model_columns] == "Positive"
                ).sum(),
                "Negative": (
                    df[model_columns] == "Negative"
                ).sum()
            })

            st.bar_chart(chart_data)

            st.caption(
                "Automatically refreshes every 5 seconds. "
                "Counts represent model predictions, not diagnoses."
            )

        except Exception as e:
            st.warning("Dashboard could not load results.")
            st.error(f"Error type: {type(e).__name__}")

    show_dashboard()
