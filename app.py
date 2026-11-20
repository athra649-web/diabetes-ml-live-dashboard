
import streamlit as st

st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("AI Diabetes Prediction Challenge")

st.write(
    "Explore how five machine learning models "
    "predict diabetes using the same input features."
)

st.info(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)

st.divider()

st.header("Enter Patient Information")

# Participant nickname
student_name = st.text_input(
    "Participant Name (Nickname)",
    placeholder="Enter your nickname"
)

st.subheader("Patient Features")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=17, value=0
    )

    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=1, max_value=300, value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=1, max_value=200, value=72
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=1, max_value=100, value=29
    )

with col2:
    insulin = st.number_input(
        "Insulin (mu U/mL)", min_value=1, max_value=900, value=125
    )

    bmi = st.number_input(
        "BMI", min_value=1.0, max_value=70.0, value=28.5
    )

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47
    )

    age = st.number_input(
        "Age", min_value=21, max_value=100, value=30
    )

predict_button = st.button(
    "Predict Diabetes",
    type="primary",
    use_container_width=True
)


# Load trained models
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    return joblib.load("diabetes_5_models.joblib")

model_package = load_models()

models = model_package["models"]
imputer = model_package["imputer"]
scaler = model_package["scaler"]
features = model_package["features"]

# Generate predictions when the button is clicked
import gspread
from google.oauth2.service_account import Credentials

SPREADSHEET_ID = "1f4_F7eF4_Qc4_4ferXlvtmPpJXu6cob_WjGkzi4cd-M"

@st.cache_resource
def connect_google_sheets():

    # Read credentials securely from Streamlit Secrets
    service_account_info = dict(
        st.secrets["gcp_service_account"]
    )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets"
    ]

    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes
    )

    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    return spreadsheet.worksheet("Results")



from datetime import datetime, timezone

def save_prediction_to_sheets(participant, predictions):

    worksheet = connect_google_sheets()

    # Convert predictions into a dictionary
    prediction_dict = {
        item["Model"]: item["Prediction"]
        for item in predictions
    }

    # Prepare a new row
    row = [
        datetime.now(timezone.utc).isoformat(),
        participant,
        prediction_dict.get("KNN", ""),
        prediction_dict.get("BNB", ""),
        prediction_dict.get("DT", ""),
        prediction_dict.get("LR", ""),
        prediction_dict.get("SVM", "")
    ]

    # Append the new row to Google Sheets
    worksheet.append_row(
        row,
        value_input_option="RAW"
    )

    return True


if predict_button:

    if not student_name.strip():
        st.warning("Please enter a participant nickname.")

    else:
        # Prepare input data in correct feature order
        patient_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }

        input_df = pd.DataFrame(
            [patient_data],
            columns=features
        )

        # Apply the same preprocessing used during training
        input_processed = input_df.copy()

        zero_columns = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        input_processed[zero_columns] = (
            input_processed[zero_columns]
            .replace(0, float("nan"))
        )

        input_imputed = imputer.transform(input_processed)
        input_scaled = scaler.transform(input_imputed)

        # Predict with all five models
        prediction_results = []

        for name, model in models.items():
            prediction = int(model.predict(input_scaled)[0])
            probability = float(
                model.predict_proba(input_scaled)[0, 1]
            )

            prediction_results.append({
                "Model": name,
                "Prediction": (
                    "Positive" if prediction == 1 else "Negative"
                ),
                "Model Score": round(probability * 100, 2)
            })

        st.success(
            f"Prediction completed for {student_name}!"
        )

        st.subheader("Five Models Prediction Results")

        results_table = pd.DataFrame(prediction_results)

        # Save predictions to Google Sheets
        try:
            save_prediction_to_sheets(
                student_name.strip(),
                prediction_results
            )
            st.success("Results saved to Live Dashboard!")
        except Exception as e:
            st.warning(
                "Prediction succeeded, but results were not saved."
            )


        st.dataframe(
            results_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Model scores are estimates, not clinical "
            "probabilities or medical diagnoses."
        )


