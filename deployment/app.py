"""
Streamlit App for Wellness Tourism Package Prediction
This application allows users to input customer data and predict
whether they will purchase the Wellness Tourism Package.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# TODO: Replace with your Hugging Face username
HF_USERNAME = "BaskaranAIExpert"  # Change this!

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# Download and load the model
@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/wellness-tourism-model",
            filename="wellness_tourism_model_v1.joblib"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model is uploaded to Hugging Face Hub and the username is correct.")
        return None

# Load model
model = load_model()

# Streamlit UI
st.title("✈️ Wellness Tourism Package Prediction App")
st.markdown("""
This application predicts whether a customer will purchase the **Wellness Tourism Package** 
based on their profile and interaction data. Enter the customer information below to get a prediction.
""")

if model is None:
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Details")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", [
        "Salaried", "Freelancer", "Small Business", "Large Business", "Other"
    ])
    designation = st.selectbox("Designation", [
        "Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"
    ])
    monthly_income = st.number_input(
        "Monthly Income (₹)", 
        min_value=0, 
        max_value=1000000, 
        value=50000, 
        step=1000
    )
    
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    number_of_trips = st.number_input(
        "Number of Trips (Annual Average)", 
        min_value=0, 
        max_value=20, 
        value=2, 
        step=1
    )
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("👨‍👩‍👧‍👦 Travel Details")
    
    number_of_persons = st.number_input(
        "Number of Persons Visiting", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1
    )
    number_of_children = st.number_input(
        "Number of Children Visiting (Below 5 years)", 
        min_value=0, 
        max_value=5, 
        value=0, 
        step=1
    )
    preferred_property_star = st.selectbox(
        "Preferred Property Star Rating", 
        [3, 4, 5], 
        index=1
    )
    
    st.subheader("📞 Interaction Details")
    
    type_of_contact = st.selectbox(
        "Type of Contact", 
        ["Company Invited", "Self Inquiry"]
    )
    product_pitched = st.selectbox(
        "Product Pitched", 
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score", 
        min_value=1, 
        max_value=5, 
        value=3, 
        step=1
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", 
        min_value=0, 
        max_value=10, 
        value=2, 
        step=1
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", 
        min_value=0.0, 
        max_value=60.0, 
        value=10.0, 
        step=0.5
    )

# Encode categorical variables (matching the preprocessing in prep.py)
def encode_categorical(value, category_type):
    """Encode categorical values to match training data encoding"""
    encodings = {
        'Gender': {'Male': 0, 'Female': 1},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'TypeofContact': {'Company Invited': 0, 'Self Inquiry': 1},
        'CityTier': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
        'Occupation': {
            'Salaried': 0, 'Freelancer': 1, 'Small Business': 2, 
            'Large Business': 3, 'Other': 4
        },
        'Designation': {
            'Executive': 0, 'Manager': 1, 'Senior Manager': 2,
            'AVP': 3, 'VP': 4, 'Other': 5
        },
        'ProductPitched': {
            'Basic': 0, 'Standard': 1, 'Deluxe': 2,
            'Super Deluxe': 3, 'King': 4
        }
    }
    return encodings.get(category_type, {}).get(value, 0)

# Assemble input into DataFrame
if st.button("🔮 Predict Purchase Likelihood", type="primary"):
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': encode_categorical(type_of_contact, 'TypeofContact'),
        'CityTier': encode_categorical(city_tier, 'CityTier'),
        'Occupation': encode_categorical(occupation, 'Occupation'),
        'Gender': encode_categorical(gender, 'Gender'),
        'NumberOfPersonVisiting': number_of_persons,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': encode_categorical(marital_status, 'MaritalStatus'),
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children,
        'Designation': encode_categorical(designation, 'Designation'),
        'MonthlyIncome': monthly_income,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'ProductPitched': encode_categorical(product_pitched, 'ProductPitched'),
        'NumberOfFollowups': number_of_followups,
        'DurationOfPitch': duration_of_pitch
    }])
    
    try:
        # Get expected columns from the preprocessing step in the pipeline
        # The model is a Pipeline with a ColumnTransformer as the first step
        expected_cols = None
        if hasattr(model, 'steps') and len(model.steps) > 0:
            preprocessor = model.steps[0][1]  # Get the ColumnTransformer
            if hasattr(preprocessor, 'feature_names_in_'):
                expected_cols = list(preprocessor.feature_names_in_)
        
        # If model expects 'Unnamed: 0', add it (workaround for current model)
        # This will be fixed when the model is retrained without this column
        if expected_cols and 'Unnamed: 0' in expected_cols:
            if 'Unnamed: 0' not in input_data.columns:
                input_data['Unnamed: 0'] = 0
        
        # Reorder columns to match expected order if available
        if expected_cols:
            # Ensure all expected columns are present
            for col in expected_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            # Select columns in the expected order
            input_data = input_data[expected_cols]
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.success(f"✅ **The customer is LIKELY to purchase the Wellness Tourism Package!**")
            st.info(f"Confidence: {prediction_proba[1]*100:.2f}%")
        else:
            st.warning(f"❌ **The customer is NOT LIKELY to purchase the Wellness Tourism Package.**")
            st.info(f"Confidence: {prediction_proba[0]*100:.2f}%")
        
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("Probability of Purchase", f"{prediction_proba[1]*100:.2f}%")
        with col_prob2:
            st.metric("Probability of No Purchase", f"{prediction_proba[0]*100:.2f}%")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ for Visit with Us | MLOps Pipeline</p>
</div>
""", unsafe_allow_html=True)

