"""
Streamlit App for Wellness Tourism Package Prediction
======================================================

This application provides a user-friendly web interface for predicting
whether a customer will purchase the Wellness Tourism Package.

Features:
- Interactive input forms for customer data
- Real-time prediction with confidence scores
- Professional UI with clear visualizations

Author: Baskaran Radhakrishnan
Date: 2026
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Streamlit for web application framework
import streamlit as st

# Data manipulation
import pandas as pd

# Model loading and prediction
from huggingface_hub import hf_hub_download
import joblib


# ============================================================================
# SECTION 2: CONFIGURATION AND CONSTANTS
# ============================================================================

# Hugging Face Configuration
HF_USERNAME = "BaskaranAIExpert"
MODEL_REPO = "wellness-tourism-model"
MODEL_FILENAME = "wellness_tourism_model_v1.joblib"

# Page Configuration
PAGE_TITLE = "Wellness Tourism Package Prediction"
PAGE_ICON = "✈️"
LAYOUT = "wide"


# ============================================================================
# SECTION 3: CATEGORICAL ENCODING MAPPINGS
# ============================================================================

# Categorical value encodings (must match training data preprocessing)
CATEGORICAL_ENCODINGS = {
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


# ============================================================================
# SECTION 4: PAGE CONFIGURATION
# ============================================================================

def configure_page():
    """
    Configures Streamlit page settings.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )


# ============================================================================
# SECTION 5: MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(hf_username, model_repo, model_filename):
    """
    Loads the trained model from Hugging Face Hub.
    Uses caching to avoid reloading on every interaction.
    
    Args:
        hf_username (str): Hugging Face username
        model_repo (str): Model repository name
        model_filename (str): Name of the model file
        
    Returns:
        tuple: (model, error_message) - Model object and error message (if any)
    """
    try:
        with st.spinner("Loading model from Hugging Face Hub..."):
            model_path = hf_hub_download(
                repo_id=f"{hf_username}/{model_repo}",
                filename=model_filename
            )
            model = joblib.load(model_path)
        return model, None
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        return None, error_msg


# ============================================================================
# SECTION 6: CATEGORICAL ENCODING
# ============================================================================

def encode_categorical(value, category_type):
    """
    Encodes categorical values to match training data encoding.
    
    Args:
        value (str): Categorical value to encode
        category_type (str): Type of category (e.g., 'Gender', 'CityTier')
        
    Returns:
        int: Encoded value (defaults to 0 if not found)
    """
    return CATEGORICAL_ENCODINGS.get(category_type, {}).get(value, 0)


# ============================================================================
# SECTION 7: USER INPUT COLLECTION
# ============================================================================

def collect_customer_details():
    """
    Collects customer demographic and profile information.
    
    Returns:
        dict: Dictionary containing customer details
    """
    st.subheader("📋 Customer Details")
    
    customer_data = {
        'age': st.number_input("Age", min_value=18, max_value=100, value=35, step=1),
        'gender': st.selectbox("Gender", ["Male", "Female"]),
        'marital_status': st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        'occupation': st.selectbox("Occupation", [
            "Salaried", "Freelancer", "Small Business", "Large Business", "Other"
        ]),
        'designation': st.selectbox("Designation", [
            "Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"
        ]),
        'monthly_income': st.number_input(
            "Monthly Income (₹)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000
        ),
        'city_tier': st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"]),
        'number_of_trips': st.number_input(
            "Number of Trips (Annual Average)",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        ),
        'passport': st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'own_car': st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    }
    
    return customer_data


def collect_travel_details():
    """
    Collects travel-related information.
    
    Returns:
        dict: Dictionary containing travel details
    """
    st.subheader("👨‍👩‍👧‍👦 Travel Details")
    
    travel_data = {
        'number_of_persons': st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        ),
        'number_of_children': st.number_input(
            "Number of Children Visiting (Below 5 years)",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        ),
        'preferred_property_star': st.selectbox(
            "Preferred Property Star Rating",
            [3, 4, 5],
            index=1
        )
    }
    
    return travel_data


def collect_interaction_details():
    """
    Collects customer interaction and sales pitch information.
    
    Returns:
        dict: Dictionary containing interaction details
    """
    st.subheader("📞 Interaction Details")
    
    interaction_data = {
        'type_of_contact': st.selectbox(
            "Type of Contact",
            ["Company Invited", "Self Inquiry"]
        ),
        'product_pitched': st.selectbox(
            "Product Pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        ),
        'pitch_satisfaction_score': st.slider(
            "Pitch Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        ),
        'number_of_followups': st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        ),
        'duration_of_pitch': st.number_input(
            "Duration of Pitch (minutes)",
            min_value=0.0,
            max_value=60.0,
            value=10.0,
            step=0.5
        )
    }
    
    return interaction_data


# ============================================================================
# SECTION 8: DATA PREPARATION FOR PREDICTION
# ============================================================================

def prepare_input_data(customer_data, travel_data, interaction_data):
    """
    Prepares input data in the format expected by the model.
    
    Args:
        customer_data (dict): Customer demographic information
        travel_data (dict): Travel-related information
        interaction_data (dict): Interaction details
        
    Returns:
        pd.DataFrame: Prepared input data
    """
    input_data = pd.DataFrame([{
        'Age': customer_data['age'],
        'TypeofContact': encode_categorical(interaction_data['type_of_contact'], 'TypeofContact'),
        'CityTier': encode_categorical(customer_data['city_tier'], 'CityTier'),
        'Occupation': encode_categorical(customer_data['occupation'], 'Occupation'),
        'Gender': encode_categorical(customer_data['gender'], 'Gender'),
        'NumberOfPersonVisiting': travel_data['number_of_persons'],
        'PreferredPropertyStar': travel_data['preferred_property_star'],
        'MaritalStatus': encode_categorical(customer_data['marital_status'], 'MaritalStatus'),
        'NumberOfTrips': customer_data['number_of_trips'],
        'Passport': customer_data['passport'],
        'OwnCar': customer_data['own_car'],
        'NumberOfChildrenVisiting': travel_data['number_of_children'],
        'Designation': encode_categorical(customer_data['designation'], 'Designation'),
        'MonthlyIncome': customer_data['monthly_income'],
        'PitchSatisfactionScore': interaction_data['pitch_satisfaction_score'],
        'ProductPitched': encode_categorical(interaction_data['product_pitched'], 'ProductPitched'),
        'NumberOfFollowups': interaction_data['number_of_followups'],
        'DurationOfPitch': interaction_data['duration_of_pitch']
    }])
    
    return input_data


def align_input_with_model(input_data, model):
    """
    Aligns input data columns with model's expected feature order.
    
    Args:
        input_data (pd.DataFrame): Input data
        model: Trained model pipeline
        
    Returns:
        pd.DataFrame: Aligned input data
    """
    # Get expected columns from the preprocessing step in the pipeline
    expected_cols = None
    if hasattr(model, 'steps') and len(model.steps) > 0:
        preprocessor = model.steps[0][1]  # Get the ColumnTransformer
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_cols = list(preprocessor.feature_names_in_)
    
    # Handle 'Unnamed: 0' column if model expects it
    if expected_cols and 'Unnamed: 0' in expected_cols:
        if 'Unnamed: 0' not in input_data.columns:
            input_data['Unnamed: 0'] = 0
    
    # Reorder columns to match expected order
    if expected_cols:
        # Ensure all expected columns are present
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        # Select columns in the expected order
        input_data = input_data[expected_cols]
    
    return input_data


# ============================================================================
# SECTION 9: PREDICTION AND DISPLAY
# ============================================================================

def make_prediction(model, input_data):
    """
    Makes prediction using the trained model.
    
    Args:
        model: Trained model
        input_data (pd.DataFrame): Prepared input data
        
    Returns:
        tuple: (prediction, prediction_proba) - Prediction and probabilities
    """
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    return prediction, prediction_proba


def display_prediction_results(prediction, prediction_proba):
    """
    Displays prediction results with visualizations.
    
    Args:
        prediction (int): Predicted class (0 or 1)
        prediction_proba (np.array): Prediction probabilities
    """
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    # Display main prediction
    if prediction == 1:
        st.success(f"✅ **The customer is LIKELY to purchase the Wellness Tourism Package!**")
        st.info(f"**Confidence Level:** {prediction_proba[1]*100:.2f}%")
    else:
        st.warning(f"❌ **The customer is NOT LIKELY to purchase the Wellness Tourism Package.**")
        st.info(f"**Confidence Level:** {prediction_proba[0]*100:.2f}%")
    
    # Display probability metrics
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric(
            "Probability of Purchase",
            f"{prediction_proba[1]*100:.2f}%",
            delta=f"{prediction_proba[1]*100 - 50:.2f}%"
        )
    with col_prob2:
        st.metric(
            "Probability of No Purchase",
            f"{prediction_proba[0]*100:.2f}%",
            delta=f"{prediction_proba[0]*100 - 50:.2f}%"
        )
    
    # Display recommendation
    if prediction == 1:
        st.info("💡 **Recommendation:** This customer shows high purchase likelihood. Consider prioritizing follow-up communication.")
    else:
        st.info("💡 **Recommendation:** This customer shows low purchase likelihood. Consider alternative marketing strategies.")


# ============================================================================
# SECTION 10: MAIN APPLICATION UI
# ============================================================================

def render_header():
    """
    Renders the application header and description.
    """
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("""
    This application predicts whether a customer will purchase the **Wellness Tourism Package**
    based on their profile and interaction data. Enter the customer information below to get a prediction.
    """)


def render_footer():
    """
    Renders the application footer.
    """
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>Built with ❤️ for Visit with Us</strong></p>
        <p>MLOps Pipeline | Production Ready</p>
        <p style='font-size: 0.8em;'>Model Version: v1.0 | Last Updated: 2024</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """
    Main application function that orchestrates the Streamlit UI.
    """
    # Configure page
    configure_page()
    
    # Render header
    render_header()
    
    # Load model
    model, error = load_model(HF_USERNAME, MODEL_REPO, MODEL_FILENAME)
    
    # Handle model loading error
    if model is None:
        st.error(f"⚠️ {error}")
        st.info("💡 Please ensure:")
        st.info("1. The model is uploaded to Hugging Face Hub")
        st.info("2. The username is correct in the configuration")
        st.info("3. You have internet connectivity")
        st.stop()
    
    # Display success message
    st.success("✓ Model loaded successfully!")
    
    # Create input form layout
    col1, col2 = st.columns(2)
    
    with col1:
        customer_data = collect_customer_details()
    
    with col2:
        travel_data = collect_travel_details()
        interaction_data = collect_interaction_details()
    
    # Prediction button
    if st.button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = prepare_input_data(customer_data, travel_data, interaction_data)
            
            # Align with model expectations
            input_data = align_input_with_model(input_data, model)
            
            # Make prediction
            prediction, prediction_proba = make_prediction(model, input_data)
            
            # Display results
            display_prediction_results(prediction, prediction_proba)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check the input values and try again.")
    
    # Render footer
    render_footer()


# ============================================================================
# SECTION 11: SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()