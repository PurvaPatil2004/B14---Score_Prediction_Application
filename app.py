import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'knn.pkl': {e}")
    st.stop()

# Custom CSS for shadow and card layout
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2222;
    }
    .prediction-box {
        padding: 24px;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-top: 25px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🎓 Student Exam Performance Predictor")
st.write("Enter the student details below to generate a prediction using your KNeighborsRegressor model.")

# Input Form with Layout Containers
with st.form("prediction_form"):
    st.subheader("📝 Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course", ["Science", "Arts", "Commerce", "Engineering", "Other"])
        study_hours = st.number_input("Study Hours (Daily)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=80)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.number_input("Sleep Hours (Nightly)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
        study_method = st.selectbox("Study Method", ["Self-study", "Online videos", "Group study", "Coaching"])
        facility_rating = st.slider("Facility Rating (1-5)", min_value=1, max_value=5, value=3)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

    submitted = st.form_submit_button("Predict Score")

# Prediction Logic with Explicit Mapping & Casting
if submitted:
    # Categorical Mappings
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    course_map = {"Science": 0, "Arts": 1, "Commerce": 2, "Engineering": 3, "Other": 4}
    internet_map = {"Yes": 1, "No": 0}
    sleep_quality_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
    study_method_map = {"Self-study": 0, "Online videos": 1, "Group study": 2, "Coaching": 3}
    exam_difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}

    # Create DataFrame and explicitly convert everything to float
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender_map[gender]],
        'course': [course_map[course]],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_map[internet_access]],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality_map[sleep_quality]],
        'study_method': [study_method_map[study_method]],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty_map[exam_difficulty]]
    }).astype(float)
    
    try:
        prediction = model.predict(input_data)
        st.markdown(f"""
            <div class="prediction-box">
                <h4 style="color: #333333; margin-bottom: 5px;">Predicted Result</h4>
                <h1 style="color: #ff4b4b; margin-top: 0;">{prediction[0]:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")
