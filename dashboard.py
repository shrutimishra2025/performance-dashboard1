import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("RedBull Talent Management Dashboard")

# Load dataset with caching and detailed error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce resource usage
def load_data():
    st.write("Attempting to load dataset from URL...")
    try:
        # Use raw string to handle URL encoding
        data = pd.read_csv(r"https://raw.githubusercontent.com/shrutimishra2025/performance-dashboard1/refs/heads/main/employee_data_100%20(1).csv")
        st.write("Dataset URL accessed successfully.")
        data.columns = data.columns.str.strip()
        required_cols = ['EmployeeID', 'Department', 'Diplomatic', 'Balanced', 'Sociable', 'Innovative', 
                         'Performance_H1', 'Performance_H2', 'Potential', 'Tenure']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}. Available columns: {list(data.columns)}")
            st.stop()
        for col in ['Diplomatic', 'Balanced', 'Sociable', 'Innovative', 'Performance_H1', 'Performance_H2', 'Potential']:
            numeric_data = pd.to_numeric(data[col], errors='coerce')
            if numeric_data.isna().any():
                invalid_rows = data[numeric_data.isna()][[col, 'EmployeeID']]
                st.error(f"Non-numeric data found in column '{col}': {invalid_rows.to_dict()}")
                st.stop()
        st.write("Dataset loaded and validated successfully.")
        return data
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}. Please check the URL or CSV format. Error details: {str(e)}")
        st.stop()

data = load_data()

# Precomputed personas (replace with your Colab output)
personas = {
    'Sales': {'Diplomatic': 85.80, 'Balanced': 87.10, 'Sociable': 78.30, 'Innovative': 83.00},
    'Retail': {'Diplomatic': 78.60, 'Balanced': 81.00, 'Sociable': 87.30, 'Innovative': 71.00},
    'Product': {'Diplomatic': 70.50, 'Balanced': 86.50, 'Sociable': 76.70, 'Innovative': 90.60},
    'HR': {'Diplomatic': 86.50, 'Balanced': 90.50, 'Sociable': 80.60, 'Innovative': 76.30},
    'Legal': {'Diplomatic': 90.40, 'Balanced': 85.60, 'Sociable': 70.40, 'Innovative': 80.60}
}

# Display personas
st.header("Department Personas")
persona_df = pd.DataFrame(personas).T
st.dataframe(persona_df.round(2))

# Function to calculate distance (weighted Euclidean)
def match_candidate(diplomatic, balanced, sociable, innovative, dept_weights):
    if not (0 <= diplomatic <= 100 and 0 <= balanced <= 100 and 0 <= sociable <= 100 and 0 <= innovative <= 100):
        return "Invalid", {"Invalid": 0}
    if diplomatic == balanced == sociable == innovative == 0:
        return "Invalid", {"Invalid": 0}
    
    candidate = np.array([diplomatic, balanced, sociable, innovative])
    distances = {}
    for dept in personas:
        persona_scores = np.array([personas[dept][m] for m in ['Diplomatic', 'Balanced', 'Sociable', 'Innovative']])
        weights = dept_weights.get(dept, np.array([0.25, 0.25, 0.25, 0.25]))
        diff = (candidate - persona_scores) * weights
        distance = np.sqrt(np.sum(diff ** 2))
        distances[dept] = distance
    best_match = min(distances, key=distances.get)
    return best_match, distances

# Default weights
default_weights = {
    'Sales': np.array([0.2, 0.2, 0.4, 0.2]),  # Emphasize Sociable
    'Retail': np.array([0.25, 0.25, 0.25, 0.25]),
    'Product': np.array([0.2, 0.2, 0.1, 0.5]),  # Emphasize Innovative
    'HR': np.array([0.3, 0.3, 0.2, 0.2]),
    'Legal': np.array([0.4, 0.2, 0.2, 0.2])  # Emphasize Diplomatic
}

# Learning and career plans based on 9-box placement
learning_plans = {
    "Low Performance, Low Potential": "Basic skills training, reassignment consideration.",
    "Low Performance, Moderate Potential": "Coaching, targeted skill development.",
    "Low Performance, High Potential": "Mentoring, leadership training.",
    "Moderate Performance, Low Potential": "Performance improvement plan, role fit assessment.",
    "Moderate Performance, Moderate Potential": "Cross-training, project exposure.",
    "Moderate Performance, High Potential": "Advanced leadership program, stretch assignments.",
    "High Performance, Low Potential": "Recognition, retention focus.",
    "High Performance, Moderate Potential": "Specialized training, lateral move options.",
    "High Performance, High Potential": "Succession planning, executive coaching."
}

# Succession pipeline candidates
def get_succession_pipeline(data, personas):
    try:
        data['Potential_Score'] = data['Potential']  # Use pre-assigned Potential
        top_10 = data.nlargest(int(len(data) * 0.1), 'Potential_Score')  # Top 10%
        return top_10[['EmployeeID', 'Department', 'Potential_Score']]
    except Exception as e:
        st.error(f"Succession pipeline error: {str(e)}")
        return pd.DataFrame(columns=['EmployeeID', 'Department', 'Potential_Score'])

# 9-box placement
def get_9box_placement(performance, potential):
    perf_cat = "Low" if performance < 70 else "Moderate" if performance < 85 else "High"
    pot_cat = "Low" if potential < 70 else "Moderate" if potential < 85 else "High"
    return f"{perf_cat} Performance, {pot_cat} Potential"

# Main dashboard
st.header("Employee Performance and 9-Box Matrix")

# Select half-year
half_year = st.selectbox("Select Half-Year", ["H1", "H2"])
performance_col = f"Performance_{half_year}"

# Display employee data with error handling
st.subheader("Employee Performance Scores")
try:
    employees_df = data[['EmployeeID', 'Department', 'Diplomatic', 'Balanced', 'Sociable', 'Innovative', performance_col, 'Potential']].copy()
    employees_df['Best_Match'] = employees_df.apply(lambda row: match_candidate(row['Diplomatic'], row['Balanced'], 
                                                                              row['Sociable'], row['Innovative'], default_weights)[0], axis=1)
    employees_df['Performance'] = pd.to_numeric(employees_df[performance_col], errors='coerce')
    if employees_df['Performance'].isna().any():
        st.error("Non-numeric performance data detected. Please check the CSV.")
        st.stop()
    employees_df['Potential'] = pd.to_numeric(employees_df['Potential'], errors='coerce')
    if employees_df['Potential'].isna().any():
        st.error("Non-numeric potential data detected. Please check the CSV.")
        st.stop()
    employees_df['9-Box_Placement'] = employees_df.apply(lambda row: get_9box_placement(row['Performance'], row['Potential']), axis=1)
    employees_df['Learning_Plan'] = employees_df['9-Box_Placement'].map(learning_plans)
    st.dataframe(employees_df.round(2))
except Exception as e:
    st.error(f"Error processing employee data: {str(e)}")

# 9-Box Matrix Visualization
st.subheader("9-Box McKinsey Matrix")
try:
    matrix = pd.DataFrame(index=["High", "Moderate", "Low"], columns=["High", "Moderate", "Low"])
    for _, row in employees_df.iterrows():
        if pd.notna(row['9-Box_Placement']):
            perf, pot = row['9-Box_Placement'].split(", ")
            perf = perf.replace(" Performance", "")
            matrix.at[pot.replace(" Potential", ""), perf] = matrix.at[pot.replace(" Potential", ""), perf] + 1 if pd.notna(matrix.at[pot.replace(" Potential", ""), perf]) else 1
    st.table(matrix.fillna(0).astype(int))
except Exception as e:
    st.error(f"Error generating 9-box matrix: {str(e)}")

# Succession Pipeline
st.subheader("Succession Pipeline")
try:
    succession_pipeline = get_succession_pipeline(data, personas)
    if not succession_pipeline.empty:
        st.dataframe(succession_pipeline)
    else:
        st.warning("No succession pipeline data available due to an error or insufficient data.")
except Exception as e:
    st.error(f"Error generating succession pipeline: {str(e)}")

# Interactive candidate matching
st.header("Validate Individual Candidate")
with st.form(key="candidate_form"):
    diplomatic = st.slider("Diplomatic", 0, 100, 85)
    balanced = st.slider("Balanced", 0, 100, 90)
    sociable = st.slider("Sociable", 0, 100, 75)
    innovative = st.slider("Innovative", 0, 100, 80)
    dept_weights = st.selectbox("Select Department Weights", list(default_weights.keys()), index=0)
    submit = st.form_submit_button("Match Candidate")
if submit:
    try:
        best_dept, distances = match_candidate(diplomatic, balanced, sociable, innovative, default_weights[dept_weights])
        st.write(f"Best Match: {best_dept}")
        dist_df = pd.DataFrame(list(distances.items()), columns=['Department', 'Distance'])
        st.dataframe(dist_df.round(2))
        fig = px.bar(dist_df, x='Department', y='Distance', title='Match Scores')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error matching candidate: {str(e)}")