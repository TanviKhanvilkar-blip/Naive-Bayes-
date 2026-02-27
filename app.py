import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.utils.multiclass import type_of_target

# --- Page Setup ---
st.set_page_config(page_title="Smart ML Studio", layout="wide", page_icon="🧠")

# --- Initialize Models Dictionary ---
AVAILABLE_MODELS = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# ==========================================
# SIDEBAR: Configuration & Inputs
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    
    # 1. Input Dataset
    uploaded_file = st.file_uploader("📁 Upload Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        all_columns = df.columns.tolist()
        
        st.divider()
        st.subheader("Data Settings")
        
        # 2. Target
        target_col = st.selectbox("🎯 Target Variable", options=all_columns)
        
        # 3. Features (Filter for numeric only)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        available_features = [col for col in numeric_columns if col != target_col]
        
        feature_cols = st.multiselect(
            "📈 Feature Variables", 
            options=available_features, 
            default=available_features
        )
        
        # 4. Train Test Split
        test_size_pct = st.slider("✂️ Train/Test Split (% Test)", 10, 50, 20, 5)
        test_size_decimal = test_size_pct / 100.0
        
        st.divider()
        st.subheader("Model Settings")
        
        # 5. Model Selection Dropdown
        selected_model_name = st.selectbox("🤖 Choose Classification Model", options=list(AVAILABLE_MODELS.keys()))
        
        # 6. Evaluate Button
        run_evaluation = st.button("🚀 Train & Evaluate Model", type="primary", use_container_width=True)

# ==========================================
# MAIN PAGE: Title & Tabs
# ==========================================
st.title("🧠 Smart ML Studio")

if uploaded_file is None:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
else:
    # Create the Tabs
    tab_eda, tab_model = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🎯 Model Evaluation"])
    
    # ------------------------------------------
    # TAB 1: Exploratory Data Analysis (EDA)
    # ------------------------------------------
    with tab_eda:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary Statistics")
            # Show basic math stats for numerical columns
            st.dataframe(df.describe(), use_container_width=True)
            
        with col2:
            st.subheader(f"Target Distribution: '{target_col}'")
            # Show a bar chart counting how many of each category exist
            val_counts = df[target_col].value_counts()
            st.bar_chart(val_counts)
            
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("Missing Count"), use_container_width=True)

    # ------------------------------------------
    # TAB 2: Model Evaluation
    # ------------------------------------------
    with tab_model:
        if run_evaluation:
            if not feature_cols:
                st.error("Please select at least one feature column in the sidebar.")
            else:
                X = df[feature_cols]
                y = df[target_col].copy() 
                
                # Auto-Binning for Continuous Targets (Keeps classifiers from crashing)
                if type_of_target(y) == 'continuous':
                    st.info(f"💡 **Note:** '{target_col}' contains continuous numbers. To use classification models, the app automatically grouped these numbers into 'Low', 'Medium', and 'High' categories.")
                    try:
                        y = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])
                    except ValueError:
                        y = pd.cut(y, bins=3, labels=["Low", "Medium", "High"])

                try:
                    # Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_decimal, random_state=42
                    )
                    
                    # Fetch and Train the chosen model
                    model = AVAILABLE_MODELS[selected_model_name]
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    predictions = model.predict(X_test)
                    
                    # --- Results Display ---
                    st.markdown(f"### Results for **{selected_model_name}**")
                    
                    # 1. Top Level Metrics
                    accuracy = accuracy_score(y_test, predictions)
                    st.metric(label="Overall Accuracy", value=f"{accuracy * 100:.2f}%")
                    
                    st.divider()
                    
                    # Create two columns for detailed results
                    res_col1, res_col2 = st.columns(2)
                    
                    # 2. Detailed Classification Report
                    with res_col1:
                        st.subheader("Classification Report")
                        st.write("Shows precision, recall, and f1-score for each specific category.")
                        # Convert scikit-learn report to a beautiful pandas dataframe
                        report_dict = classification_report(y_test, predictions, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                        
                    # 3. Confusion Matrix
                    with res_col2:
                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        cm = confusion_matrix(y_test, predictions)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                        disp.plot(ax=ax, cmap="Blues", colorbar=False)
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"An unexpected error occurred during training: {e}")
        else:
            st.write("Configure your settings in the sidebar and click **🚀 Train & Evaluate Model** to see results here.")