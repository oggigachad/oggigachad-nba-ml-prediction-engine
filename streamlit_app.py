import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="NBA Prediction Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        margin-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "game_data_with_target.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_models():
    models = {}
    try:
        with open(MODELS_DIR / "game_logreg.pkl", "rb") as f:
            models["game_logreg"] = pickle.load(f)
        with open(MODELS_DIR / "game_svm.pkl", "rb") as f:
            models["game_svm"] = pickle.load(f)
        with open(MODELS_DIR / "game_scaler.pkl", "rb") as f:
            models["game_scaler"] = pickle.load(f)
        with open(MODELS_DIR / "game_features.pkl", "rb") as f:
            models["game_features"] = pickle.load(f)
        
        with open(MODELS_DIR / "player_linreg.pkl", "rb") as f:
            models["player_linreg"] = pickle.load(f)
        with open(MODELS_DIR / "player_svr.pkl", "rb") as f:
            models["player_svr"] = pickle.load(f)
        with open(MODELS_DIR / "player_scaler.pkl", "rb") as f:
            models["player_scaler"] = pickle.load(f)
        with open(MODELS_DIR / "player_features.pkl", "rb") as f:
            models["player_features"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    return models

@st.cache_data
def load_plots():
    plots_dir = OUTPUTS_DIR
    plots = {}
    
    plot_files = {
        "Score Distribution": "score_distribution.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Feature Correlation": "feature_correlation.png",
        "Learning Curves": "learning_curves.png",
        "Confusion Matrix (Logistic Regression)": "cm_logreg.png",
        "Decision Tree": "decision_tree.png",
        "Random Forest": "random_forest.png",
        "Final Comparison": "final_comparison.png",
        "PCA Analysis": "pca_analysis.png",
        "K-Means Clustering": "kmeans_clusters.png",
    }
    
    for name, filename in plot_files.items():
        path = plots_dir / filename
        if path.exists():
            plots[name] = path
    
    return plots

# Header
st.markdown("# 🏀 NBA Prediction Engine")
st.markdown(
    "Predict NBA game outcomes and player performance using machine learning"
)

# Load data and models
data = load_data()
models = load_models()

if models is None:
    st.error("⚠️ Could not load models. Please ensure all model files are in the `models/` directory.")
    st.stop()

plots = load_plots()

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Select a page:",
        [
            "🎮 Game Prediction",
            "👤 Player Prediction",
            "📈 Model Comparison",
            "📊 EDA & Analysis",
            "ℹ️ About"
        ]
    )

# ================= PAGE 1: GAME PREDICTION =================
if page == "🎮 Game Prediction":
    st.markdown("## Predict NBA Game Outcomes")
    st.markdown("Use team statistics to predict whether the home team will win.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Home Team Stats")
        home_fgm = st.number_input("Field Goals Made (FGM)", 0, 60, 30)
        home_fga = st.number_input("Field Goals Attempted (FGA)", 0, 100, 60)
        home_fg_pct = st.number_input("Field Goal %", 0.0, 1.0, 0.5, step=0.01)
        home_fg3m = st.number_input("3-Pointers Made", 0, 20, 10)
        home_reb = st.number_input("Rebounds", 0, 60, 30)
        home_ast = st.number_input("Assists", 0, 40, 20)
        home_tov = st.number_input("Turnovers", 0, 30, 15)
    
    with col2:
        st.markdown("### ✈️ Away Team Stats")
        away_fgm = st.number_input("Field Goals Made (FGM) ", 0, 60, 30)
        away_fga = st.number_input("Field Goals Attempted (FGA) ", 0, 100, 60)
        away_fg_pct = st.number_input("Field Goal % ", 0.0, 1.0, 0.5, step=0.01)
        away_fg3m = st.number_input("3-Pointers Made ", 0, 20, 10)
        away_reb = st.number_input("Rebounds ", 0, 60, 30)
        away_ast = st.number_input("Assists ", 0, 40, 20)
        away_tov = st.number_input("Turnovers ", 0, 30, 15)
    
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "SVM"]
    )
    
    if st.button("🔮 Predict", key="predict_game"):
        # Prepare input
        input_data = {
            'fgm_home': home_fgm,
            'fga_home': home_fga,
            'fg_pct_home': home_fg_pct,
            'fg3m_home': home_fg3m,
            'reb_home': home_reb,
            'ast_home': home_ast,
            'tov_home': home_tov,
            'fgm_away': away_fgm,
            'fga_away': away_fga,
            'fg_pct_away': away_fg_pct,
            'fg3m_away': away_fg3m,
            'reb_away': away_reb,
            'ast_away': away_ast,
            'tov_away': away_tov,
        }
        
        # Create DataFrame for scaling
        input_df = pd.DataFrame([input_data])
        
        # Get feature list
        features = models["game_features"]
        
        # Ensure all features are present
        for feat in features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        # Scale
        X_scaled = models["game_scaler"].transform(input_df[features])
        
        # Predict
        if model_choice == "Logistic Regression":
            model = models["game_logreg"]
            pred_proba = model.predict_proba(X_scaled)[0]
            pred = model.predict(X_scaled)[0]
        else:
            model = models["game_svm"]
            pred = model.predict(X_scaled)[0]
            pred_proba = [0, 0]  # SVM doesn't have predict_proba by default
        
        # Display results
        st.markdown("### 🎯 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred == 1:
                st.success("🏠 Home Team Wins")
            else:
                st.error("✈️ Away Team Wins")
        
        with col2:
            st.metric("Prediction", "Home Win" if pred == 1 else "Away Win")
        
        with col3:
            if len(pred_proba) > 1:
                confidence = max(pred_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")

# ================= PAGE 2: PLAYER PREDICTION =================
elif page == "👤 Player Prediction":
    st.markdown("## Predict Player Performance")
    st.markdown("Estimate how many points a player will score based on their stats.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        minutes_played = st.number_input("Minutes Played", 0, 48, 30)
        fgm = st.number_input("Field Goals Made", 0, 20, 5)
        fga = st.number_input("Field Goals Attempted", 0, 30, 12)
        fg3m = st.number_input("3-Pointers Made", 0, 10, 2)
        ftm = st.number_input("Free Throws Made", 0, 15, 2)
    
    with col2:
        assists = st.number_input("Assists", 0, 15, 3)
        rebounds = st.number_input("Rebounds", 0, 20, 5)
        steals = st.number_input("Steals", 0, 5, 1)
        blocks = st.number_input("Blocks", 0, 5, 1)
        turnovers = st.number_input("Turnovers", 0, 8, 1)
    
    model_choice_player = st.selectbox(
        "Select Model",
        ["Linear Regression", "SVR"],
        key="player_model"
    )
    
    if st.button("🔮 Predict Points", key="predict_player"):
        input_data_player = {
            'minutes_played': minutes_played,
            'field_goals_made': fgm,
            'field_goals_attempted': fga,
            'three_pointers_made': fg3m,
            'free_throws_made': ftm,
            'assists': assists,
            'rebounds': rebounds,
            'steals': steals,
            'blocks': blocks,
            'turnovers': turnovers,
        }
        
        input_df_player = pd.DataFrame([input_data_player])
        features_player = models["player_features"]
        
        X_scaled_player = models["player_scaler"].transform(input_df_player[features_player])
        
        if model_choice_player == "Linear Regression":
            model_p = models["player_linreg"]
        else:
            model_p = models["player_svr"]
        
        pred_points = model_p.predict(X_scaled_player)[0]
        
        st.markdown("### 🎯 Predicted Points")
        st.metric("Estimated Points", f"{max(0, pred_points):.2f}")

# ================= PAGE 3: MODEL COMPARISON =================
elif page == "📈 Model Comparison":
    st.markdown("## Model Performance Comparison")
    
    comparison_file = OUTPUTS_DIR / "model_comparison.csv"
    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Model comparison file not found. Run the ML workflow to generate it.")

# ================= PAGE 4: EDA & ANALYSIS =================
elif page == "📊 EDA & Analysis":
    st.markdown("## Exploratory Data Analysis")
    
    if not plots:
        st.info("No plots found. The ML workflow hasn't been run yet.")
    else:
        plot_list = list(plots.keys())
        selected_plot = st.selectbox("Select Plot", plot_list)
        
        if selected_plot in plots:
            st.image(str(plots[selected_plot]), use_column_width=True)
    
    st.markdown("---")
    st.markdown("## Dataset Overview")
    st.dataframe(data.head(10), use_container_width=True)
    st.markdown(f"**Dataset shape:** {data.shape[0]} rows × {data.shape[1]} columns")

# ================= PAGE 5: ABOUT =================
elif page == "ℹ️ About":
    st.markdown("## About NBA Prediction Engine")
    
    st.markdown("""
    ### 🎯 Project Overview
    This is a comprehensive machine learning project for predicting:
    - **Game Outcomes**: Will the home team win?
    - **Player Performance**: How many points will a player score?
    
    ### 🧠 Models Used
    
    **Classification (Game Prediction):**
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Decision Tree
    - Random Forest
    
    **Regression (Player Prediction):**
    - Linear Regression
    - Support Vector Regression (SVR)
    - Ridge Regression
    - Lasso Regression
    
    ### 📊 Features
    - **Game Statistics**: Field goals, 3-pointers, rebounds, assists, turnovers
    - **Player Statistics**: Minutes played, shooting stats, assists, rebounds, steals, blocks
    
    ### 🔧 Technologies
    - **Backend**: Python, scikit-learn, Flask
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    
    ### 📈 Model Improvements
    - Feature scaling using StandardScaler
    - Regularization (Ridge & Lasso)
    - Dimensionality reduction (PCA)
    - Unsupervised learning (K-Means clustering)
    - Hyperparameter tuning (GridSearchCV)
    
    ### 📁 Project Structure
    ```
    project/
    ├── app.py                    # Flask API backend
    ├── streamlit_app.py          # Streamlit web interface
    ├── ml_course_workflow.py     # End-to-end ML workflow
    ├── models/                   # Trained ML models
    ├── outputs/                  # EDA plots & results
    ├── frontend/                 # React + Vite frontend
    ├── requirements.txt          # Python dependencies
    └── README.md                 # Project documentation
    ```
    
    ### 🚀 Getting Started
    ```bash
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
    """)
    
    st.markdown("---")
    st.markdown("""
    **Author**: Aakash Sarang  
    **License**: MIT  
    **GitHub**: [oggigachad/oggigachad-nba-ml-prediction-engine](https://github.com/oggigachad/oggigachad-nba-ml-prediction-engine)
    """)
