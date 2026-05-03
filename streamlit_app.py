import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Prediction Engine",
    page_icon="basketball",
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
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "game_logreg": "game_logreg.pkl",
        "game_svm": "game_svm.pkl",
        "game_scaler": "game_scaler.pkl",
        "game_features": "game_features.pkl",
        "player_linreg": "player_linreg.pkl",
        "player_svr": "player_svr.pkl",
        "player_scaler": "player_scaler.pkl",
        "player_features": "player_features.pkl",
    }
    
    try:
        for model_name, filename in model_files.items():
            filepath = MODELS_DIR / filename
            if filepath.exists():
                with open(filepath, "rb") as f:
                    models[model_name] = pickle.load(f)
            else:
                st.warning(f"Model file not found: {filename}")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    
    return models if models else None

@st.cache_data
def load_plots():
    plots_dir = OUTPUTS_DIR
    plots = {}
    
    plot_files = {
        "Score Distribution": "score_distribution.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Feature Correlation": "feature_correlation.png",
        "Confusion Matrix": "cm_logreg.png",
        "PCA Analysis": "pca_analysis.png",
        "K-Means Clustering": "kmeans_clusters.png",
    }
    
    for name, filename in plot_files.items():
        path = plots_dir / filename
        if path.exists():
            plots[name] = path
    
    return plots

# Header
st.markdown("# NBA Prediction Engine")
st.markdown("Predict NBA game outcomes and player performance using machine learning models")

# Load data and models
data = load_data()
models = load_models()

if models is None or data is None:
    st.error("ERROR: Could not load all required models and data. Please ensure all model files are in the models/ directory and game data is available.")
    st.stop()

plots = load_plots()

# Get unique team names
if data is not None and 'team_name_home' in data.columns:
    home_teams = sorted(data['team_name_home'].unique())
else:
    home_teams = []

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select a page:",
        [
            "Game Prediction",
            "Player Prediction",
            "Model Comparison",
            "Data Analysis",
            "About"
        ]
    )

# ================= PAGE 1: GAME PREDICTION =================
if page == "Game Prediction":
    st.markdown("## Predict NBA Game Outcomes")
    st.markdown("Compare two NBA teams and predict the game winner using team statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Home Team")
        home_team_name = st.selectbox("Select Home Team", home_teams, key="home_team")
        home_fgm = st.number_input("Field Goals Made", 0, 60, 30, key="h_fgm")
        home_fga = st.number_input("Field Goals Attempted", 0, 100, 60, key="h_fga")
        home_fg_pct = st.number_input("Field Goal Percentage", 0.0, 1.0, 0.5, step=0.01, key="h_fg_pct")
        home_fg3m = st.number_input("3-Pointers Made", 0, 20, 10, key="h_fg3m")
        home_reb = st.number_input("Rebounds", 0, 60, 30, key="h_reb")
        home_ast = st.number_input("Assists", 0, 40, 20, key="h_ast")
        home_tov = st.number_input("Turnovers", 0, 30, 15, key="h_tov")
    
    with col2:
        st.markdown("### Away Team")
        away_team_name = st.selectbox("Select Away Team", home_teams, key="away_team")
        away_fgm = st.number_input("Field Goals Made", 0, 60, 30, key="a_fgm")
        away_fga = st.number_input("Field Goals Attempted", 0, 100, 60, key="a_fga")
        away_fg_pct = st.number_input("Field Goal Percentage", 0.0, 1.0, 0.5, step=0.01, key="a_fg_pct")
        away_fg3m = st.number_input("3-Pointers Made", 0, 20, 10, key="a_fg3m")
        away_reb = st.number_input("Rebounds", 0, 60, 30, key="a_reb")
        away_ast = st.number_input("Assists", 0, 40, 20, key="a_ast")
        away_tov = st.number_input("Turnovers", 0, 30, 15, key="a_tov")
    
    model_choice = st.selectbox(
        "Select Model for Prediction",
        ["Logistic Regression", "SVM"]
    )
    
    if st.button("Predict Winner", key="predict_game"):
        try:
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
                pred_proba = [0.5, 0.5]
            
            # Display results
            st.markdown("### Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred == 1:
                    st.success(f"Winner: {home_team_name}")
                else:
                    st.error(f"Winner: {away_team_name}")
            
            with col2:
                winner = f"{home_team_name}" if pred == 1 else f"{away_team_name}"
                st.metric("Predicted Winner", winner)
            
            with col3:
                if len(pred_proba) > 1:
                    confidence = max(pred_proba) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            # Additional stats
            st.markdown("### Team Comparison")
            comparison_data = {
                "Metric": ["Field Goals Made", "Field Goals Attempted", "FG %", "3-Pointers", "Rebounds", "Assists", "Turnovers"],
                home_team_name: [home_fgm, home_fga, f"{home_fg_pct:.1%}", home_fg3m, home_reb, home_ast, home_tov],
                away_team_name: [away_fgm, away_fga, f"{away_fg_pct:.1%}", away_fg3m, away_reb, away_ast, away_tov]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ================= PAGE 2: PLAYER PREDICTION =================
elif page == "Player Prediction":
    st.markdown("## Predict Player Points")
    st.markdown("Estimate how many points a player will score based on performance statistics")
    
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
        "Select Model for Prediction",
        ["Linear Regression", "SVR"],
        key="player_model"
    )
    
    if st.button("Predict Points", key="predict_player"):
        try:
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
            
            st.markdown("### Predicted Points")
            st.metric("Estimated Points", f"{max(0, pred_points):.2f}")
            
            # Player stats summary
            st.markdown("### Player Statistics")
            stats_data = {
                "Statistic": ["Minutes Played", "Field Goals Made", "Field Goals Attempted", "3-Pointers Made", "Free Throws Made", "Assists", "Rebounds", "Steals", "Blocks", "Turnovers"],
                "Value": [minutes_played, fgm, fga, fg3m, ftm, assists, rebounds, steals, blocks, turnovers]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ================= PAGE 3: MODEL COMPARISON =================
elif page == "Model Comparison":
    st.markdown("## Model Performance Comparison")
    
    comparison_file = OUTPUTS_DIR / "model_comparison.csv"
    if comparison_file.exists():
        try:
            comparison_df = pd.read_csv(comparison_file)
            st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown("### Model Description")
            st.markdown("""
            **Classification Models (Game Prediction):**
            - Logistic Regression: Linear classification model
            - SVM (Support Vector Machine): Kernel-based classification model
            
            **Regression Models (Player Points Prediction):**
            - Linear Regression: Standard linear regression model
            - SVR (Support Vector Regression): Kernel-based regression model
            """)
        except Exception as e:
            st.error(f"Error loading model comparison: {e}")
    else:
        st.info("Model comparison file not found. Run the ML workflow to generate it.")

# ================= PAGE 4: DATA ANALYSIS =================
elif page == "Data Analysis":
    st.markdown("## Exploratory Data Analysis")
    
    if not plots:
        st.info("No plots found. The ML workflow has not been run yet.")
    else:
        plot_list = list(plots.keys())
        selected_plot = st.selectbox("Select Plot", plot_list)
        
        if selected_plot in plots:
            try:
                st.image(str(plots[selected_plot]), use_column_width=True)
            except Exception as e:
                st.error(f"Error loading plot: {e}")
    
    st.markdown("---")
    st.markdown("## Dataset Overview")
    if data is not None:
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown(f"**Dataset size:** {data.shape[0]} games x {data.shape[1]} columns")
    else:
        st.error("Could not load dataset")

# ================= PAGE 5: ABOUT =================
elif page == "About":
    st.markdown("## About NBA Prediction Engine")
    
    st.markdown("""
    ### Project Overview
    This is a machine learning project for predicting:
    - **Game Outcomes**: Predict whether the home team will win
    - **Player Performance**: Estimate how many points a player will score
    
    ### Available Models
    
    **Classification (Game Prediction):**
    - Logistic Regression
    - Support Vector Machine (SVM)
    
    **Regression (Player Points):**
    - Linear Regression
    - Support Vector Regression (SVR)
    
    ### Input Features
    
    **Game Statistics:**
    - Field Goals Made (FGM)
    - Field Goals Attempted (FGA)
    - Field Goal Percentage (FG%)
    - 3-Pointers Made
    - Rebounds
    - Assists
    - Turnovers
    
    **Player Statistics:**
    - Minutes Played
    - Field Goals Made
    - Field Goals Attempted
    - 3-Pointers Made
    - Free Throws Made
    - Assists
    - Rebounds
    - Steals
    - Blocks
    - Turnovers
    
    ### Technologies Used
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: scikit-learn
    - **Web Framework**: Streamlit
    - **Backend API**: Flask
    - **Visualization**: Matplotlib, Seaborn
    
    ### Project Features
    - Multiple model options for predictions
    - Real-time predictions with confidence scores
    - Team name comparison
    - Player statistics summary
    - Model performance metrics
    - Exploratory data analysis plots
    
    ### Project Structure
    ```
    project/
    ├── streamlit_app.py          # Main web application
    ├── app.py                    # Flask API backend
    ├── ml_course_workflow.py     # ML training pipeline
    ├── models/                   # Trained models
    ├── outputs/                  # EDA plots
    ├── requirements.txt          # Dependencies
    └── README.md                 # Documentation
    ```
    
    ### Getting Started
    ```bash
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
    """)
    
    st.markdown("---")
    st.markdown("""
    **Author**: Aakash Sarang
    **Repository**: [GitHub](https://github.com/oggigachad/oggigachad-nba-ml-prediction-engine)
    """)
