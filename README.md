# 🏀 NBA Prediction Engine

A comprehensive machine learning project for predicting NBA game outcomes and player performance, with a rubric-ready workflow, Streamlit web app, and React frontend.

## 📋 Project Structure

```
project/
├── frontend/               # React + Vite web interface
│   ├── src/
│   │   ├── components/
│   │   │   ├── GamePrediction.jsx
│   │   │   ├── PlayerPrediction.jsx
│   │   │   └── SearchableSelect.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── README.md
├── models/                 # Trained ML models (pickle files)
├── outputs/                # EDA plots, metrics, comparisons
├── app.py                  # Flask REST API backend
├── streamlit_app.py        # Streamlit web application (RECOMMENDED)
├── ml_course_workflow.py   # End-to-end ML pipeline
├── nba_prediction.ipynb    # Jupyter notebook with analysis
├── game_data_with_target.csv    # Dataset
├── requirements.txt        # Python dependencies
├── QUICKSTART.md          # Quick start guide
├── MIGRATION.md           # Migration notes
└── README.md              # This file
```

## ✨ Features

### 🎯 Streamlit App (Recommended for Users)
- **Single command**: `streamlit run streamlit_app.py`
- **No setup required**: Load and predict instantly
- **5 interactive pages**:
  - 🎮 Game Outcome Prediction
  - 👤 Player Performance Prediction
  - 📈 Model Comparison Dashboard
  - 📊 EDA Visualizations & Analysis
  - ℹ️ Project Information
- **Model transparency**: View model metrics and feature importance
- **Real-time predictions**: Instant results with model selection

### 🖥️ Frontend (React + Vite)
- **Modern UI**: Clean, responsive design
- **Game Prediction**: Predict match winners using team metrics
- **Player Prediction**: Forecast individual player points
- **Multiple Models**: Switch between algorithms on the fly
- **Responsive**: Works on desktop and mobile devices

### 🔌 Backend (Flask API)
- **RESTful API**: Clean, documented endpoints
- **CORS Enabled**: Frontend-backend communication
- **Model Selection**: Choose from multiple algorithms
- **Fast predictions**: Pre-loaded, cached models

### 📊 ML Workflow (Course Project)
- **Problem**: Binary classification + Regression
- **Data**: Complete preprocessing, handling missing values
- **Baseline**: Logistic Regression, Linear Regression
- **Improvement**: Scaling, regularization, feature engineering
- **Unsupervised**: PCA, K-Means clustering
- **Advanced**: Decision Tree, Random Forest, Hyperparameter tuning
- **Output**: Model comparison, rubric-ready assessment

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App (Easiest)
```bash
streamlit run streamlit_app.py
```
Open browser to `http://localhost:8501`

### 3️⃣ Or Run Flask Backend
```bash
python app.py
```
API runs on `http://localhost:5000`

### 4️⃣ Or Run Frontend (Advanced)
```bash
cd frontend
npm install
npm run dev
```
App runs on `http://localhost:5173`

## 📚 Detailed Setup

### Backend Setup

1. **Install Python 3.8+**

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify models exist**:
   ```
   models/
   ├── game_logreg.pkl
   ├── game_svm.pkl
   ├── game_scaler.pkl
   ├── game_features.pkl
   ├── player_linreg.pkl
   ├── player_svr.pkl
   ├── player_scaler.pkl
   └── player_features.pkl
   ```

5. **Train models** (if needed):
   ```bash
   python ml_course_workflow.py
   ```

### Run Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

Features:
- Zero configuration needed
- Load dataset and models automatically
- Interactive predictions
- EDA visualizations
- Model comparison table

### Run Flask API

```bash
python app.py
```

API Endpoints:
- `GET /` - API info
- `POST /predict_game` - Game outcome prediction
- `POST /predict_player` - Player points prediction

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
npm run build  # Production
```

## 🤖 Models & Performance

### Classification Models (Game Prediction)
| Model | Accuracy | Best For |
|-------|----------|----------|
| Logistic Regression | ~64% | Fast, interpretable |
| SVM | ~66% | Non-linear patterns |
| Decision Tree | ~68% | Feature importance |
| Random Forest | ~72% | Robust, best overall |

### Regression Models (Player Prediction)
| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.82 | 3.2 |
| Ridge | 0.83 | 3.1 |
| SVR | 0.81 | 3.4 |

## 📊 Workflow Outputs

Running `python ml_course_workflow.py` generates:

```
outputs/
├── model_comparison.csv          # Metrics table
├── workflow_summary.json         # Pipeline summary
├── rubric_assessment.json        # Scoring rubric
├── score_distribution.png        # EDA plot
├── correlation_heatmap.png       # Feature correlation
├── feature_correlation.png       # Target correlation
├── learning_curves.png           # Overfitting analysis
├── cm_logreg.png                # Confusion matrix
├── decision_tree.png            # Tree visualization
├── random_forest.png            # RF importance
├── final_comparison.png         # Model comparison
├── pca_analysis.png             # Dimensionality reduction
└── kmeans_clusters.png          # Clustering visualization
```

## 🔍 API Endpoints

### GET /
Returns API information

### POST /predict_game
Predict game outcome

**Request (form-data)**:
- `home_score`, `away_score`
- `home_fg_pct`, `away_fg_pct`
- `home_rebounds`, `away_rebounds`
- `home_assists`, `away_assists`
- `model_choice` (logreg|svm)

**Response**:
```json
{
  "success": true,
  "prediction": 1,
  "result": "Home Win",
  "confidence": 87.5
}
```

### POST /predict_player
Predict player points

**Request (form-data)**:
- `minutes_played`, `field_goals_made`, `field_goals_attempted`
- `three_pointers_made`, `free_throws_made`
- `assists`, `rebounds`, `steals`, `blocks`, `turnovers`
- `model_choice` (linreg|svr)

**Response**:
```json
{
  "success": true,
  "predicted_points": 24.5,
  "model_used": "Linear Regression"
}
```

## 🛠️ Technologies

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Streamlit** - Data app framework
- **Matplotlib/Seaborn** - Visualization

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **CSS3** - Styling
- **FontAwesome** - Icons

## 📈 Project Highlights

✅ Complete ML pipeline from data to deployment
✅ Multiple models with comparison metrics
✅ Feature engineering & hyperparameter tuning
✅ Unsupervised learning (PCA, K-Means)
✅ Professional web interfaces (Streamlit & React)
✅ RESTful API for integration
✅ Production-ready code structure
✅ Comprehensive documentation

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing & EDA
- Baseline model development
- Regularization & scaling
- Dimensionality reduction
- Clustering analysis
- Ensemble methods
- Hyperparameter optimization
- Model evaluation & comparison
- Web app development
- API design

## 📝 Files Description

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main web interface (start here!) |
| `app.py` | Flask REST API backend |
| `ml_course_workflow.py` | Complete ML pipeline |
| `nba_prediction.ipynb` | Jupyter notebook |
| `requirements.txt` | Python dependencies |

## 🐛 Troubleshooting

**Models not found?**
```bash
python ml_course_workflow.py
```

**Port already in use?**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

## 📞 Support

For issues or questions:
1. Check `QUICKSTART.md` for quick answers
2. See `MIGRATION.md` for setup issues
3. Review notebook for detailed analysis

## 📄 License

MIT License - See LICENSE file

## 👤 Author

**Aakash Sarang**

## 🔗 Links

- **GitHub**: [oggigachad/oggigachad-nba-ml-prediction-engine](https://github.com/oggigachad/oggigachad-nba-ml-prediction-engine)
- **Data Source**: NBA stats
- **Course**: ML for AI Course Project

---

**Last Updated**: May 2026
**Status**: ✅ Production Ready

```bash
streamlit run streamlit_app.py
```

The Streamlit app is the recommended interface for a normal user. It loads the prepared dataset, checks the selected model, and provides a one-page prediction workflow with comparison and EDA panels.

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```
   The app will run on `http://localhost:5173` (or another port if 5173 is busy)

4. **Build for production**:
   ```bash
   npm run build
   ```

The React/Flask stack is kept for the original API-style workflow, but the Streamlit app is the simplest way to present the project to a non-technical reviewer.

## API Endpoints

### `GET /`
Returns API information and available endpoints.

### `POST /predict_game`
Predicts game outcome.

**Request Body** (form-data):
- `home_score`: Home team score
- `away_score`: Away team score
- `home_fg_pct`: Home field goal percentage (0-1)
- `away_fg_pct`: Away field goal percentage (0-1)
- `home_rebounds`: Home team rebounds
- `away_rebounds`: Away team rebounds
- `home_assists`: Home team assists
- `away_assists`: Away team assists
- `model_choice`: `logreg` or `svm`

**Response**:
```json
{
  "success": true,
  "prediction": 1,
  "result": "Home Win",
  "model_used": "Logistic Regression",
  "confidence": 87.5,
  "input_data": {...}
}
```

### `POST /predict_player`
Predicts player points.

**Request Body** (form-data):
- `minutes_played`: Minutes played
- `turnovers`: Turnovers
- `field_goals_made`: Field goals made
- `field_goals_attempted`: Field goals attempted
- `three_pointers_made`: Three-pointers made
- `free_throws_made`: Free throws made
- `assists`: Assists
- `rebounds`: Rebounds
- `steals`: Steals
- `blocks`: Blocks
- `model_choice`: `linreg` or `svr`

**Response**:
```json
{
  "success": true,
  "predicted_points": 24.5,
  "model_used": "Linear Regression",
  "input_data": {...}
}
```

## Development

### Frontend Development
- Hot reload enabled in development mode
- Proxy configured to forward API calls to Flask backend
- FontAwesome icons included
- Google Fonts: Outfit and Space Grotesk

### Backend Development
- Debug mode enabled by default
- CORS enabled for cross-origin requests
- Error handling for missing models
- Flexible feature mapping

## Technologies Used

### Frontend
- **React 18**: UI library
- **Vite**: Build tool and dev server
- **CSS3**: Modern styling with custom properties
- **FontAwesome**: Icon library
- **Google Fonts**: Typography

### Backend
- **Flask**: Web framework
- **Flask-CORS**: CORS support
- **scikit-learn**: Machine learning models
- **NumPy**: Numerical computing
- **Pandas**: Data handling and preprocessing
- **Matplotlib + Seaborn**: EDA and result visualization
- **joblib**: Model serialization

## License

MIT License

## Course Submission Checklist

- Source code repository (this project)
- Dataset included: `game_data_with_target.csv` (public-source derived)
- Run instructions included in `README.md` and `QUICKSTART.md`
- Rubric outputs: `outputs/workflow_summary.json`, `outputs/model_comparison.csv`, `outputs/rubric_assessment.json`
- Naming format to follow before final upload:
  - Report: `FirstName_RollNo_MLProject.pdf`
  - Code archive: `FirstName_RollNo_MLProject_Code.zip`

# nba-ml-prediction-engine
