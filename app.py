from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pickle
import numpy as np
import os
import sqlite3

app = Flask(__name__)
CORS(app)

DB_PATH = 'nba.sqlite'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Load models and scalers
model_dir = 'models'

try:
    # Game Models
    game_logreg = joblib.load(os.path.join(model_dir, 'game_logreg.pkl'))
    game_svm = joblib.load(os.path.join(model_dir, 'game_svm.pkl'))
    game_scaler = joblib.load(os.path.join(model_dir, 'game_scaler.pkl'))
    with open(os.path.join(model_dir, 'game_features.pkl'), 'rb') as f:
        game_features = pickle.load(f)

    # Player Models
    player_linreg = joblib.load(os.path.join(model_dir, 'player_linreg.pkl'))
    player_svr = joblib.load(os.path.join(model_dir, 'player_svr.pkl'))
    player_scaler = joblib.load(os.path.join(model_dir, 'player_scaler.pkl'))
    with open(os.path.join(model_dir, 'player_features.pkl'), 'rb') as f:
        player_features = pickle.load(f)
except Exception as e:
    print(f"Error loading models. Ensure you have trained logic/models: {e}")

@app.route('/')
def home():
    return jsonify({
        'message': 'NBA Prediction API',
        'endpoints': {
            'predict_game': '/predict_game (POST)',
            'predict_player': '/predict_player (POST)',
            'players': '/players (GET)',
            'teams': '/teams (GET)',
            'team_stats': '/team_stats/<team_id> (GET)'
        }
    })

@app.route('/players', methods=['GET'])
def get_players():
    try:
        conn = get_db_connection()
        players = conn.execute(
            "SELECT id, full_name FROM player WHERE is_active = 1 ORDER BY full_name"
        ).fetchall()
        conn.close()
        return jsonify({
            'success': True,
            'players': [{'id': p['id'], 'name': p['full_name']} for p in players]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        conn = get_db_connection()
        teams = conn.execute(
            "SELECT id, full_name, abbreviation FROM team ORDER BY full_name"
        ).fetchall()
        conn.close()
        return jsonify({
            'success': True,
            'teams': [{'id': t['id'], 'name': t['full_name'], 'abbr': t['abbreviation']} for t in teams]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/team_stats/<team_id>', methods=['GET'])
def get_team_stats(team_id):
    try:
        conn = get_db_connection()
        # Get average stats for a team from recent games (as home team)
        stats = conn.execute("""
            SELECT 
                AVG(fg_pct_home) as fg_pct,
                AVG(reb_home) as rebounds,
                AVG(ast_home) as assists,
                AVG(pts_home) as points,
                team_name_home as team_name
            FROM game 
            WHERE team_id_home = ? 
            AND fg_pct_home IS NOT NULL
            GROUP BY team_id_home
        """, (team_id,)).fetchone()
        
        if not stats:
            # Try as away team
            stats = conn.execute("""
                SELECT 
                    AVG(fg_pct_away) as fg_pct,
                    AVG(reb_away) as rebounds,
                    AVG(ast_away) as assists,
                    AVG(pts_away) as points,
                    team_name_away as team_name
                FROM game 
                WHERE team_id_away = ? 
                AND fg_pct_away IS NOT NULL
                GROUP BY team_id_away
            """, (team_id,)).fetchone()
        
        conn.close()
        
        if stats:
            return jsonify({
                'success': True,
                'stats': {
                    'team_name': stats['team_name'],
                    'fg_pct': round(stats['fg_pct'] or 0.45, 3),
                    'rebounds': round(stats['rebounds'] or 42, 1),
                    'assists': round(stats['assists'] or 23, 1),
                    'points': round(stats['points'] or 105, 1)
                }
            })
        else:
            return jsonify({
                'success': True,
                'stats': {'fg_pct': 0.45, 'rebounds': 42, 'assists': 23, 'points': 105}
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_game', methods=['POST'])
def predict_game():
    try:
        data = request.form.to_dict()
        model_choice = data.get('model_choice', 'logreg')
        
        feature_dict = {f: 0.0 for f in game_features} # Initialize with 0s
        
        # Map known inputs to features where possible
        feature_dict['fg_pct_home'] = float(data.get('home_fg_pct', 0))
        feature_dict['fg_pct_away'] = float(data.get('away_fg_pct', 0))
        feature_dict['reb_home'] = float(data.get('home_rebounds', 0))
        feature_dict['reb_away'] = float(data.get('away_rebounds', 0))
        feature_dict['ast_home'] = float(data.get('home_assists', 0))
        feature_dict['ast_away'] = float(data.get('away_assists', 0))
        
        features_array = np.array([feature_dict[f] for f in game_features]).reshape(1, -1)
        
        if game_scaler:
            features_array = game_scaler.transform(features_array)
            
        model = game_svm if model_choice == 'svm' else game_logreg
        prediction = int(model.predict(features_array)[0])
        
        confidence = "N/A"
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_array)[0]
            confidence = round(max(probs) * 100, 2)
            
        return jsonify({
            'success': True,
            'prediction': prediction,
            'result': 'Home Win' if prediction == 1 else 'Home Loss',
            'model_used': 'Logistic Regression' if model_choice == 'logreg' else 'SVM Classifier',
            'confidence': confidence,
            'input_data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_player', methods=['POST'])
def predict_player():
    try:
        data = request.form.to_dict()
        model_choice = data.get('model_choice', 'linreg')
        
        feature_dict = {f: 0.0 for f in player_features}
        for f in player_features:
            if f in data:
                feature_dict[f] = float(data[f])
                
        features_array = np.array([feature_dict[f] for f in player_features]).reshape(1, -1)
        
        if player_scaler:
            features_array = player_scaler.transform(features_array)
            
        model = player_svr if model_choice == 'svr' else player_linreg
        predicted_points = float(model.predict(features_array)[0])
        predicted_points = max(0, round(predicted_points, 1))
        
        return jsonify({
            'success': True,
            'predicted_points': predicted_points,
            'model_used': 'Linear Regression' if model_choice == 'linreg' else 'Support Vector Regression',
            'input_data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
