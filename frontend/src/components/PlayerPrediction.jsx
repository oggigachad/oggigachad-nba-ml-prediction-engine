import { useState, useEffect } from 'react'
import SearchableSelect from './SearchableSelect'

function PlayerPrediction() {
  const [players, setPlayers] = useState([])
  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [formData, setFormData] = useState({
    minutes_played: 32,
    turnovers: 2,
    field_goals_made: 7,
    field_goals_attempted: 15,
    three_pointers_made: 2,
    free_throws_made: 4,
    assists: 5,
    rebounds: 6,
    steals: 1,
    blocks: 1,
    model_choice: 'linreg'
  })
  
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  // Fetch players on mount
  useEffect(() => {
    fetch('/players')
      .then(res => res.json())
      .then(data => {
        if (data.success) setPlayers(data.players)
      })
      .catch(console.error)
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const formBody = new URLSearchParams(formData)
      const response = await fetch('/predict_player', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formBody
      })
      
      const data = await response.json()
      setResult(data)
    } catch (error) {
      setResult({ success: false, error: error.message })
    } finally {
      setLoading(false)
    }
  }

  const getPerfClass = (pts) => {
    if (pts > 20) return 'star-perf'
    if (pts > 10) return 'solid-perf'
    return 'low-perf'
  }

  return (
    <div className="glass-panel prediction-card player-card">
      <div className="panel-header">
        <div className="panel-icon"><i className="fa-solid fa-user-astronaut"></i></div>
        <div>
          <h2>Player Performance</h2>
          <p>Forecast individual player points</p>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="modern-form">
        <div className="form-group full-width player-selector">
          <label><i className="fa-solid fa-user"></i> Select Player</label>
          <SearchableSelect
            options={players}
            value={selectedPlayer}
            onChange={setSelectedPlayer}
            placeholder="Search for a player..."
            icon="fa-solid fa-user"
          />
        </div>
        
        {selectedPlayer && (
          <div className="selected-player-info">
            <i className="fa-solid fa-basketball"></i>
            <span>Enter expected stats for: <strong>{players.find(p => p.id === selectedPlayer)?.name}</strong></span>
          </div>
        )}
        
        <div className="form-grid">
          <div className="form-group">
            <label><i className="fa-regular fa-clock"></i> Minutes</label>
            <input 
              type="number" 
              name="minutes_played" 
              value={formData.minutes_played}
              onChange={handleChange}
              min="1" 
              max="48" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Turnovers</label>
            <input 
              type="number" 
              name="turnovers" 
              value={formData.turnovers}
              onChange={handleChange}
              min="0" 
              max="10" 
              required
            />
          </div>
          <div className="form-group">
            <label>FGM (Made)</label>
            <input 
              type="number" 
              name="field_goals_made" 
              value={formData.field_goals_made}
              onChange={handleChange}
              min="0" 
              max="25" 
              required 
            />
          </div>
          <div className="form-group">
            <label>FGA (Attempted)</label>
            <input 
              type="number" 
              name="field_goals_attempted" 
              value={formData.field_goals_attempted}
              onChange={handleChange}
              min="0" 
              max="35" 
              required 
            />
          </div>
          <div className="form-group">
            <label>3PM (Threes)</label>
            <input 
              type="number" 
              name="three_pointers_made" 
              value={formData.three_pointers_made}
              onChange={handleChange}
              min="0" 
              max="15" 
              required 
            />
          </div>
          <div className="form-group">
            <label>FTM (Free Throws)</label>
            <input 
              type="number" 
              name="free_throws_made" 
              value={formData.free_throws_made}
              onChange={handleChange}
              min="0" 
              max="20" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Assists</label>
            <input 
              type="number" 
              name="assists" 
              value={formData.assists}
              onChange={handleChange}
              min="0" 
              max="20" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Rebounds</label>
            <input 
              type="number" 
              name="rebounds" 
              value={formData.rebounds}
              onChange={handleChange}
              min="0" 
              max="25" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Steals</label>
            <input 
              type="number" 
              name="steals" 
              value={formData.steals}
              onChange={handleChange}
              min="0" 
              max="10" 
              step="0.5" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Blocks</label>
            <input 
              type="number" 
              name="blocks" 
              value={formData.blocks}
              onChange={handleChange}
              min="0" 
              max="10" 
              step="0.5" 
              required 
            />
          </div>
        </div>
        
        <div className="form-group model-selector">
          <label><i className="fa-solid fa-network-wired"></i> Regression Model</label>
          <select name="model_choice" value={formData.model_choice} onChange={handleChange}>
            <option value="linreg">Linear Regression (Baseline)</option>
            <option value="svr">Support Vector Regression (Advanced)</option>
          </select>
        </div>
        
        <button type="submit" className="action-btn btn-indigo" disabled={loading}>
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> Calculating...</>
          ) : (
            <><span>Predict Points</span><i className="fa-solid fa-bolt"></i></>
          )}
        </button>
      </form>
      
      {result && (
        <div className="result-display">
          {result.success ? (
            <div className={`result-card points-result ${getPerfClass(result.predicted_points)}`}>
              {selectedPlayer && (
                <div className="player-name-display">
                  <i className="fa-solid fa-user-check"></i>
                  {players.find(p => p.id === selectedPlayer)?.name}
                </div>
              )}
              <div className="points-circle">
                <span className="number">{result.predicted_points}</span>
                <span className="label">PTS</span>
              </div>
              <div className="result-stats">
                <div className="stat-pill">
                  <span>Regressor</span> <strong>{result.model_used}</strong>
                </div>
                <div className="stat-pill">
                  <span>Minutes</span> <strong>{result.input_data.minutes_played}</strong>
                </div>
              </div>
            </div>
          ) : (
            <div className="result-card error">
              <i className="fa-solid fa-triangle-exclamation"></i> {result.error}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PlayerPrediction
