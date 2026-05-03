import { useState, useEffect } from 'react'
import SearchableSelect from './SearchableSelect'

function GamePrediction() {
  const [teams, setTeams] = useState([])
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [formData, setFormData] = useState({
    home_fg_pct: 0.46,
    away_fg_pct: 0.44,
    home_rebounds: 45,
    away_rebounds: 42,
    home_assists: 25,
    away_assists: 22,
    model_choice: 'logreg'
  })
  
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  // Fetch teams on mount
  useEffect(() => {
    fetch('/teams')
      .then(res => res.json())
      .then(data => {
        if (data.success) setTeams(data.teams)
      })
      .catch(console.error)
  }, [])

  // Fetch team stats when home team changes
  useEffect(() => {
    if (homeTeam) {
      fetch(`/team_stats/${homeTeam}`)
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setFormData(prev => ({
              ...prev,
              home_fg_pct: data.stats.fg_pct,
              home_rebounds: data.stats.rebounds,
              home_assists: data.stats.assists
            }))
          }
        })
        .catch(console.error)
    }
  }, [homeTeam])

  // Fetch team stats when away team changes
  useEffect(() => {
    if (awayTeam) {
      fetch(`/team_stats/${awayTeam}`)
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setFormData(prev => ({
              ...prev,
              away_fg_pct: data.stats.fg_pct,
              away_rebounds: data.stats.rebounds,
              away_assists: data.stats.assists
            }))
          }
        })
        .catch(console.error)
    }
  }, [awayTeam])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      const formBody = new URLSearchParams(formData)
      const response = await fetch('/predict_game', {
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

  return (
    <div className="glass-panel prediction-card game-card">
      <div className="panel-header">
        <div className="panel-icon"><i className="fa-solid fa-trophy"></i></div>
        <div>
          <h2>Game Outcome</h2>
          <p>Predict match winner using team metrics</p>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="modern-form">
        <div className="form-grid team-selectors">
          <div className="form-group full-width">
            <label><i className="fa-solid fa-house"></i> Home Team</label>
            <SearchableSelect
              options={teams}
              value={homeTeam}
              onChange={setHomeTeam}
              placeholder="Search and select home team..."
              icon="fa-solid fa-house"
            />
          </div>
          <div className="form-group full-width">
            <label><i className="fa-solid fa-plane"></i> Away Team</label>
            <SearchableSelect
              options={teams}
              value={awayTeam}
              onChange={setAwayTeam}
              placeholder="Search and select away team..."
              icon="fa-solid fa-plane"
            />
          </div>
        </div>
        
        <div className="form-grid">
          <div className="form-group">
            <label>Home FG% <small>(0-1)</small></label>
            <input 
              type="number" 
              name="home_fg_pct" 
              value={formData.home_fg_pct}
              onChange={handleChange}
              min="0"
              max="1"
              step="0.01" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Away FG% <small>(0-1)</small></label>
            <input 
              type="number" 
              name="away_fg_pct" 
              value={formData.away_fg_pct}
              onChange={handleChange}
              min="0" 
              max="1" 
              step="0.01" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Home REB</label>
            <input 
              type="number" 
              name="home_rebounds" 
              value={formData.home_rebounds}
              onChange={handleChange}
              min="20" 
              max="80" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Away REB</label>
            <input 
              type="number" 
              name="away_rebounds" 
              value={formData.away_rebounds}
              onChange={handleChange}
              min="20" 
              max="80" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Home AST</label>
            <input 
              type="number" 
              name="home_assists" 
              value={formData.home_assists}
              onChange={handleChange}
              min="10" 
              max="50" 
              required 
            />
          </div>
          <div className="form-group">
            <label>Away AST</label>
            <input 
              type="number" 
              name="away_assists" 
              value={formData.away_assists}
              onChange={handleChange}
              min="10" 
              max="50" 
              required 
            />
          </div>
        </div>
        
        <div className="form-group model-selector">
          <label><i className="fa-solid fa-microchip"></i> Prediction Model</label>
          <select name="model_choice" value={formData.model_choice} onChange={handleChange}>
            <option value="logreg">Logistic Regression (Fast & Reliable)</option>
            <option value="svm">Support Vector Machine (Complex Boundaries)</option>
          </select>
        </div>
        
        <button type="submit" className="action-btn btn-orange" disabled={loading}>
          {loading ? (
            <><i className="fa-solid fa-circle-notch fa-spin"></i> Processing...</>
          ) : (
            <><span>Analyze Gameplay</span><i className="fa-solid fa-arrow-right"></i></>
          )}
        </button>
      </form>
      
      {result && (
        <div className="result-display">
          {result.success ? (
            <div className={`result-card ${result.prediction === 1 ? 'success-win' : 'danger-loss'}`}>
              {homeTeam && awayTeam && (
                <div className="matchup-display">
                  <span className="team-name home">{teams.find(t => t.id === homeTeam)?.abbr}</span>
                  <span className="vs">vs</span>
                  <span className="team-name away">{teams.find(t => t.id === awayTeam)?.abbr}</span>
                </div>
              )}
              <div className="result-headline">
                <span className="res-icon">
                  {result.prediction === 1 ? 
                    <i className="fa-solid fa-crown"></i> : 
                    <i className="fa-solid fa-xmark"></i>
                  }
                </span>
                <h3>{result.prediction === 1 
                  ? `${teams.find(t => t.id === homeTeam)?.name || 'Home'} Wins!`
                  : `${teams.find(t => t.id === awayTeam)?.name || 'Away'} Wins!`
                }</h3>
              </div>
              <div className="result-stats">
                <div className="stat-pill">
                  <span>Model </span> <strong>{result.model_used}</strong>
                </div>
                <div className="stat-pill">
                  <span>Confidence </span> <strong>{result.confidence}%</strong>
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

export default GamePrediction
