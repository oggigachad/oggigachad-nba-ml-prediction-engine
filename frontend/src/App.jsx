import { useState } from 'react'
import './App.css'
import GamePrediction from './components/GamePrediction'
import PlayerPrediction from './components/PlayerPrediction'

function App() {
  return (
    <>
      <div className="bg-orb bg-orb-1"></div>
      <div className="bg-orb bg-orb-2"></div>
      
      <div className="container">
        <header className="app-header">
          <div className="logo-wrapper">
            <i className="fa-solid fa-basketball fa-spin-hover"></i>
            <h1>NBA <span>Prediction Pro</span></h1>
          </div>
          <p>Advanced Machine Learning Models for Game Outcomes & Player Analytics</p>
        </header>

        <div className="dashboard-grid">
          <GamePrediction />
          <PlayerPrediction />
        </div>

        <footer>
          <p><i className="fa-solid fa-code"></i> Developed for NBA Machine Learning Analytics</p>
        </footer>
      </div>
    </>
  )
}

export default App
