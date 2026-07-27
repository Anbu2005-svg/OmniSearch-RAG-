import { useState } from 'react'
import Scene3D from './components/Scene3D'
import StatsBar from './components/StatsBar'
import SearchPanel from './components/SearchPanel'
import ResultsPanel from './components/ResultsPanel'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState(null)

  const handleSearch = async (searchParams) => {
    setIsSearching(true)
    try {
      const res = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(searchParams)
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`)
      }

      const data = await res.json()
      setResults(data)
    } catch (err) {
      console.error('Search error:', err)
      setResults({
        answer: `⚠️ Connection Error: Could not reach backend server at ${API_BASE}. Make sure server.py is running.`,
        sources: [],
        query: searchParams.query,
        latency: 0
      })
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="app-container">
      {/* 3D Background */}
      <Scene3D searching={isSearching} />

      {/* UI Overlay */}
      <div className="ui-overlay">
        <StatsBar />

        <main className="main-content">
          <SearchPanel onSearch={handleSearch} isSearching={isSearching} />
          <ResultsPanel results={results} isSearching={isSearching} />
        </main>
      </div>
    </div>
  )
}
