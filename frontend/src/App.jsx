import { useState } from 'react'
import Scene3D from './components/Scene3D'
import StatsBar from './components/StatsBar'
import SearchPanel from './components/SearchPanel'
import ResultsPanel from './components/ResultsPanel'

// SEC-13: Backend URL from environment only — no hardcoded deployment URLs in client code
const BACKENDS = [
  import.meta.env.VITE_API_BASE_URL,           // Set in Vercel/deployment dashboard
  'http://localhost:8000',                       // Local development only
].filter(Boolean)

// Resilient fetch: tries each backend in order until one succeeds
async function resilientFetch(path, options = {}) {
  let lastErr = null
  for (const base of BACKENDS) {
    try {
      const res = await fetch(`${base}${path}`, {
        ...options,
        signal: AbortSignal.timeout(15000) // 15s timeout per backend
      })
      // Cloudflare challenge pages return 403 with HTML, not JSON
      if (res.status === 403) {
        const ct = res.headers.get('content-type') || ''
        if (ct.includes('text/html')) {
          throw new Error('Cloudflare challenge – skipping this backend')
        }
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res
    } catch (err) {
      lastErr = err
      console.warn(`[resilientFetch] ${base}${path} failed:`, err.message)
    }
  }
  throw lastErr || new Error('All backends unreachable')
}

// Export for StatsBar reuse
export { resilientFetch, BACKENDS }

export default function App() {
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState(null)

  const handleSearch = async (searchParams) => {
    setIsSearching(true)
    try {
      const res = await resilientFetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(searchParams)
      })

      const data = await res.json()
      setResults(data)
    } catch (err) {
      console.error('Search error:', err)
      setResults({
        answer: `⚠️ Connection Error: Could not reach any backend server. All backends may be sleeping — please try again in ~60 seconds.`,
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
      {/* Background Glow */}
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
