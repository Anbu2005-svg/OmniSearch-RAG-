import { useState } from 'react'
import { motion } from 'framer-motion'

const SAMPLE_QUERIES = [
  { icon: '⛽', text: 'Quality of petrol and diesel fuels directive' },
  { icon: '🔬', text: 'Ultra-fine particle emissions from GDI engines' },
  { icon: '🚨', text: 'Emergency fuel availability and quality exemptions' },
  { icon: '🏛️', text: 'European Parliament legislative procedure' },
  { icon: '🌿', text: 'Environmental specifications for fuels' },
]

export default function SearchPanel({ onSearch, isSearching }) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0)

  const handleSearch = () => {
    if (!query.trim()) return
    onSearch({
      query: query.trim(),
      top_k: topK,
      score_threshold: threshold,
    })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch()
  }

  const handleSampleClick = (text) => {
    setQuery(text)
    onSearch({
      query: text,
      top_k: topK,
      score_threshold: threshold,
    })
  }

  return (
    <motion.div
      className="search-panel glass-panel"
      initial={{ x: -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div>
        <h2>🔍 Vector Search</h2>
        <p className="subtitle">Query 200,000 document vectors with neural retrieval</p>
      </div>

      {/* Search Input */}
      <div className="search-input-wrapper">
        <span className="search-icon">⚡</span>
        <input
          type="text"
          className="search-input"
          placeholder="Ask a question about your documents..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSearching}
        />
      </div>

      <button
        className="search-btn"
        onClick={handleSearch}
        disabled={isSearching || !query.trim()}
      >
        {isSearching ? '⏳ Searching...' : '🚀 Search Documents'}
      </button>

      {/* Controls */}
      <div className="control-group">
        <div className="control-label">
          <span>Top-K Documents</span>
          <span className="control-value">{topK}</span>
        </div>
        <input
          type="range"
          className="slider"
          min="1"
          max="20"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-label">
          <span>Similarity Threshold</span>
          <span className="control-value">{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          className="slider"
          min="0"
          max="1"
          step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
        />
      </div>

      {/* Sample Queries */}
      <div className="sample-queries">
        <h4>💡 Quick Queries</h4>
        {SAMPLE_QUERIES.map((sq, i) => (
          <button
            key={i}
            className="sample-btn"
            onClick={() => handleSampleClick(sq.text)}
            disabled={isSearching}
          >
            {sq.icon} {sq.text}
          </button>
        ))}
      </div>
    </motion.div>
  )
}
