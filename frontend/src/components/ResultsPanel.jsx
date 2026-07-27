import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function ResultsPanel({ results, isSearching }) {
  const [expandedId, setExpandedId] = useState(null)
  const [contentFilter, setContentFilter] = useState('')

  if (isSearching) {
    return (
      <motion.div
        className="results-panel glass-panel"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="loading-spinner">
          <div className="spinner" />
          <span>Fixing typos & performing vector search across 200K documents...</span>
        </div>
      </motion.div>
    )
  }

  if (!results) {
    return (
      <motion.div
        className="results-panel glass-panel"
        initial={{ x: 80, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <h3>Neural RAG Search Engine</h3>
          <p>Enter a query in the search box. Typos and spelling errors will be automatically corrected before vector search!</p>
        </div>
      </motion.div>
    )
  }

  // Filter retrieved source documents based on content search input
  const filteredSources = results.sources.filter(s => {
    if (!contentFilter.trim()) return true
    return s.text.toLowerCase().includes(contentFilter.toLowerCase().trim())
  })

  // Highlight matching keywords in snippet
  const renderHighlightedText = (text, highlight) => {
    if (!highlight || !highlight.trim()) return text
    const parts = text.split(new RegExp(`(${highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
    return parts.map((part, i) =>
      part.toLowerCase() === highlight.toLowerCase().trim() ? (
        <mark key={i} style={{ background: '#f59e0b', color: '#000', borderRadius: 2, padding: '0 2px' }}>
          {part}
        </mark>
      ) : (
        part
      )
    )
  }

  const isQueryCorrected = results.corrected_query && results.corrected_query.toLowerCase().trim() !== results.query.toLowerCase().trim()

  return (
    <motion.div
      className="results-panel glass-panel"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Auto-Correction Banner */}
      {isQueryCorrected && (
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          style={{
            padding: '10px 16px',
            background: 'rgba(99, 102, 241, 0.15)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            borderRadius: 10,
            fontSize: 13,
            color: '#818cf8',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 8
          }}
        >
          <span>✨</span>
          <div>
            <strong>Auto-Corrected Query:</strong> Searching for <em style={{ color: '#22d3ee' }}>"{results.corrected_query}"</em> (corrected from <s>"{results.query}"</s>)
          </div>
        </motion.div>
      )}

      {/* AI Answer Card */}
      <motion.div
        className="answer-card"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        <div className="answer-header">
          <h3>✨ AI Response</h3>
          <span className="answer-badge">
            {results.sources.length} sources matched
          </span>
        </div>
        <div className="answer-text">{results.answer}</div>
        <div className="latency-badge">
          ⚡ {results.latency}s latency (Auto-Corrected Vector Search)
        </div>
      </motion.div>

      {/* Content Search & Filter Box */}
      {results.sources.length > 0 && (
        <div style={{ marginTop: 12, marginBottom: 4 }}>
          <div className="search-input-wrapper">
            <span className="search-icon">🔎</span>
            <input
              type="text"
              className="search-input"
              style={{ fontSize: 13, padding: '10px 14px 10px 40px' }}
              placeholder="Type to filter & highlight text inside retrieved document contents..."
              value={contentFilter}
              onChange={(e) => setContentFilter(e.target.value)}
            />
            {contentFilter && (
              <button
                onClick={() => setContentFilter('')}
                style={{
                  position: 'absolute',
                  right: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  fontSize: 14
                }}
              >
                ✕
              </button>
            )}
          </div>
        </div>
      )}

      {/* Sources List */}
      {results.sources.length > 0 && (
        <>
          <div className="sources-header">
            📚 Source Documents ({filteredSources.length} of {results.sources.length} shown)
          </div>

          <AnimatePresence>
            {filteredSources.length > 0 ? (
              filteredSources.map((source, index) => (
                <motion.div
                  key={`${source.doc_id}-${source.chunk_id}`}
                  className={`source-card ${expandedId === index ? 'expanded' : ''}`}
                  initial={{ y: 30, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  onClick={() => setExpandedId(expandedId === index ? null : index)}
                >
                  <div className="source-header">
                    <div className="source-rank">
                      <div className="rank-badge">#{source.rank}</div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>
                          Doc #{source.doc_id}
                        </div>
                        <div className="source-meta">
                          Chunk {source.chunk_id} • Vector {source.vector_id}
                        </div>
                      </div>
                    </div>
                    <div className="similarity-label">
                      {(source.similarity * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="similarity-bar">
                    <motion.div
                      className="similarity-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(source.similarity * 100, 2)}%` }}
                      transition={{ duration: 0.8, delay: index * 0.05 }}
                    />
                  </div>

                  <div className="source-text">
                    {renderHighlightedText(source.text, contentFilter)}
                  </div>

                  <div className="source-meta" style={{ marginTop: 6 }}>
                    {expandedId === index ? '▲ Click to collapse' : '▼ Click to expand text snippet'}
                  </div>
                </motion.div>
              ))
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>
                No document content matched the filter term "{contentFilter}".
              </div>
            )}
          </AnimatePresence>
        </>
      )}
    </motion.div>
  )
}
