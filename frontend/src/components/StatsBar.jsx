import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const API_BASE = 'http://localhost:8000'

export default function StatsBar() {
  const [stats, setStats] = useState(null)
  const [online, setOnline] = useState(false)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/stats`)
        if (res.ok) {
          const data = await res.json()
          setStats(data)
          setOnline(data.engine_ready)
        }
      } catch {
        setOnline(false)
      }
    }
    fetchStats()
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <motion.div
      className="stats-bar glass-panel"
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="brand">
        <div className="brand-icon">🧠</div>
        <span className="brand-text">RAG Intelligence</span>
      </div>

      <div className="stats-group">
        <div className="stat-item">
          <div className="stat-value">
            {stats ? stats.total_vectors.toLocaleString() : '---'}
          </div>
          <div className="stat-label">Vectors</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">
            {stats ? `${stats.vector_dim}d` : '---'}
          </div>
          <div className="stat-label">Dimension</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">
            {stats ? `${stats.index_size_mb} MB` : '---'}
          </div>
          <div className="stat-label">Index</div>
        </div>
        <div className={`status-badge ${online ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {online ? 'Online' : 'Offline'}
        </div>
      </div>
    </motion.div>
  )
}
