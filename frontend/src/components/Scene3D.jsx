import { motion } from 'framer-motion'

export default function Scene3D({ searching = false }) {
  return (
    <div className="canvas-container" style={{ overflow: 'hidden', background: '#050510' }}>
      {/* Sleek 0% GPU Lightweight CSS Ambient Glow Mesh */}
      <motion.div
        animate={{
          scale: searching ? [1, 1.2, 1] : [1, 1.05, 1],
          opacity: searching ? [0.4, 0.7, 0.4] : [0.25, 0.35, 0.25]
        }}
        transition={{ duration: searching ? 2 : 6, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          position: 'absolute',
          top: '-20%',
          left: '20%',
          width: '600px',
          height: '600px',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(34, 211, 238, 0.1) 50%, rgba(0,0,0,0) 70%)',
          borderRadius: '50%',
          filter: 'blur(80px)',
          pointerEvents: 'none'
        }}
      />

      <motion.div
        animate={{
          scale: searching ? [1.1, 1.3, 1.1] : [1, 1.08, 1],
          opacity: searching ? [0.3, 0.6, 0.3] : [0.2, 0.3, 0.2]
        }}
        transition={{ duration: searching ? 2.5 : 8, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
        style={{
          position: 'absolute',
          bottom: '-10%',
          right: '10%',
          width: '500px',
          height: '500px',
          background: 'radial-gradient(circle, rgba(45, 212, 191, 0.2) 0%, rgba(99, 102, 241, 0.08) 50%, rgba(0,0,0,0) 70%)',
          borderRadius: '50%',
          filter: 'blur(90px)',
          pointerEvents: 'none'
        }}
      />
    </div>
  )
}
