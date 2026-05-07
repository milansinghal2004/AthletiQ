import React from 'react';

const ProcessingOverlay = ({ message = "SCANNING MOTION ARCHITECTURE..." }) => {
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(2, 8, 18, 0.95)',
      backdropFilter: 'blur(20px)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 99999,
      gap: '32px'
    }}>
      <div className="processing-ring" style={{
        width: '120px',
        height: '120px',
        borderRadius: '50%',
        border: '2px solid rgba(0, 255, 136, 0.1)',
        borderTopColor: 'var(--acid)',
        animation: 'spin 1s linear infinite'
      }} />
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: "'Rajdhani', sans-serif",
          fontSize: '28px',
          fontWeight: 700,
          letterSpacing: '8px',
          color: 'var(--acid)',
          textTransform: 'uppercase',
          marginBottom: '12px',
          animation: 'pulse 2s ease-in-out infinite'
        }}>
          {message}
        </div>
        <div style={{
          fontFamily: "'Share Tech Mono', monospace",
          fontSize: '10px',
          letterSpacing: '3px',
          color: 'var(--text-dim)',
          textTransform: 'uppercase'
        }}>
          Establishing Neural Link with AthletiQ Backend
        </div>
      </div>
      
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(0.98); }
        }
      `}</style>
    </div>
  );
};

export default ProcessingOverlay;
