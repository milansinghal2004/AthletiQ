import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ProfilePanel = ({ user }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchHistory();
    }
  }, [user]);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`http://127.0.0.1:3000/api/history/${user.id}`);
      if (res.data.success) {
        setHistory(res.data.history);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      setLoading(false);
    }
  };

  const calculateAverage = () => {
    if (history.length === 0) return 0;
    const scores = history.map(h => h.accuracy_score).filter(s => s != null);
    if (scores.length === 0) return 0;
    return (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1);
  };

  return (
    <div className="workflow-card" style={{ marginBottom: '32px' }}>
      <div className="wf-header">
        <div className="wf-header-title">// USER PROFILE & ANALYTICS</div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button 
            onClick={fetchHistory}
            className="btn-ghost"
            style={{ padding: '4px 12px', fontSize: '9px' }}
          >
            ↻ REFRESH DATA
          </button>
          <div className="wf-dots">
            <div className="wf-dot" style={{ background: '#ff5f57' }}></div>
            <div className="wf-dot" style={{ background: '#febc2e' }}></div>
            <div className="wf-dot" style={{ background: '#28c840' }}></div>
          </div>
        </div>
      </div>
      <div className="wf-panel">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', marginBottom: '32px' }}>
          <div className="stat-box" style={{ background: 'var(--bg2)', padding: '24px', border: '1px solid var(--border)' }}>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '2px' }}>AVERAGE ACCURACY</div>
            <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: '38px', fontWeight: '700', color: 'var(--acid)', marginTop: '8px' }}>
              {calculateAverage()}%
            </div>
          </div>
          <div className="stat-box" style={{ background: 'var(--bg2)', padding: '24px', border: '1px solid var(--border)' }}>
            <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '11px', color: 'var(--text-dim)', letterSpacing: '2px' }}>TOTAL SESSIONS</div>
            <div style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: '38px', fontWeight: '700', color: 'var(--acid2)', marginTop: '8px' }}>
              {history.length}
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'left' }}>
          <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '12px', color: 'var(--text-dim)', letterSpacing: '3px', marginBottom: '16px', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
            PERFORMANCE HISTORY
          </div>
          {loading ? (
            <div style={{ color: 'var(--text-dim)', fontFamily: "'Share Tech Mono', monospace" }}>SYNCING DATA...</div>
          ) : history.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', fontFamily: "'Share Tech Mono', monospace" }}>No analysis data found. Run a session to begin tracking.</div>
          ) : (
            <div style={{ display: 'grid', gap: '12px' }}>
              {history.map((entry) => (
                <div key={entry.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px', background: 'var(--bg1)', border: '1px solid var(--border)' }}>
                  <div>
                    <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '12px', color: '#fff' }}>{entry.shot_type || 'Unknown Shot'}</div>
                    <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '9px', color: 'var(--text-dim)', marginTop: '4px' }}>
                      {new Date(entry.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '16px', color: entry.accuracy_score > 80 ? 'var(--acid)' : 'var(--acid2)', fontWeight: '700' }}>
                    {entry.accuracy_score?.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProfilePanel;
