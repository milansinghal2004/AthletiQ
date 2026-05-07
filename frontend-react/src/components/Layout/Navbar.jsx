import React from 'react';

const Navbar = ({ user, onLogout, setView, currentView, systemStatus = "SYSTEM ONLINE" }) => {
  return (
    <nav>
      <div className="nav-logo">Athleti<span>Q</span></div>
      <div className="nav-links">
        <a 
          href="#workflow" 
          onClick={() => setView('practice')}
          className={currentView === 'practice' ? 'active' : ''}
        >
          {user ? 'Practice Pipeline' : 'Upload & Analyse'}
        </a>
        {user && (
          <a 
            href="#workflow" 
            onClick={() => setView('profile')}
            className={currentView === 'profile' ? 'active' : ''}
          >
            Analytics Dashboard
          </a>
        )}
        <a href="#pipeline">Pipeline</a>
        <a href="#features">Features</a>
        <a href="#timeline">Timeline</a>
      </div>
      <div className="nav-status" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div className="status-dot"></div>
          <span id="systemStatus">{systemStatus}</span>
        </div>
        {user && (
          <button 
            id="userBadge" 
            onClick={onLogout}
            title="Logout"
            style={{
              minWidth: '36px',
              minHeight: '36px',
              borderRadius: '50%',
              border: '1px solid rgba(0,255,136,0.35)',
              background: 'rgba(0,255,136,0.08)',
              color: '#fff',
              fontFamily: "'Share Tech Mono', monospace",
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            {user.username?.[0].toUpperCase() || 'U'}
          </button>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
