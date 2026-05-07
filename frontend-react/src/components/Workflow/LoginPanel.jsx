import React, { useState } from 'react';
import axios from 'axios';

const LoginPanel = ({ onLoginSuccess }) => {
  const [tab, setTab] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    setStatus('');
    try {
      const res = await axios.post('http://127.0.0.1:3000/api/login', { username, password });
      if (res.data.success) {
        setStatus('✔ Login successful');
        onLoginSuccess(res.data.user);
      } else {
        setStatus(res.data.message || 'Login failed.');
      }
    } catch (err) {
      setStatus('Server connection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    setLoading(true);
    setStatus('');
    try {
      const res = await axios.post('http://127.0.0.1:3000/api/register', { username, password, email });
      if (res.data.success) {
        setStatus('✔ Registration successful');
        setTab('login');
      } else {
        setStatus(res.data.message || 'Registration failed.');
      }
    } catch (err) {
      setStatus('Server connection failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="workflow-card" style={{ marginBottom: '32px', padding: '24px', background: 'var(--bg2)', border: '1px solid var(--border)' }}>
      <div className="wf-header">
        <div className="wf-header-title">// USER LOGIN & PROFILE</div>
        <div className="wf-dots">
          <div className="wf-dot" style={{ background: '#ff5f57' }}></div>
          <div className="wf-dot" style={{ background: '#febc2e' }}></div>
          <div className="wf-dot" style={{ background: '#28c840' }}></div>
        </div>
      </div>
      <div style={{ display: 'grid', gap: '22px' }}>
        <div style={{ display: 'flex', gap: '12px', marginBottom: '6px' }}>
          <button className={`btn-ghost ${tab === 'login' ? 'active' : ''}`} onClick={() => setTab('login')} style={{ flex: 1 }}>Login</button>
          <button className={`btn-ghost ${tab === 'register' ? 'active' : ''}`} onClick={() => setTab('register')} style={{ flex: 1 }}>Register</button>
        </div>
        
        {tab === 'login' ? (
            <div id="loginForm" style={{ display: 'grid', gap: '10px' }}>
              <div style={{ fontFamily: "'Share Tech Mono',monospace", fontSize: '12px', color: 'var(--text-dim)', letterSpacing: '2px', textTransform: 'uppercase' }}>Login</div>
              <input value={username} onChange={e => setUsername(e.target.value)} type="text" placeholder="Username" />
              <input value={password} onChange={e => setPassword(e.target.value)} type="password" placeholder="Password" />
              <button className="btn-primary" onClick={handleLogin} disabled={loading} style={{ width: '100%' }}>{loading ? 'PROCESSING...' : 'LOGIN'}</button>
              <div id="loginStatus" style={{ fontFamily: "'Share Tech Mono',monospace", fontSize: '10px', color: 'var(--acid)', minHeight: '18px' }}>{status}</div>
            </div>
          ) : (
            <div id="registerForm" style={{ display: 'grid', gap: '10px' }}>
              <div style={{ fontFamily: "'Share Tech Mono',monospace", fontSize: '12px', color: 'var(--text-dim)', letterSpacing: '2px', textTransform: 'uppercase' }}>Register</div>
              <input value={username} onChange={e => setUsername(e.target.value)} type="text" placeholder="Username" />
              <input value={password} onChange={e => setPassword(e.target.value)} type="password" placeholder="Password" />
              <input value={email} onChange={e => setEmail(e.target.value)} type="email" placeholder="Email" />
              <button className="btn-primary" onClick={handleRegister} disabled={loading} style={{ width: '100%' }}>{loading ? 'PROCESSING...' : 'REGISTER'}</button>
              <div id="registerStatus" style={{ fontFamily: "'Share Tech Mono',monospace", fontSize: '10px', color: 'var(--acid)', minHeight: '18px' }}>{status}</div>
            </div>
          )}
      </div>
    </div>
  );
};

export default LoginPanel;
