import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Layout/Navbar';
import Hero from './components/Home/Hero';
import StatsStrip from './components/Home/StatsStrip';
import LoginPanel from './components/Workflow/LoginPanel';
import UploadPanel from './components/Workflow/UploadPanel';
import AnalysePanel from './components/Workflow/AnalysePanel';
import StepTracker from './components/Workflow/StepTracker';
import ProfilePanel from './components/Workflow/ProfilePanel';
import Pipeline from './components/Information/Pipeline';
import Features from './components/Information/Features';
import Timeline from './components/Information/Timeline';
import Cursor from './components/UI/Cursor';

import ProcessingOverlay from './components/UI/ProcessingOverlay';

// Configure axios defaults for the backend
axios.defaults.baseURL = 'http://127.0.0.1:3000';
function App() {
  const [user, setUser] = useState(null);
  const [step, setStep] = useState(1);
  const [currentFile, setCurrentFile] = useState(null);
  const [isGated, setIsGated] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [view, setView] = useState('practice'); // 'practice' or 'profile'

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    setIsGated(false);
  };

  const handleLogout = () => {
    setUser(null);
    setIsGated(true);
    setStep(1);
    setCurrentFile(null);
    setView('practice');
  };

  const handleFileSelect = (file) => {
    setCurrentFile(file);
  };

  const handleRemoveFile = () => {
    setCurrentFile(null);
  };

  const handleProceed = () => {
    if (currentFile) setStep(2);
  };

  const handleLaunchDashboard = async () => {
    if (!currentFile) return;
    
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('video', currentFile);
      
      // Use the fast upload endpoint instead of the full analysis runner
      const uploadRes = await axios.post('/api/upload', formData);
      if (uploadRes.data.success && uploadRes.data.video_path) {
        const launchRes = await axios.get(`/launch-dashboard?video=${encodeURIComponent(uploadRes.data.video_path)}&user_id=${user?.id || ''}`);
        if (launchRes.data.success) {
          window.open(launchRes.data.url, '_blank');
          setIsProcessing(false);
        }
      }
    } catch (err) {
      console.error('Launch failed:', err);
      setIsProcessing(false);
      const errMsg = err.response?.data?.error || err.message || 'Unknown Network Error';
      alert(`Neural Link failed: ${errMsg}\n\nPlease ensure the backend server is running on port 3000.`);
    }
  };

  return (
    <div className="page">
      <div className="grid-bg"></div>
      <div className="orb orb1"></div><div className="orb orb2"></div><div className="orb orb3"></div>
      <div className="scanline"></div>
      <Cursor />
      
      {isProcessing && <ProcessingOverlay />}
      
      <Navbar user={user} onLogout={handleLogout} setView={setView} currentView={view} />
      
      <main>
        <Hero />
        <StatsStrip />
        
        <section className="workflow-section" id="workflow">
          {!user && <LoginPanel onLoginSuccess={handleLoginSuccess} />}
          
          <div id="gatedContent" className={isGated ? 'gated-locked' : ''}>
            {isGated && (
              <div className="gate-overlay" style={{ display: 'flex', zIndex: 10 }}>
                AUTHENTICATION REQUIRED TO ACCESS PIPELINE
              </div>
            )}
            
            {view === 'profile' ? (
              <ProfilePanel user={user} />
            ) : (
              <>
                <StepTracker currentStep={step} />
                <div className="workflow-card">
                  <div className="wf-header">
                    <div className="wf-header-title">
                      {step === 1 ? '// STEP 01 — UPLOAD VIDEO' : '// STEP 02 — CONFIGURE ANALYSIS'}
                    </div>
                    <div className="wf-dots">
                      <div className="wf-dot" style={{ background: '#ff5f57' }}></div>
                      <div className="wf-dot" style={{ background: '#febc2e' }}></div>
                      <div className="wf-dot" style={{ background: '#28c840' }}></div>
                    </div>
                  </div>
                  
                  {step === 1 ? (
                    <UploadPanel 
                      currentFile={currentFile} 
                      onFileSelect={handleFileSelect} 
                      onRemoveFile={handleRemoveFile}
                      onProceed={handleProceed}
                    />
                  ) : (
                    <AnalysePanel 
                      currentFile={currentFile} 
                      onBack={() => setStep(1)} 
                      onLaunchDashboard={handleLaunchDashboard}
                    />
                  )}
                </div>
              </>
            )}
          </div>
        </section>

        <div className="info-sections">
          <Pipeline />
          <Features />
          <Timeline />
        </div>
      </main>

      <footer>
        <div className="footer-logo">Athleti<span>Q</span></div>
        <div className="footer-copy">AI-POWERED CRICKET MOTION ANALYSIS &nbsp;·&nbsp; MARCH 2025 – APRIL 2025</div>
        <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: '10px', color: 'var(--text-dim)' }}>
          <span style={{ color: 'var(--acid)' }}>■</span> SYSTEM ACTIVE
        </div>
      </footer>
    </div>
  );
}

export default App;
