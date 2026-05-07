import React from 'react';

const Hero = () => {
  return (
    <section className="hero">
      <div className="hero-eyebrow">AI-Powered Sports Analytics // Cricket Motion Analysis</div>
      <h1 className="hero-title glitch" data-text="MOTION">
        <span className="line1">CRICKET</span>
        <span className="line2">MOTION</span>
        <span className="line3">INTELLIGENCE</span>
      </h1>
      <p className="hero-desc">
        Upload your cricket footage. AthletiQ extracts skeletal keypoints,
        detects your batting shot, and scores your technique against reference
        patterns — delivering coaching feedback at machine speed.
      </p>
      <div className="hero-actions">
        <a href="#workflow" className="btn-primary">Start Analysis</a>
        <a href="#pipeline" className="btn-secondary">View Pipeline</a>
      </div>

      <div className="hero-visual">
        <svg viewBox="0 0 300 420" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            <filter id="glow2"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            <radialGradient id="bodyGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#00ff88" stopOpacity=".6"/>
              <stop offset="100%" stopColor="#00e5ff" stopOpacity=".1"/>
            </radialGradient>
          </defs>
          <ellipse cx="150" cy="280" rx="100" ry="120" fill="url(#bodyGrad)" opacity=".15" filter="url(#glow2)"/>
          <g className="float-anim" filter="url(#glow)">
            <circle cx="160" cy="60" r="18" stroke="#00ff88" strokeWidth="1.5" fill="rgba(0,255,136,0.05)"/>
            <line x1="160" y1="78" x2="160" y2="100" stroke="#00e5ff" strokeWidth="1.5"/>
            <line x1="110" y1="110" x2="210" y2="110" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="160" y1="100" x2="155" y2="220" stroke="#00e5ff" strokeWidth="1.5" strokeDasharray="4 3"/>
            <line x1="110" y1="110" x2="70" y2="160" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="70" y1="160" x2="40" y2="130" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="210" y1="110" x2="240" y2="150" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="240" y1="150" x2="255" y2="200" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="40" y1="130" x2="20" y2="200" stroke="#ff6b35" strokeWidth="3"/>
            <rect x="12" y="195" width="14" height="50" rx="2" fill="rgba(255,107,53,0.3)" stroke="#ff6b35" strokeWidth="1.5"/>
            <line x1="125" y1="220" x2="185" y2="220" stroke="#00e5ff" strokeWidth="1.5"/>
            <line x1="125" y1="220" x2="105" y2="300" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="105" y1="300" x2="115" y2="370" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="185" y1="220" x2="195" y2="300" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="195" y1="300" x2="225" y2="360" stroke="#00ff88" strokeWidth="1.5"/>
            <line x1="115" y1="370" x2="95" y2="380" stroke="#00e5ff" strokeWidth="2"/>
            <line x1="225" y1="360" x2="255" y2="370" stroke="#00e5ff" strokeWidth="2"/>
            <circle cx="160" cy="78" r="4" fill="#00ff88" opacity=".9"/>
            <circle cx="110" cy="110" r="5" fill="#00ff88"/><circle cx="210" cy="110" r="5" fill="#00ff88"/>
            <circle cx="70" cy="160" r="4" fill="#00e5ff"/><circle cx="240" cy="150" r="4" fill="#00e5ff"/>
            <circle cx="40" cy="130" r="5" fill="#00ff88"/><circle cx="255" cy="200" r="5" fill="#00e5ff"/>
            <circle cx="155" cy="220" r="5" fill="#00ff88"/>
            <circle cx="105" cy="300" r="5" fill="#00ff88"/><circle cx="195" cy="300" r="5" fill="#00ff88"/>
            <text x="218" y="108" fill="#00ff88" fontFamily="Share Tech Mono" fontSize="8" opacity=".6">0.97</text>
            <text x="60"  y="108" fill="#00ff88" fontFamily="Share Tech Mono" fontSize="8" opacity=".6">0.95</text>
          </g>
          <rect x="0" y="0" width="20" height="1" fill="#00ff88" opacity=".5"/>
          <rect x="0" y="0" width="1" height="20" fill="#00ff88" opacity=".5"/>
          <rect x="280" y="0" width="20" height="1" fill="#00ff88" opacity=".5"/>
          <rect x="299" y="0" width="1" height="20" fill="#00ff88" opacity=".5"/>
          <rect x="0" y="419" width="20" height="1" fill="#00ff88" opacity=".5"/>
          <rect x="0" y="400" width="1" height="20" fill="#00ff88" opacity=".5"/>
          <rect x="280" y="419" width="20" height="1" fill="#00ff88" opacity=".5"/>
          <rect x="299" y="400" width="1" height="20" fill="#00ff88" opacity=".5"/>
          <text x="8" y="12"  fill="#00ff88" fontFamily="Share Tech Mono" fontSize="8" opacity=".5">POSE_TRACK</text>
          <text x="8" y="410" fill="#00e5ff" fontFamily="Share Tech Mono" fontSize="8" opacity=".5">CONF:0.96</text>
        </svg>
      </div>
    </section>
  );
};

export default Hero;
