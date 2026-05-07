import React from 'react';

const features = [
  { icon: '🦴', title: 'Pose Estimation', desc: 'Extracts 33 skeletal keypoints per frame using MediaPipe Pose, with confidence scoring and temporal smoothing.', tag: 'MediaPipe · Real-time' },
  { icon: '🏏', title: 'Shot Detection', desc: 'Classifies batting shots — drive, pull, sweep, cut and more — from skeletal trajectory and joint velocity signatures.', tag: 'ML Classification · 8 Shots', accent: 'feat-card-accent' },
  { icon: '📐', title: 'DTW Analysis', desc: 'Dynamic Time Warping aligns your motion with expert reference clips, handling tempo variation for true similarity scores.', tag: 'DTW · Joint-level', accent: 'feat-card-fire' },
  { icon: '⚡', title: 'Clip Extraction', desc: 'Locates batting action windows within full footage, isolating each stroke for frame-accurate analysis automatically.', tag: 'OpenCV · Auto-trim', accent: 'feat-card-accent' },
  { icon: '🎯', title: 'Performance Scoring', desc: 'Per-joint and aggregate technique scores reveal exactly which body segments deviate from optimal form.', tag: 'Quantitative · Per-joint' },
  { icon: '🤖', title: 'AI Feedback', desc: 'Converts numerical deviations into natural language coaching cues — specific, actionable corrections per joint.', tag: 'LLM-powered · Actionable', accent: 'feat-card-fire' }
];

const Features = () => {
  return (
    <section className="section-slim" id="features">
      <div className="section-tag">Core Capabilities</div>
      <h2 className="section-title">What <em>AthletiQ</em> Does</h2>
      <div className="features-grid">
        {features.map((f, i) => (
          <div key={i} className={`feat-card ${f.accent || ''}`}>
            <span className="feat-icon">{f.icon}</span>
            <div className="feat-title">{f.title}</div>
            <div className="feat-desc">{f.desc}</div>
            <span className="feat-tag">{f.tag}</span>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Features;
