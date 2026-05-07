import React from 'react';

const phases = [
  { num: '01', tag: 'Phase I', title: 'Pose Detection', desc: 'MediaPipe integration, keypoint extraction, skeleton smoothing and frame-level validation pipeline.', date: 'Mar 11 – Mar 21' },
  { num: '02', tag: 'Phase II', title: 'Shot Detection', desc: 'Shot classifier training, clip extraction logic, action window segmentation from full footage.', date: 'Mar 21 – Apr 01', tagColor: 'var(--acid2)', tagBorder: 'rgba(0,229,255,.2)' },
  { num: '03', tag: 'Phase III', title: 'Movement Analysis', desc: 'DTW implementation, reference motion library, joint scoring system and AI feedback generation.', date: 'Apr 01 – Apr 15', tagColor: 'var(--fire)', tagBorder: 'rgba(255,107,53,.2)' },
  { num: '04', tag: 'Phase IV', title: 'UI Integration', desc: 'Dashboard build, backend API connection, visualization components and full system integration testing.', date: 'Apr 15 – Apr 25', tagColor: 'var(--acid2)', tagBorder: 'rgba(0,229,255,.2)' }
];

const Timeline = () => {
  return (
    <section className="section-slim" id="timeline">
      <div className="section-tag">Development Roadmap</div>
      <h2 className="section-title">Project <em>Timeline</em></h2>
      <div className="phase-grid">
        {phases.map(p => (
          <div key={p.num} className="phase-card">
            <div className="phase-num">{p.num}</div>
            <span className="phase-tag" style={{ color: p.tagColor, borderColor: p.tagBorder }}>{p.tag}</span>
            <div className="phase-title">{p.title}</div>
            <div className="phase-desc">{p.desc}</div>
            <div className="phase-date">{p.date}</div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Timeline;
