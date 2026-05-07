import React from 'react';

const Pipeline = () => {
  const steps1 = [
    { icon: '🎥', num: '01', label: 'Video Input' },
    { icon: '🦴', num: '02', label: 'Pose Estimation' },
    { icon: '✅', num: '03', label: 'Validation' },
    { icon: '🏏', num: '04', label: 'Shot Detection' },
    { icon: '✂️', num: '05', label: 'Clip Extraction' }
  ];

  const steps2 = [
    { icon: '🔄', num: '06', label: 'Synchronization' },
    { icon: '📊', num: '07', label: 'DTW Comparison' },
    { icon: '🎯', num: '08', label: 'Performance Score' },
    { icon: '🤖', num: '09', label: 'AI Feedback' },
    { icon: '📱', num: '10', label: 'Visualization' }
  ];

  return (
    <section className="section-slim" id="pipeline">
      <div className="section-tag">System Architecture</div>
      <h2 className="section-title">10-Stage <em>Analysis</em> Pipeline</h2>
      <div className="pipeline">
        {steps1.map(s => (
          <div key={s.num} className="pipe-step">
            <div className="pipe-node">
              <span className="pipe-icon">{s.icon}</span>
              <span className="pipe-num">{s.num}</span>
            </div>
            <div className="pipe-label">{s.label}</div>
          </div>
        ))}
      </div>
      <div className="pipeline2" style={{ marginTop: '32px' }}>
        {steps2.map(s => (
          <div key={s.num} className="pipe-step">
            <div className="pipe-node">
              <span className="pipe-icon">{s.icon}</span>
              <span className="pipe-num">{s.num}</span>
            </div>
            <div className="pipe-label">{s.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Pipeline;
