import React, { useState } from 'react';

const AnalysePanel = ({ currentFile, onBack, onLaunchDashboard }) => {
  const [shotType, setShotType] = useState('None');
  const [clickX, setClickX] = useState(320);
  const [clickY, setClickY] = useState(240);

  return (
    <div className="wf-panel active">
      <div className="analyse-panel-inner">
        <div className="analyse-file-badge">
          <span>🎬</span><span>{currentFile?.name || 'video.mp4'}</span>
        </div>
        <div className="analyse-desc">
          AthletiQ will run <span style={{ color: 'var(--acid)', fontFamily: "'Share Tech Mono', monospace" }}>pose_estimation.py</span>
          on your footage — extracting keypoints, detecting your batting shot,
          and comparing your form against reference patterns.
        </div>
        
        <div style={{ display: 'grid', gap: '14px', maxWidth: '520px', width: '100%', textAlign: 'left' }}>
          <label style={{ display: 'grid', gap: '8px', fontFamily: "'Share Tech Mono', monospace", fontSize: '11px', color: 'var(--text-dim)' }}>
            Shot Type
            <select 
              value={shotType} 
              onChange={e => setShotType(e.target.value)}
            >
              <option value="None">Auto detect</option>
              <option value="cover">Cover Drive</option>
              <option value="defense">Defense</option>
              <option value="flick">Flick</option>
              <option value="hook">Hook</option>
              <option value="late_cut">Late Cut</option>
              <option value="lofted">Lofted Shot</option>
              <option value="pull">Pull Shot</option>
              <option value="square_cut">Square Cut</option>
              <option value="straight">Straight Drive</option>
              <option value="sweep">Sweep Shot</option>
            </select>
          </label>
          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
            <label style={{ display: 'grid', gap: '8px', fontFamily: "'Share Tech Mono', monospace", fontSize: '11px', color: 'var(--text-dim)', width: '100%', maxWidth: '210px' }}>
              Click X
              <input value={clickX} onChange={e => setClickX(e.target.value)} type="number" />
            </label>
            <label style={{ display: 'grid', gap: '8px', fontFamily: "'Share Tech Mono', monospace", fontSize: '11px', color: 'var(--text-dim)', width: '100%', maxWidth: '210px' }}>
              Click Y
              <input value={clickY} onChange={e => setClickY(e.target.value)} type="number" />
            </label>
          </div>
        </div>

        <button className="analyse-btn-large" onClick={onLaunchDashboard}>⚡ ANALYSE SHOT</button>
        <div className="analyse-desc" style={{ fontSize: '11px', marginTop: '-16px' }}>
          Clicking will open the advanced technical workspace.
        </div>
        
        <button className="btn-back" onClick={onBack} style={{ marginTop: '8px' }}>← Change Video</button>
      </div>
    </div>
  );
};

export default AnalysePanel;
