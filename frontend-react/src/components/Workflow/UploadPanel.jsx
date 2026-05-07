import React, { useState, useRef } from 'react';

const UploadPanel = ({ onFileSelect, currentFile, onRemoveFile, onProceed }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onFileSelect(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <div className="wf-panel active">
      <div 
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-icon">🎥</div>
        <div className="drop-title">Drop Cricket Footage Here</div>
        <div className="drop-sub">MP4 · MOV · AVI &nbsp;·&nbsp; MAX 500MB &nbsp;·&nbsp; 720p+ RECOMMENDED</div>
        <div className="drop-or">— OR —</div>
        <button className="btn-primary" onClick={() => fileInputRef.current.click()}>Browse File</button>
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          accept="video/*" 
          style={{ display: 'none' }} 
        />
      </div>

      {currentFile && (
        <div className="file-bar visible">
          <div className="file-bar-icon">🎬</div>
          <div className="file-bar-info">
            <div className="file-bar-name">{currentFile.name}</div>
            <div className="file-bar-meta">{(currentFile.size / (1024 * 1024)).toFixed(2)} MB</div>
          </div>
          <button className="file-bar-remove" onClick={onRemoveFile}>✕ REMOVE</button>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '24px' }}>
        <button 
          className="btn-primary" 
          onClick={onProceed}
          disabled={!currentFile} 
          style={{ opacity: currentFile ? 1 : 0.4 }}
        >
          PROCEED TO ANALYSE →
        </button>
      </div>
    </div>
  );
};

export default UploadPanel;
