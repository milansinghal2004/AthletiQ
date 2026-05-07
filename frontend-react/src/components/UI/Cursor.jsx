import React, { useEffect, useState, useRef } from 'react';

const Cursor = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [ringPosition, setRingPosition] = useState({ x: 0, y: 0 });
  const [batMode, setBatMode] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  
  const ringRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    const animateRing = () => {
      ringRef.current.x += (position.x - ringRef.current.x) * 0.15;
      ringRef.current.y += (position.y - ringRef.current.y) * 0.15;
      setRingPosition({ x: ringRef.current.x, y: ringRef.current.y });
      requestAnimationFrame(animateRing);
    };

    const handleMouseEnter = () => setIsHovering(true);
    const handleMouseLeave = () => setIsHovering(false);

    window.addEventListener('mousemove', handleMouseMove);
    const animId = requestAnimationFrame(animateRing);

    const interactiveElements = document.querySelectorAll('a, button, select, .pipe-step, .feat-card, .drop-zone');
    interactiveElements.forEach(el => {
      el.addEventListener('mouseenter', handleMouseEnter);
      el.addEventListener('mouseleave', handleMouseLeave);
    });

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animId);
      interactiveElements.forEach(el => {
        el.removeEventListener('mouseenter', handleMouseEnter);
        el.removeEventListener('mouseleave', handleMouseLeave);
      });
    };
  }, [position.x, position.y]);

  const toggleBatMode = () => setBatMode(!batMode);

  return (
    <>
      <div 
        className="cursor" 
        style={{ 
          left: `${position.x}px`, 
          top: `${position.y}px`,
          opacity: batMode ? 0 : 1,
          width: isHovering ? '14px' : '8px',
          height: isHovering ? '14px' : '8px',
          pointerEvents: 'none'
        }} 
      />
      <div 
        className="cursor-ring" 
        style={{ 
          left: `${ringPosition.x}px`, 
          top: `${ringPosition.y}px`,
          opacity: batMode ? 0 : 1,
          width: isHovering ? '56px' : '36px',
          height: isHovering ? '56px' : '36px',
          borderColor: isHovering ? 'rgba(0, 255, 136, 0.8)' : 'rgba(0, 255, 136, 0.4)',
          pointerEvents: 'none'
        }} 
      />
      <div 
        className="bat-cursor" 
        style={{ 
          left: `${position.x}px`, 
          top: `${position.y}px`,
          opacity: batMode ? 1 : 0,
          fontSize: isHovering ? '32px' : '24px',
          pointerEvents: 'none'
        }}
      >
        🏏
      </div>
      <button 
        className="cursor-toggle" 
        onClick={toggleBatMode}
        style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 10000 }}
      >
        {batMode ? '🔵 Normal Cursor' : '🏏 Bat Cursor'}
      </button>
    </>
  );
};

export default Cursor;
