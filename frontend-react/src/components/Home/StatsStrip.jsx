import React, { useState, useEffect, useRef } from 'react';

const AnimatedCounter = ({ target, suffix = "" }) => {
  const [count, setCount] = useState(0);
  const elementRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        let start = 0;
        const duration = 1500;
        const increment = target / (duration / 16);
        
        const timer = setInterval(() => {
          start += increment;
          if (start >= target) {
            setCount(target);
            clearInterval(timer);
          } else {
            setCount(Math.floor(start));
          }
        }, 16);
        observer.disconnect();
      }
    }, { threshold: 0.1 });

    if (elementRef.current) observer.observe(elementRef.current);
    return () => observer.disconnect();
  }, [target]);

  return <div ref={elementRef} className="stat-val">{count}{suffix}</div>;
};

const StatsStrip = () => {
  return (
    <div className="stats-strip">
      <div className="stat-item">
        <AnimatedCounter target={33} />
        <div className="stat-label">Skeleton Keypoints</div>
      </div>
      <div className="stat-item">
        <AnimatedCounter target={8} />
        <div className="stat-label">Shot Types Detected</div>
      </div>
      <div className="stat-item">
        <AnimatedCounter target={96} suffix="%" />
        <div className="stat-label">Detection Accuracy</div>
      </div>
      <div className="stat-item">
        <AnimatedCounter target={30} />
        <div className="stat-label">FPS Processing</div>
      </div>
    </div>
  );
};

export default StatsStrip;
