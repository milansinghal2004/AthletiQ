import React from 'react';

const StepTracker = ({ currentStep }) => {
  const steps = [
    { num: '01', label: 'Upload' },
    { num: '02', label: 'Analyse' },
    { num: '03', label: 'Results' }
  ];

  return (
    <div className="steps-tracker" style={{ opacity: 1 }}>
      {steps.map((step, index) => (
        <React.Fragment key={step.num}>
          <div className={`step-node ${currentStep >= index + 1 ? 'active' : ''}`}>
            <div className={`step-circle ${currentStep === index + 1 ? 'active' : ''} ${currentStep > index + 1 ? 'done' : ''}`}>
              {step.num}
            </div>
            <div className="step-label">{step.label}</div>
          </div>
          {index < steps.length - 1 && (
            <div className={`step-line ${currentStep > index + 1 ? 'done' : ''} ${currentStep === index + 1 ? 'active' : ''}`}></div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default StepTracker;
