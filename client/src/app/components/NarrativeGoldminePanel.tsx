import React from 'react';


const NarrativeGoldminePanel: React.FC = () => {
  const handleOpenInNewTab = () => {
    window.open('https://narrativegoldmine.com/#/page/', '_blank', 'noopener,noreferrer');
  };

  const panelStyle: React.CSSProperties = {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a1a1a',
    color: '#ffffff',
    padding: '2rem',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '1rem 2rem',
    fontSize: '1rem',
    backgroundColor: '#4a5568',
    color: 'white',
    border: 'none',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  };

  return (
    <div style={panelStyle}>
      <h2 style={{ marginBottom: '1rem' }}>Narrative Goldmine</h2>
      <p style={{ marginBottom: '2rem', textAlign: 'center', color: '#a0aec0' }}>
        Click below to open Narrative Goldmine in a new tab
      </p>
      <button
        onClick={handleOpenInNewTab}
        style={buttonStyle}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#2d3748';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = '#4a5568';
        }}
      >
        Open Narrative Goldmine
      </button>
    </div>
  );
};

export default NarrativeGoldminePanel;
