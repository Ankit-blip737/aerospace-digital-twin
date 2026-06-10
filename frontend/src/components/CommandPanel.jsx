import React from 'react';

const FAULT_ACTIONS = [
  { faultClass: 1, label: 'Rotor Damage',   icon: '🔴', desc: 'High vibration', color: '#f97316' },
  { faultClass: 2, label: 'Sensor Drift',   icon: '🟠', desc: 'IMU/Mag error',  color: '#a855f7' },
  { faultClass: 3, label: 'GPS Blackout',   icon: '🟡', desc: 'Signal loss',    color: '#eab308' },
  { faultClass: 4, label: 'Battery Failure',icon: '⚡', desc: 'Voltage sag',    color: '#ef4444' },
];

export default function CommandPanel({ selectedDrone, onInjectFault, onReset, onRTL }) {
  return (
    <div className="cmd-panel">
      <div className="cmd-panel-header">
        <span className="cmd-panel-icon">⌨</span>
        <span className="cmd-panel-title">COMMAND CONSOLE</span>
        <span className="cmd-panel-target">{selectedDrone}</span>
      </div>

      <p className="cmd-panel-section-label">⚠ FAULT INJECTION</p>
      <div className="cmd-fault-grid">
        {FAULT_ACTIONS.map(action => (
          <button
            key={action.faultClass}
            className="cmd-btn cmd-btn-fault"
            style={{ '--accent': action.color }}
            onClick={() => onInjectFault(selectedDrone, action.faultClass)}
            title={action.desc}
          >
            <span className="cmd-btn-icon">{action.icon}</span>
            <span className="cmd-btn-label">{action.label}</span>
            <span className="cmd-btn-desc">{action.desc}</span>
          </button>
        ))}
      </div>

      <p className="cmd-panel-section-label" style={{ marginTop: '1rem' }}>✦ MISSION COMMANDS</p>
      <div className="cmd-mission-grid">
        <button
          className="cmd-btn cmd-btn-rtl"
          onClick={() => onRTL(selectedDrone)}
        >
          <span>🛬</span>
          <span>RTL</span>
          <span className="cmd-btn-desc">Return to Launch</span>
        </button>
        <button
          className="cmd-btn cmd-btn-reset"
          onClick={() => onReset(selectedDrone)}
        >
          <span>✅</span>
          <span>RESET</span>
          <span className="cmd-btn-desc">Clear all faults</span>
        </button>
      </div>
    </div>
  );
}
