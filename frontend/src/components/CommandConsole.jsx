import React, { useRef, useEffect } from 'react';

const LOG_COLORS = {
  nominal:         '#10b981',
  vibration_fault: '#f97316',
  sensor_drift:    '#a855f7',
  gps_fault:       '#eab308',
  battery_fault:   '#ef4444',
  system:          '#60a5fa',
  cmd:             '#38bdf8',
};

export default function CommandConsole({ logs }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="console-outer">
      <div className="console-header">
        <span className="console-dot red" />
        <span className="console-dot yellow" />
        <span className="console-dot green" />
        <span className="console-title">FLEET COMMAND LOG</span>
        <span className="console-live-badge">● LIVE</span>
      </div>
      <div className="console-body">
        {logs.length === 0 && (
          <p className="console-empty">Awaiting telemetry stream...</p>
        )}
        {logs.map((log, i) => {
          const color = LOG_COLORS[log.type] || LOG_COLORS.system;
          return (
            <div key={i} className="console-line">
              <span className="console-time">[{log.time}]</span>
              <span className="console-drone" style={{ color: '#60a5fa' }}>{log.drone}</span>
              <span className="console-sep"> › </span>
              <span className="console-msg" style={{ color }}>
                {log.type === 'nominal' ? '✓' :
                 log.type === 'vibration_fault' ? '⚠' :
                 log.type === 'sensor_drift'    ? '◈' :
                 log.type === 'gps_fault'       ? '⊗' :
                 log.type === 'battery_fault'   ? '⚡' :
                 log.type === 'cmd'             ? '▶' : '•'} {log.message}
              </span>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
