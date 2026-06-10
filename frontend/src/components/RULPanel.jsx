import React from 'react';

const DRONE_COLORS = {
  'UAV-ALPHA-01':   '#60a5fa',
  'UAV-BETA-02':    '#34d399',
  'UAV-GAMMA-03':   '#a78bfa',
  'UAV-DELTA-04':   '#fb923c',
  'UAV-EPSILON-05': '#f472b6',
};

const STATUS_COLORS = {
  nominal:         '#10b981',
  vibration_fault: '#f97316',
  sensor_drift:    '#a855f7',
  gps_fault:       '#eab308',
  battery_fault:   '#ef4444',
};

function RULBar({ rul, health, status }) {
  const pct = rul !== null && rul !== undefined
    ? Math.min(100, (rul / 60) * 100)
    : health;
  const color = STATUS_COLORS[status] || '#10b981';

  return (
    <div className="rul-bar-track">
      <div
        className="rul-bar-fill"
        style={{
          width:      `${pct}%`,
          background: color,
          boxShadow:  `0 0 6px ${color}80`,
          transition: 'width 0.8s ease-out',
        }}
      />
    </div>
  );
}

export default function RULPanel({ droneData }) {
  const drones = Object.entries(droneData || {});

  return (
    <div className="rul-panel">
      <div className="panel-header-row">
        <span className="panel-icon">⏱</span>
        <span className="panel-title">REMAINING USEFUL LIFE</span>
      </div>

      <div className="rul-list">
        {drones.map(([droneId, data]) => {
          const rul    = data?.rul;
          const ml     = data?.ml_prediction;
          const health = ml?.health_index ?? 100;
          const status = ml?.status || 'nominal';
          const color  = DRONE_COLORS[droneId] || '#fff';
          const statusColor = STATUS_COLORS[status] || '#10b981';

          const rulMin = rul?.rul_minutes;
          const trend  = rul?.trend || 'stable';

          return (
            <div key={droneId} className="rul-row">
              <div className="rul-row-header">
                <span className="rul-drone-id" style={{ color }}>{droneId}</span>
                <div className="rul-status-badges">
                  <span
                    className="rul-status-badge"
                    style={{ color: statusColor, borderColor: statusColor, background: `${statusColor}15` }}
                  >
                    {ml?.status_label || 'NOMINAL'}
                  </span>
                  {rulMin !== null && rulMin !== undefined ? (
                    <span
                      className={`rul-time-badge ${trend === 'critical' ? 'rul-time-critical' : trend === 'warning' ? 'rul-time-warn' : 'rul-time-ok'}`}
                    >
                      {trend === 'critical' ? '⚠' : trend === 'warning' ? '◌' : '✓'}&nbsp;
                      {rulMin >= 60
                        ? `${(rulMin / 60).toFixed(1)}h`
                        : `${rulMin.toFixed(0)}min`}
                    </span>
                  ) : (
                    <span className="rul-time-badge rul-time-ok">∞ Stable</span>
                  )}
                </div>
              </div>

              <RULBar rul={rulMin} health={health} status={status} />

              <div className="rul-detail-row">
                <span className="rul-detail">Health: <strong style={{ color: statusColor }}>{health.toFixed(1)}%</strong></span>
                {rul?.slope !== undefined && (
                  <span className="rul-detail">
                    Trend: <strong style={{ color: rul.slope < -0.01 ? '#ef4444' : '#10b981' }}>
                      {rul.slope < 0 ? '▼' : '▲'} {Math.abs(rul.slope * 240).toFixed(2)}%/min
                    </strong>
                  </span>
                )}
                <span className="rul-detail">Conf: <strong>{ml?.confidence || '–'}</strong></span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
