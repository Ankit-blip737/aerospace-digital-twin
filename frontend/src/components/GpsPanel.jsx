import React, { useRef, useEffect } from 'react';

const SCALE = 80000; // pixels per degree

function latLngToCanvas(lat, lng, originLat, originLng, w, h) {
  const x = (lng - originLng) * SCALE + w / 2;
  const y = (lat - originLat) * -SCALE + h / 2;
  return { x, y };
}

const DRONE_COLORS = {
  'UAV-ALPHA-01':   '#60a5fa',
  'UAV-BETA-02':    '#34d399',
  'UAV-GAMMA-03':   '#a78bfa',
  'UAV-DELTA-04':   '#fb923c',
  'UAV-EPSILON-05': '#f472b6',
};

const FAULT_COLORS = {
  nominal:         '#10b981',
  vibration_fault: '#f97316',
  sensor_drift:    '#a855f7',
  gps_fault:       '#eab308',
  battery_fault:   '#ef4444',
};

export default function GpsPanel({ droneData, selectedDrone }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Background
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 30) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = 0; gy < H; gy += 30) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // Origin crosshair
    ctx.strokeStyle = '#1e3a5f';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

    // Range rings
    [40, 80, 120].forEach(r => {
      ctx.beginPath();
      ctx.arc(W / 2, H / 2, r, 0, Math.PI * 2);
      ctx.strokeStyle = '#1e3a5f';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });

    // Determine dynamic origin (follow selected drone or default to center)
    let originLat = -35.3632838;
    let originLng = 149.163061;
    if (selectedDrone && droneData[selectedDrone]?.metrics?.gps) {
        originLat = droneData[selectedDrone].metrics.gps.lat || originLat;
        originLng = droneData[selectedDrone].metrics.gps.lng || originLng;
    } else {
        // Fallback to average or just first drone
        const firstDrone = Object.values(droneData || {})[0];
        if (firstDrone?.metrics?.gps) {
            originLat = firstDrone.metrics.gps.lat || originLat;
            originLng = firstDrone.metrics.gps.lng || originLng;
        }
    }

    // Draw each drone
    Object.entries(droneData || {}).forEach(([droneId, data]) => {
      const gps = data?.metrics?.gps;
      if (!gps || !gps.lat || !gps.lng) return;
      const { x, y } = latLngToCanvas(gps.lat, gps.lng, originLat, originLng, W, H);
      const status = data?.ml_prediction?.status || 'nominal';
      const fillColor = FAULT_COLORS[status] || '#10b981';
      const droneColor = DRONE_COLORS[droneId] || '#fff';

      // Glow
      ctx.shadowColor = fillColor;
      ctx.shadowBlur = 12;

      // Drone dot
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Direction indicator (yaw)
      const yawRad = ((gps.yaw || 0) * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.sin(yawRad) * 14, y - Math.cos(yawRad) * 14);
      ctx.strokeStyle = droneColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = 'bold 9px monospace';
      ctx.fillStyle = droneColor;
      ctx.fillText(droneId.split('-').pop(), x + 10, y + 4);
    });

    // GPS signal quality display
    ctx.font = '9px monospace';
    ctx.fillStyle = '#475569';
    ctx.fillText('GPS PLOT', 8, 14);
    ctx.fillText(`ORIGIN: ${originLat.toFixed(4)}, ${originLng.toFixed(4)}`, 8, 26);

  }, [droneData, selectedDrone]);

  // Satellite status summary
  const droneList = Object.entries(droneData || {});

  return (
    <div className="gps-panel">
      <div className="panel-header-row">
        <span className="panel-icon">📡</span>
        <span className="panel-title">GPS / NAVIGATION MAP</span>
      </div>

      <canvas
        ref={canvasRef}
        width={260}
        height={200}
        className="gps-canvas"
      />

      <div className="gps-stats">
        {droneList.map(([id, data]) => {
          const gps = data?.metrics?.gps;
          const sats = gps?.nSats || 0;
          const hdop = gps?.hdop || 99;
          const status = data?.ml_prediction?.status || 'nominal';
          return (
            <div key={id} className="gps-stat-row">
              <span className="gps-drone-label" style={{ color: DRONE_COLORS[id] }}>
                {id.split('-')[1]}
              </span>
              <span className="gps-sats">
                {Array.from({ length: 10 }, (_, i) => (
                  <span
                    key={i}
                    className="gps-sat-bar"
                    style={{ background: i < sats ? (status === 'gps_fault' ? '#eab308' : '#10b981') : '#1e293b' }}
                  />
                ))}
              </span>
              <span className="gps-hdop">{sats}sat HDOP:{hdop.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
