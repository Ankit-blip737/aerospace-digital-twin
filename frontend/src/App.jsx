import React, { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import DroneViewer from './DroneModel';
import HealthChart from './components/HealthChart';
import AttitudeIndicator from './components/AttitudeIndicator';
import CommandConsole from './components/CommandConsole';
import CommandPanel from './components/CommandPanel';
import GpsPanel from './components/GpsPanel';
import RULPanel from './components/RULPanel';
import { useAudioAlarm } from './hooks/useAudioAlarm';
import { useDroneAudio } from './hooks/useDroneAudio';
import './App.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';
const socket = io(BACKEND_URL);

const SWARM_NODES = [
  { id: 'UAV-ALPHA-01' },
  { id: 'UAV-BETA-02' },
  { id: 'UAV-GAMMA-03' },
  { id: 'UAV-DELTA-04' },
  { id: 'UAV-EPSILON-05' },
];

const HISTORY_LEN = 100;
const MAX_LOGS = 150;

const STATUS_CONFIG = {
  nominal:         { label: 'NOMINAL',           color: '#10b981', bg: '#052e16' },
  vibration_fault: { label: 'VIBRATION FAULT',   color: '#f97316', bg: '#431407' },
  sensor_drift:    { label: 'SENSOR DRIFT',       color: '#a855f7', bg: '#2e1065' },
  gps_fault:       { label: 'GPS / COMMS FAULT', color: '#eab308', bg: '#422006' },
  battery_fault:   { label: 'BATTERY FAULT',     color: '#ef4444', bg: '#450a0a' },
};

function now() {
  return new Date().toLocaleTimeString('en-US', { hour12: false });
}

export default function App() {
  const [allDroneData,  setAllDroneData]  = useState({});
  const [selectedDrone, setSelectedDrone] = useState(SWARM_NODES[0].id);
  const [isLive,        setIsLive]        = useState(true);
  const [logs,          setLogs]          = useState([]);
  const [activeTab,     setActiveTab]     = useState('charts'); // charts | attitude | gps | command

  // Rolling histories per drone
  const healthHist = useRef({});
  const voltHist   = useRef({});
  const vibeHist   = useRef({});
  const lastStatus = useRef({});

  const pushLog = useCallback((drone, message, type = 'system') => {
    setLogs(prev => {
      const newLog = { time: now(), drone, message, type };
      const next = [...prev, newLog];
      return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
    });
  }, []);

  useEffect(() => {
    pushLog('SYSTEM', 'Command Center online. Awaiting telemetry...', 'system');

    socket.on('connect', () => {
      pushLog('SYSTEM', `Socket connected: ${socket.id}`, 'system');
    });

    socket.on('telemetry_stream', (raw) => {
      if (!isLive) return;
      try {
        const data = JSON.parse(raw);
        const id   = data.deviceId;
        if (!id) return;

        setAllDroneData(prev => ({ ...prev, [id]: data }));

        // Update rolling histories
        if (!healthHist.current[id]) healthHist.current[id] = [];
        if (!voltHist.current[id])   voltHist.current[id]   = [];
        if (!vibeHist.current[id])   vibeHist.current[id]   = [];

        const health = data.ml_prediction?.health_index;
        const volt   = data.metrics?.power?.battV;
        const vibe   = data.metrics?.vibration
          ? Math.sqrt(
              (data.metrics.vibration.vibeX || 0) ** 2 +
              (data.metrics.vibration.vibeY || 0) ** 2 +
              (data.metrics.vibration.vibeZ || 0) ** 2
            )
          : null;

        if (health !== undefined) {
          healthHist.current[id].push(health);
          if (healthHist.current[id].length > HISTORY_LEN) healthHist.current[id].shift();
        }
        if (volt !== undefined) {
          voltHist.current[id].push(volt);
          if (voltHist.current[id].length > HISTORY_LEN) voltHist.current[id].shift();
        }
        if (vibe !== null) {
          vibeHist.current[id].push(parseFloat(vibe.toFixed(4)));
          if (vibeHist.current[id].length > HISTORY_LEN) vibeHist.current[id].shift();
        }

        // Log state transitions
        const newStatus = data.ml_prediction?.status;
        if (newStatus && newStatus !== lastStatus.current[id]) {
          lastStatus.current[id] = newStatus;
          const cfg = STATUS_CONFIG[newStatus];
          if (newStatus !== 'nominal') {
            pushLog(id, `${cfg?.label || newStatus} detected | Health: ${data.ml_prediction?.health_index}% | ${data.ml_prediction?.confidence}`, newStatus);
          } else if (Object.keys(lastStatus.current).length > 0) {
            pushLog(id, 'Returned to NOMINAL operation.', 'nominal');
          }
        }

      } catch { /* ignore malformed packets */ }
    });

    socket.on('command_ack', ({ droneId, message, action }) => {
      pushLog(droneId, message, 'cmd');
    });

    return () => {
      socket.off('telemetry_stream');
      socket.off('command_ack');
      socket.off('connect');
    };
  }, [isLive, pushLog]);

  // Command handlers
  const handleInjectFault = (droneId, faultClass) => {
    socket.emit('inject_fault', { droneId, faultClass });
  };
  const handleReset = (droneId) => {
    socket.emit('reset_drone', { droneId });
  };
  const handleRTL = (droneId) => {
    socket.emit('command_rtl', { droneId });
    pushLog(droneId, 'RTL command issued — returning to launch site.', 'cmd');
  };

  // Safe Audio Alarm Logic
  const hasFault = Object.values(allDroneData).some(
    d => d?.ml_prediction?.status && d.ml_prediction.status !== 'nominal'
  );
  useAudioAlarm(hasFault);

  const selectedData = allDroneData[selectedDrone];
  const selectedML   = selectedData?.ml_prediction;
  const statusCfg    = STATUS_CONFIG[selectedML?.status || 'nominal'];

  // Ambient Drone Propeller Audio
  const isFlying = selectedData?.mode && selectedData.mode !== 'landed';
  let flightIntensity = 1.0;
  if (selectedML?.status === 'battery_fault') flightIntensity = 0.6; // Struggles, RPM drops
  if (selectedML?.status === 'vibration_fault') flightIntensity = 1.4; // Revs aggressively
  useDroneAudio(isFlying, flightIntensity);

  return (
    <div className="app-root">

      {/* ── TOP HEADER ── */}
      <header className="app-header">
        <div className="header-brand">
          <div className="header-logo">⬡</div>
          <div>
            <h1 className="header-title">Aerospace Digital Twin</h1>
            <p className="header-subtitle">MANET Swarm Command Center · 5 UAVs Online</p>
          </div>
        </div>

        <div className="header-status-bar">
          {SWARM_NODES.map(node => {
            const nd = allDroneData[node.id];
            const s  = nd?.ml_prediction?.status || 'nominal';
            const cfg = STATUS_CONFIG[s];
            return (
              <button
                key={node.id}
                className={`swarm-node-btn ${node.id === selectedDrone ? 'swarm-node-btn-selected' : ''}`}
                style={{ '--s-color': cfg.color }}
                onClick={() => setSelectedDrone(node.id)}
              >
                <span className="swarm-node-dot" style={{ background: cfg.color, boxShadow: `0 0 6px ${cfg.color}` }} />
                <span className="swarm-node-id">{node.id.split('-')[1]}</span>
              </button>
            );
          })}
        </div>

        <div className="header-controls">
          <div
            className="status-badge"
            style={{ color: statusCfg.color, borderColor: statusCfg.color, background: statusCfg.bg + '88' }}
          >
            {statusCfg.label}
          </div>
          <button
            className={`live-btn ${isLive ? 'live-btn-active' : 'live-btn-paused'}`}
            onClick={() => setIsLive(v => !v)}
          >
            {isLive ? '⏸ PAUSE' : '▶ RESUME'}
          </button>
        </div>
      </header>

      {/* ── 3D CANVAS ── */}
      <div className="canvas-section">
        <DroneViewer
          swarmNodes={SWARM_NODES}
          selectedDrone={selectedDrone}
          allDroneData={allDroneData}
        />

        {/* Overlay HUD */}
        <div className="hud-overlay hud-top-left">
          <div className="hud-chip">
            <span className="hud-dot" style={{ background: selectedML?.status_color || '#10b981' }} />
            {selectedDrone}
          </div>
          <div className="hud-chip hud-metric">
            Health <strong style={{ color: selectedML?.status_color || '#10b981' }}>
              {selectedML?.health_index?.toFixed(1) ?? '--'}%
            </strong>
          </div>
          <div className="hud-chip hud-metric">
            Mode <strong>{selectedData?.mode?.toUpperCase() ?? 'PATROL'}</strong>
          </div>
        </div>

        <div className="hud-overlay hud-top-right">
          <div className="hud-chip hud-metric">
            Roll <strong>{selectedData?.metrics?.kinematics?.roll?.toFixed(2) ?? '--'}°</strong>
          </div>
          <div className="hud-chip hud-metric">
            Pitch <strong>{selectedData?.metrics?.kinematics?.pitch?.toFixed(2) ?? '--'}°</strong>
          </div>
          <div className="hud-chip hud-metric">
            Yaw <strong>{selectedData?.metrics?.kinematics?.yaw?.toFixed(2) ?? '--'}°</strong>
          </div>
        </div>

        <div className="hud-overlay hud-bottom-left">
          <div className="hud-chip hud-metric">
            Bat <strong style={{ color: '#f59e0b' }}>{selectedData?.metrics?.power?.battV?.toFixed(2) ?? '--'}V</strong>
          </div>
          <div className="hud-chip hud-metric">
            {selectedData?.metrics?.power?.remPct?.toFixed(0) ?? '--'}%&nbsp;
            <span style={{ color: '#64748b' }}>remaining</span>
          </div>
          <div className="hud-chip hud-metric">
            Sats <strong>{selectedData?.metrics?.gps?.nSats ?? '--'}</strong>
          </div>
        </div>

        <div className="hud-overlay hud-bottom-right">
          <div className="hud-chip hud-metric">
            Vibe <strong style={{ color: '#a855f7' }}>
              {selectedData?.metrics?.vibration
                ? Math.sqrt(
                    (selectedData.metrics.vibration.vibeX || 0) ** 2 +
                    (selectedData.metrics.vibration.vibeY || 0) ** 2 +
                    (selectedData.metrics.vibration.vibeZ || 0) ** 2
                  ).toFixed(3)
                : '--'} m/s²
            </strong>
          </div>
          <div className="hud-chip hud-metric">
            Conf <strong>{selectedML?.confidence ?? '--'}</strong>
          </div>
        </div>
      </div>

      {/* ── DASHBOARD BELOW ── */}
      <div className="dashboard-section">
        <div className="dashboard-grid">

          {/* LEFT: RUL Panel (full height) */}
          <div className="dash-col dash-col-left">
            <RULPanel droneData={allDroneData} />
          </div>

          {/* CENTER: Tab Panels */}
          <div className="dash-col dash-col-center">
            {/* Tab bar */}
            <div className="tab-bar">
              {[
                { key: 'charts',   icon: '📊', label: 'Trend Charts' },
                { key: 'attitude', icon: '✈',  label: 'Attitude' },
                { key: 'gps',      icon: '📡', label: 'GPS Map' },
                { key: 'command',  icon: '⌨',  label: 'Commands' },
              ].map(tab => (
                <button
                  key={tab.key}
                  className={`tab-btn ${activeTab === tab.key ? 'tab-btn-active' : ''}`}
                  onClick={() => setActiveTab(tab.key)}
                >
                  {tab.icon} {tab.label}
                </button>
              ))}
            </div>

            <div className="tab-content">
              {activeTab === 'charts' && (
                <HealthChart
                  healthHistory={healthHist.current[selectedDrone] || []}
                  powerHistory={voltHist.current[selectedDrone] || []}
                  vibeHistory={vibeHist.current[selectedDrone] || []}
                />
              )}
              {activeTab === 'attitude' && (
                <AttitudeIndicator kinematics={selectedData?.metrics?.kinematics} />
              )}
              {activeTab === 'gps' && (
                <GpsPanel droneData={allDroneData} selectedDrone={selectedDrone} />
              )}
              {activeTab === 'command' && (
                <CommandPanel
                  selectedDrone={selectedDrone}
                  onInjectFault={handleInjectFault}
                  onReset={handleReset}
                  onRTL={handleRTL}
                />
              )}
            </div>
          </div>

          {/* RIGHT: Telemetry detail + Console */}
          <div className="dash-col dash-col-right">
            {/* Telemetry Detail */}
            <div className="telem-detail-panel">
              <p className="panel-title-sm">⬤ SENSOR DETAIL — {selectedDrone}</p>

              <div className="telem-grid">
                {/* Kinematics */}
                <div className="telem-group">
                  <p className="telem-group-label">Kinematics</p>
                  <div className="telem-row"><span>Pitch</span><span>{selectedData?.metrics?.kinematics?.pitch?.toFixed(2) ?? '--'}°</span></div>
                  <div className="telem-row"><span>Roll</span><span>{selectedData?.metrics?.kinematics?.roll?.toFixed(2) ?? '--'}°</span></div>
                  <div className="telem-row"><span>Yaw</span><span>{selectedData?.metrics?.kinematics?.yaw?.toFixed(2) ?? '--'}°</span></div>
                  <div className="telem-row"><span>ErrRP</span><span>{selectedData?.metrics?.kinematics?.errRP?.toFixed(4) ?? '--'}</span></div>
                </div>

                {/* Power */}
                <div className="telem-group">
                  <p className="telem-group-label">Power</p>
                  <div className="telem-row"><span>Voltage</span><span style={{color:'#f59e0b'}}>{selectedData?.metrics?.power?.battV?.toFixed(3) ?? '--'}V</span></div>
                  <div className="telem-row"><span>Current</span><span>{selectedData?.metrics?.power?.curr?.toFixed(2) ?? '--'}A</span></div>
                  <div className="telem-row"><span>Bat Temp</span><span>{selectedData?.metrics?.power?.batTemp?.toFixed(1) ?? '--'}°C</span></div>
                  <div className="telem-row"><span>Remaining</span><span style={{color:'#10b981'}}>{selectedData?.metrics?.power?.remPct?.toFixed(1) ?? '--'}%</span></div>
                </div>

                {/* Vibration */}
                <div className="telem-group">
                  <p className="telem-group-label">Vibration</p>
                  <div className="telem-row"><span>X</span><span style={{color:'#a855f7'}}>{selectedData?.metrics?.vibration?.vibeX?.toFixed(4) ?? '--'}</span></div>
                  <div className="telem-row"><span>Y</span><span style={{color:'#a855f7'}}>{selectedData?.metrics?.vibration?.vibeY?.toFixed(4) ?? '--'}</span></div>
                  <div className="telem-row"><span>Z</span><span style={{color:'#a855f7'}}>{selectedData?.metrics?.vibration?.vibeZ?.toFixed(4) ?? '--'}</span></div>
                  <div className="telem-row"><span>Clip</span><span>{selectedData?.metrics?.vibration?.vibeClip ?? '--'}</span></div>
                </div>

                {/* Barometer */}
                <div className="telem-group">
                  <p className="telem-group-label">Barometer</p>
                  <div className="telem-row"><span>Alt</span><span>{selectedData?.metrics?.barometer?.alt?.toFixed(1) ?? '--'}m</span></div>
                  <div className="telem-row"><span>Press</span><span>{selectedData?.metrics?.barometer?.press?.toFixed(0) ?? '--'}Pa</span></div>
                  <div className="telem-row"><span>Climb</span><span>{selectedData?.metrics?.barometer?.climbRate?.toFixed(2) ?? '--'}m/s</span></div>
                </div>

                {/* Propulsion */}
                <div className="telem-group">
                  <p className="telem-group-label">Motors (RPM)</p>
                  <div className="telem-row"><span>M1 FR</span><span>{selectedData?.metrics?.propulsion?.motor1 ?? '--'}</span></div>
                  <div className="telem-row"><span>M2 RL</span><span>{selectedData?.metrics?.propulsion?.motor2 ?? '--'}</span></div>
                  <div className="telem-row"><span>M3 FL</span><span>{selectedData?.metrics?.propulsion?.motor3 ?? '--'}</span></div>
                  <div className="telem-row"><span>M4 RR</span><span>{selectedData?.metrics?.propulsion?.motor4 ?? '--'}</span></div>
                </div>

                {/* EKF State */}
                <div className="telem-group">
                  <p className="telem-group-label">EKF State</p>
                  <div className="telem-row"><span>VN</span><span>{selectedData?.metrics?.ekfState?.vn?.toFixed(3) ?? '--'}</span></div>
                  <div className="telem-row"><span>VE</span><span>{selectedData?.metrics?.ekfState?.ve?.toFixed(3) ?? '--'}</span></div>
                  <div className="telem-row"><span>VD</span><span>{selectedData?.metrics?.ekfState?.vd?.toFixed(3) ?? '--'}</span></div>
                  <div className="telem-row"><span>OH</span><span>{selectedData?.metrics?.ekfState?.oh?.toFixed(2) ?? '--'}</span></div>
                </div>
              </div>
            </div>

            {/* Command Console */}
            <CommandConsole logs={logs} />
          </div>
        </div>
      </div>
    </div>
  );
}