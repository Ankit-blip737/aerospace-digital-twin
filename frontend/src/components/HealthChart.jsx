import React, { useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">T-{label}s</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="chart-tooltip-item">
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</strong>
        </p>
      ))}
    </div>
  );
};

export default function HealthChart({ healthHistory, powerHistory, vibeHistory }) {
  const chartData = useMemo(() => {
    const len = Math.max(
      (healthHistory || []).length,
      (powerHistory  || []).length,
      (vibeHistory   || []).length
    );
    return Array.from({ length: len }, (_, i) => ({
      t:      len - i,
      health: healthHistory?.[i]  ?? null,
      volt:   powerHistory?.[i]   ?? null,
      vibe:   vibeHistory?.[i]    ?? null,
    }));
  }, [healthHistory, powerHistory, vibeHistory]);

  return (
    <div className="chart-wrapper">
      {/* Health Index Chart */}
      <div className="chart-section">
        <p className="chart-title">⬤ Node Health Index (%)</p>
        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="t" reversed tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 2" strokeWidth={1} />
            <ReferenceLine y={40} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} />
            <Line
              type="monotone" dataKey="health" name="Health"
              stroke="#10b981" strokeWidth={2} dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Battery Voltage Chart */}
      <div className="chart-section" style={{ marginTop: '1rem' }}>
        <p className="chart-title">⬤ Battery Voltage (V)</p>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="t" reversed tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <YAxis domain={['auto', 'auto']} tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone" dataKey="volt" name="Voltage (V)"
              stroke="#f59e0b" strokeWidth={2} dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Vibration Chart */}
      <div className="chart-section" style={{ marginTop: '1rem' }}>
        <p className="chart-title">⬤ Vibration Magnitude (m/s²)</p>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="t" reversed tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <YAxis domain={[0, 'auto']} tick={{ fill: '#475569', fontSize: 9 }} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={5} stroke="#a855f7" strokeDasharray="4 2" strokeWidth={1} />
            <Line
              type="monotone" dataKey="vibe" name="Vibe (m/s²)"
              stroke="#a855f7" strokeWidth={2} dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
