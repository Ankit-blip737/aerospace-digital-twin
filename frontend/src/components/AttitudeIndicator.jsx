import React from 'react';

export default function AttitudeIndicator({ kinematics }) {
  const pitch = kinematics?.pitch || 0;
  const roll  = kinematics?.roll  || 0;
  const yaw   = kinematics?.yaw   || 0;

  // Clamp values for visual display
  const clampedPitch = Math.max(-45, Math.min(45, pitch));
  const clampedRoll  = Math.max(-60, Math.min(60, roll));

  // Pitch offset: each degree = 2px of vertical shift
  const pitchOffset = clampedPitch * 2;

  return (
    <div className="attitude-container">
      {/* Artificial Horizon */}
      <div className="attitude-inner">
        <p className="panel-label">ATTITUDE INDICATOR</p>

        <div className="adi-viewport">
          {/* Horizon ball rotates with roll and shifts with pitch */}
          <div
            className="adi-ball"
            style={{
              transform: `rotate(${-clampedRoll}deg) translateY(${pitchOffset}px)`,
              transition: 'transform 0.15s ease-out',
            }}
          >
            {/* Sky */}
            <div className="adi-sky" />
            {/* Ground */}
            <div className="adi-ground" />
            {/* Pitch ladder lines */}
            {[-20, -10, 10, 20].map(deg => (
              <div
                key={deg}
                className="adi-pitch-line"
                style={{ top: `calc(50% + ${-deg * 2}px - 1px)` }}
              >
                <span className="adi-pitch-label left">{Math.abs(deg)}</span>
                <div className="adi-pitch-bar" />
                <span className="adi-pitch-label right">{Math.abs(deg)}</span>
              </div>
            ))}
            {/* Horizon line */}
            <div className="adi-horizon-line" />
          </div>

          {/* Fixed aircraft reference */}
          <div className="adi-aircraft">
            <div className="adi-wing-left" />
            <div className="adi-center-dot" />
            <div className="adi-wing-right" />
          </div>

          {/* Roll arc indicator (top) */}
          <div className="adi-roll-arc">
            {[-30, -20, -10, 0, 10, 20, 30].map(deg => {
              const rad = ((deg - 90) * Math.PI) / 180;
              const r = 70;
              const x = 50 + r * Math.cos(rad);
              const y = r + r * Math.sin(rad);
              return (
                <div
                  key={deg}
                  className={`adi-roll-tick ${deg === 0 ? 'adi-roll-tick-center' : ''}`}
                  style={{ left: `${x}%`, top: `${y * 0.55}%`, transform: `rotate(${deg}deg)` }}
                />
              );
            })}
          </div>

          {/* Roll indicator pointer */}
          <div
            className="adi-roll-pointer"
            style={{ transform: `translateX(-50%) rotate(${-clampedRoll}deg)` }}
          />
        </div>

        {/* Digital readouts */}
        <div className="adi-readouts">
          <div className="adi-readout">
            <span className="adi-readout-label">PITCH</span>
            <span className="adi-readout-value">{pitch.toFixed(1)}°</span>
          </div>
          <div className="adi-readout">
            <span className="adi-readout-label">ROLL</span>
            <span className="adi-readout-value">{roll.toFixed(1)}°</span>
          </div>
          <div className="adi-readout">
            <span className="adi-readout-label">YAW</span>
            <span className="adi-readout-value">{yaw.toFixed(1)}°</span>
          </div>
        </div>

        {/* Compass Rose */}
        <div className="compass-container">
          <p className="panel-label" style={{ marginBottom: '0.5rem' }}>COMPASS</p>
          <div className="compass-rose">
            <div
              className="compass-needle"
              style={{
                transform: `translate(-50%, -100%) rotate(${yaw}deg)`,
                transformOrigin: '50% 100%',
                transition: 'transform 0.2s ease-out',
              }}
            />
            <div className="compass-center-dot" />
            {['N', 'E', 'S', 'W'].map((dir, i) => (
              <span
                key={dir}
                className="compass-dir"
                style={{
                  transform: `rotate(${i * 90}deg) translateY(-38px) rotate(${-i * 90}deg)`,
                }}
              >
                {dir}
              </span>
            ))}
            <div className="compass-ring" style={{ transform: `rotate(${-yaw}deg)`, transition: 'transform 0.2s ease-out' }}>
              {Array.from({ length: 36 }, (_, i) => (
                <div
                  key={i}
                  className={`compass-tick ${i % 9 === 0 ? 'compass-tick-major' : ''}`}
                  style={{ transform: `rotate(${i * 10}deg) translateY(-44px)` }}
                />
              ))}
            </div>
          </div>
          <p className="compass-heading">{Math.round((yaw + 360) % 360).toString().padStart(3, '0')}°</p>
        </div>
      </div>
    </div>
  );
}
