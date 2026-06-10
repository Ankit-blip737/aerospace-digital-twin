import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Sky, Grid, RoundedBox, Html, Stars, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

const FAULT_CHASSIS = {
  nominal:         { chassis: '#e2e8f0', led: '#10b981', emissive: 0.6 },
  vibration_fault: { chassis: '#f97316', led: '#fbbf24', emissive: 1.2 },
  sensor_drift:    { chassis: '#a855f7', led: '#c084fc', emissive: 1.0 },
  gps_fault:       { chassis: '#eab308', led: '#fde047', emissive: 1.0 },
  battery_fault:   { chassis: '#ef4444', led: '#dc2626', emissive: 1.4 },
};

const CurvedRectangularDrone = ({ id, index, isSelected, dronePayload }) => {
  const droneRef  = useRef();
  const m1 = useRef();
  const m2 = useRef();
  const m3 = useRef();
  const m4 = useRef();
  const lightRef = useRef();

  const kinematics  = useRef({ pitch: 0, roll: 0, yaw: 0 });
  const propulsion  = useRef({ motor1: 4500, motor2: 4500, motor3: 4500, motor4: 4500 });
  const statusRef   = useRef('nominal');
  const modeRef     = useRef('patrol');
  const rtlRef      = useRef(0);
  const baseYRef    = useRef(null);

  useEffect(() => {
    if (!dronePayload) return;
    const m = dronePayload.metrics;
    if (m?.kinematics) kinematics.current = m.kinematics;
    if (m?.propulsion) propulsion.current = m.propulsion;
    if (dronePayload.ml_prediction?.status) statusRef.current = dronePayload.ml_prediction.status;
    modeRef.current = dronePayload.mode || 'patrol';
    rtlRef.current  = dronePayload.rtlProgress || 0;
  }, [dronePayload]);

  const [currentStatus, setCurrentStatus] = useState('nominal');

  useEffect(() => {
    if (dronePayload?.ml_prediction?.status) {
      setCurrentStatus(dronePayload.ml_prediction.status);
    }
  }, [dronePayload?.ml_prediction?.status]);

  const { chassis, led, emissive } = FAULT_CHASSIS[currentStatus] || FAULT_CHASSIS.nominal;
  const scale = isSelected ? 1.15 : 0.9;

  useFrame((state, delta) => {
    if (!droneRef.current) return;
    const time = state.clock.elapsedTime;

    // Apply physics reactions for faults safely
    let dropOffset = 0;
    let wobbleX = 0;
    let wobbleZ = 0;

    // Set base Y anchor (flying over the city)
    if (baseYRef.current === null) baseYRef.current = 15.0 + index * 0.05;
    const baseY = baseYRef.current;

    const mode = modeRef.current;

    if (mode === 'rtl') {
      // Smoothly return toward origin Y=0 (landing)
      const progress = rtlRef.current;
      droneRef.current.position.y = THREE.MathUtils.lerp(baseY, 0.2, progress);
    } else if (mode === 'landed') {
      droneRef.current.position.y = 0.2;
    } else {
      if (currentStatus !== 'nominal') {
        const health = parseFloat(dronePayload?.ml_prediction?.health_index) || 100;
        if (!isNaN(health) && health < 80) {
          dropOffset = -Math.max(0, (80 - health) * 0.15);
        }
        
        const severity = isNaN(health) ? 0 : Math.max(0, (100 - health) * 0.008);
        wobbleX = Math.sin(time * 20 + index) * severity;
        wobbleZ = Math.cos(time * 25 + index) * severity;
      }

      // Hover effect
      const hover = Math.sin(time * 2 + index * 1.3) * 0.1;
      droneRef.current.position.y = baseY + hover + dropOffset;
      
      // Follow macro position from server (which includes grid travel + orbit)
      const targetX = dronePayload?.metrics?.localPos?.x || 0;
      const targetZ = dronePayload?.metrics?.localPos?.z || 0;
      
      droneRef.current.position.x = THREE.MathUtils.lerp(droneRef.current.position.x, targetX, 0.1);
      droneRef.current.position.z = THREE.MathUtils.lerp(droneRef.current.position.z, targetZ, 0.1);
      
      // Face forward along the orbit tangent (handled by server yaw now, but we keep this if needed, 
      // actually server yaw already includes orbit tangent, so base yaw can be 0)
      kinematics.current.baseYaw = 0;
    }

    // Kinematics + Safe Wobble
    const kin = kinematics.current;
    const targetPitch = THREE.MathUtils.degToRad(Number(kin.pitch || 0)) + (wobbleX || 0);
    const targetRoll  = THREE.MathUtils.degToRad(Number(kin.roll  || 0)) + (wobbleZ || 0);
    
    // Add base yaw (orbit tangent) to actual yaw from telemetry
    const targetYaw   = THREE.MathUtils.degToRad(-Number(kin.yaw || 0) - Number(kin.baseYaw || 0));

    // Vibration fault jitter
    const isVibe = currentStatus === 'vibration_fault';
    const jitter = isVibe ? (Math.random() - 0.5) * 0.04 : 0;

    droneRef.current.rotation.x = THREE.MathUtils.lerp(droneRef.current.rotation.x, targetPitch + jitter, 0.2);
    droneRef.current.rotation.z = THREE.MathUtils.lerp(droneRef.current.rotation.z, -targetRoll + jitter, 0.2);
    droneRef.current.rotation.y = THREE.MathUtils.lerp(droneRef.current.rotation.y, targetYaw, 0.2);

    // Motor spin — slow down for battery fault, stop for landed
    const prop = propulsion.current;
    const baseSpd = mode === 'landed' ? 0 : currentStatus === 'battery_fault' ? 4.0 : 15.0;

    const rpm1v = (Number(prop.motor1 || 0) / 8000);
    const rpm2v = (Number(prop.motor2 || 0) / 8000);
    const rpm3v = (Number(prop.motor3 || 0) / 8000);
    const rpm4v = (Number(prop.motor4 || 0) / 8000);

    if (m1.current) m1.current.rotation.y = (m1.current.rotation.y + (baseSpd + rpm1v) * delta) % (Math.PI * 2);
    if (m2.current) m2.current.rotation.y = (m2.current.rotation.y + (baseSpd + rpm2v) * delta) % (Math.PI * 2);
    if (m3.current) m3.current.rotation.y = (m3.current.rotation.y - (baseSpd + rpm3v) * delta) % (Math.PI * 2);
    if (m4.current) m4.current.rotation.y = (m4.current.rotation.y - (baseSpd + rpm4v) * delta) % (Math.PI * 2);

    // Strobe light logic
    if (lightRef.current) {
      lightRef.current.intensity = Math.sin(time * 20) > 0 ? 3 : 0;
    }
  });

  return (
    <group ref={droneRef} scale={scale}>
      {/* Label */}
      <Html position={[0, 0.9, 0]} center zIndexRange={[100, 0]}>
        <div style={{
          padding: '2px 8px',
          borderRadius: '4px',
          fontSize: '9px',
          fontFamily: 'monospace',
          fontWeight: 'bold',
          border: `1px solid ${led}`,
          background: `${led}22`,
          color: '#fff',
          whiteSpace: 'nowrap',
          boxShadow: `0 0 8px ${led}88`,
          backdropFilter: 'blur(4px)',
        }}>
          {id}
        </div>
      </Html>

      {/* Chassis */}
      <RoundedBox args={[0.7, 0.25, 1.2]} radius={0.12} smoothness={4} position={[0, 0.1, 0.1]}>
        <meshStandardMaterial color={chassis} metalness={0.3} roughness={0.4} emissive={chassis} emissiveIntensity={currentStatus !== 'nominal' ? 0.1 : 0} />
      </RoundedBox>
      <RoundedBox args={[0.5, 0.2, 1.0]} radius={0.08} smoothness={4} position={[0, -0.05, 0.15]}>
        <meshStandardMaterial color="#334155" roughness={0.6} />
      </RoundedBox>

      {/* Safe Warning Strobe Light */}
      {currentStatus !== 'nominal' && (
        <pointLight 
          ref={lightRef}
          position={[0, 0.3, 0]} 
          color={currentStatus === 'battery_fault' ? '#ef4444' : currentStatus === 'vibration_fault' ? '#fbbf24' : '#a855f7'}
          distance={8}
        />
      )}

      {/* Camera */}
      <group position={[0, -0.1, -0.55]}>
        <mesh><boxGeometry args={[0.15, 0.15, 0.15]} /><meshStandardMaterial color="#475569" /></mesh>
        <mesh position={[0, -0.05, -0.1]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.1, 0.1, 0.18, 32]} />
          <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.1} />
        </mesh>
        <mesh position={[0, -0.05, -0.19]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.02, 16]} />
          <meshStandardMaterial color={led} emissive={led} emissiveIntensity={emissive} />
        </mesh>
      </group>

      {/* Arms */}
      {[{ x: 0.6, z: -0.6 }, { x: -0.6, z: -0.6 }, { x: 0.6, z: 0.6 }, { x: -0.6, z: 0.6 }].map((arm, i) => (
        <group key={i} position={[arm.x, 0.05, arm.z]}>
          <mesh rotation={[Math.PI / 2, 0, Math.atan2(arm.z, arm.x) + Math.PI / 2]}>
            <cylinderGeometry args={[0.04, 0.04, 1.2, 16]} />
            <meshStandardMaterial color="#64748b" />
          </mesh>
          <mesh position={[0, -0.08, 0]}>
            <sphereGeometry args={[0.03]} />
            <meshStandardMaterial
              color={i < 2 ? '#ef4444' : led}
              emissive={i < 2 ? '#ef4444' : led}
              emissiveIntensity={0.8}
            />
          </mesh>
        </group>
      ))}

      {/* Motors & Props */}
      {[{ ref: m1, x: 0.85, z: -0.85 }, { ref: m3, x: -0.85, z: -0.85 }, { ref: m2, x: -0.85, z: 0.85 }, { ref: m4, x: 0.85, z: 0.85 }].map((m, i) => (
        <group key={i} position={[m.x, 0.15, m.z]}>
          <mesh position={[0, -0.05, 0]}>
            <cylinderGeometry args={[0.1, 0.12, 0.15, 32]} />
            <meshStandardMaterial color="#94a3b8" metalness={0.8} roughness={0.2} />
          </mesh>
          <group ref={m.ref}>
            <mesh><boxGeometry args={[1.4, 0.015, 0.1]} /><meshStandardMaterial color="#1e3a8a" /></mesh>
            <mesh position={[0, 0.01, 0]}>
              <cylinderGeometry args={[0.7, 0.7, 0.01, 32]} />
              <meshBasicMaterial color="#0f172a" transparent opacity={0.2} />
            </mesh>
          </group>
        </group>
      ))}
    </group>
  );
};

const CameraController = ({ selectedDrone, allDroneData }) => {
  const { camera, controls } = useThree();
  useFrame(() => {
    if (selectedDrone && allDroneData?.[selectedDrone] && controls) {
      const pos = allDroneData[selectedDrone].metrics?.localPos;
      if (pos) {
        // Track the actual drone height
        controls.target.lerp(new THREE.Vector3(pos.x, 15, pos.z), 0.05);
        controls.update();
      }
    }
  });
  return null;
};

// Procedural City to provide visual reference for speed and movement
const CityEnvironment = () => {
  const buildings = React.useMemo(() => {
    const items = [];
    // Generating 400 buildings as requested
    for (let i = 0; i < 400; i++) {
      // Scatter over 800x800 area
      const x = (Math.random() - 0.5) * 800;
      const z = (Math.random() - 0.5) * 800;
      // Leave a small clearing around origin
      if (Math.abs(x) < 20 && Math.abs(z) < 20) continue;
      
      const width = 2 + Math.random() * 8;
      const depth = 2 + Math.random() * 8;
      const height = 2 + Math.random() * 10;
      items.push({ position: [x, height / 2, z], scale: [width, height, depth] });
    }
    return items;
  }, []);

  return (
    <group>
      {buildings.map((props, i) => (
        <mesh key={i} position={props.position} scale={props.scale} castShadow receiveShadow>
          <boxGeometry />
          <meshStandardMaterial color="#0f172a" roughness={0.9} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
};

export default function DroneViewer({ swarmNodes, selectedDrone, allDroneData }) {
  const nodes = swarmNodes || [];

  return (
    <Canvas camera={{ position: [0, 22, 20], fov: 50 }} shadows>
      <fog attach="fog" args={['#0f172a', 30, 250]} />
      <Environment preset="sunset" background blur={0.8} />
      
      <ambientLight intensity={1.5} />
      <directionalLight position={[50, 50, 20]} intensity={2.5} castShadow />
      
      {/* Realistic ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
        <planeGeometry args={[2000, 2000]} />
        <meshStandardMaterial color="#020617" roughness={1} metalness={0} />
      </mesh>
      <Grid infiniteGrid fadeDistance={250} sectionColor="#1e3a8a" cellColor="#0f172a" sectionSize={20} cellSize={4} />
      
      <CityEnvironment />
      
      {/* Contact Shadows for realism - baked once to save FPS */}
      <ContactShadows position={[0, 0, 0]} scale={50} resolution={256} far={30} blur={2} opacity={0.6} color="#000000" frames={1} />


      {nodes.map((node, index) => {
        return (
          <group key={node.id} position={[0, 0, 0]}>
            <CurvedRectangularDrone
              id={node.id}
              index={index}
              isSelected={node.id === selectedDrone}
              dronePayload={allDroneData?.[node.id] || null}
            />
          </group>
        );
      })}

      <OrbitControls makeDefault enableZoom={true} minDistance={3} maxDistance={28} maxPolarAngle={Math.PI / 2 - 0.05} />
      <CameraController selectedDrone={selectedDrone} allDroneData={allDroneData} />
    </Canvas>
  );
}