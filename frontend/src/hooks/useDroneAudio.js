import { useEffect, useRef, useState } from 'react';

export function useDroneAudio(isActive, intensity = 1.0) {
  const audioCtxRef = useRef(null);
  const oscillatorsRef = useRef([]);
  const gainNodeRef = useRef(null);
  const noiseNodeRef = useRef(null);
  const filterRef = useRef(null);
  const [userInteracted, setUserInteracted] = useState(false);

  useEffect(() => {
    const handleInteraction = () => {
      try {
        if (!audioCtxRef.current) {
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          if (AudioContext) {
            audioCtxRef.current = new AudioContext();
          }
        }
        if (audioCtxRef.current && audioCtxRef.current.state === 'suspended') {
          audioCtxRef.current.resume().catch(e => console.warn(e));
        }
        setUserInteracted(true);
      } catch (err) {
        console.warn('AudioContext init failed:', err);
      }
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('keydown', handleInteraction);
    };

    window.addEventListener('click', handleInteraction);
    window.addEventListener('keydown', handleInteraction);
    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('keydown', handleInteraction);
    };
  }, []);

  useEffect(() => {
    if (!userInteracted || !audioCtxRef.current) return;

    try {
      if (isActive) {
        const ctx = audioCtxRef.current;
        
        if (!gainNodeRef.current) {
          gainNodeRef.current = ctx.createGain();
          gainNodeRef.current.gain.value = 0; 
          gainNodeRef.current.connect(ctx.destination);

          // Create 4 oscillators to simulate quadcopter motors
          const baseFreq = 180;
          const detunes = [0, 4, -4, 8];
          
          oscillatorsRef.current = detunes.map(detune => {
            const osc = ctx.createOscillator();
            osc.type = 'sawtooth'; // Gives a buzzy, motor-like sound
            osc.frequency.value = baseFreq + detune;
            osc.connect(gainNodeRef.current);
            osc.start();
            return osc;
          });

          // Add wind/blade chopping noise using a white noise buffer
          const bufferSize = ctx.sampleRate * 2;
          const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
          const output = noiseBuffer.getChannelData(0);
          for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
          }
          noiseNodeRef.current = ctx.createBufferSource();
          noiseNodeRef.current.buffer = noiseBuffer;
          noiseNodeRef.current.loop = true;

          // Lowpass filter to muffle the harsh white noise into a "whoosh"
          filterRef.current = ctx.createBiquadFilter();
          filterRef.current.type = 'lowpass';
          filterRef.current.frequency.value = 600;
          
          noiseNodeRef.current.connect(filterRef.current);
          filterRef.current.connect(gainNodeRef.current);
          noiseNodeRef.current.start();
        }

        const safeIntensity = Math.max(0.1, Math.min(intensity, 2.0));
        
        // Target volume based on intensity (keep it subtle so it's not annoying)
        gainNodeRef.current.gain.setTargetAtTime(0.04 * safeIntensity, ctx.currentTime, 0.5);

        // Adjust motor pitch
        const baseFreq = 160 * safeIntensity;
        oscillatorsRef.current.forEach((osc, i) => {
          const detunes = [0, 4, -4, 8];
          osc.frequency.setTargetAtTime(baseFreq + detunes[i], ctx.currentTime, 0.2);
        });
        
        // Adjust wind filter
        if (filterRef.current) {
           filterRef.current.frequency.setTargetAtTime(500 + (300 * safeIntensity), ctx.currentTime, 0.2);
        }

      } else {
        // Fade out smoothly
        if (gainNodeRef.current && audioCtxRef.current) {
          gainNodeRef.current.gain.setTargetAtTime(0, audioCtxRef.current.currentTime, 0.5);
        }
      }
    } catch (err) {
      console.warn('Drone audio error:', err);
    }
  }, [isActive, intensity, userInteracted]);
}
