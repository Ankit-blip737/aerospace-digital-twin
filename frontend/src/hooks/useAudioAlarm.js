import { useEffect, useRef, useState } from 'react';

export function useAudioAlarm(isAlarmActive) {
  const audioCtxRef = useRef(null);
  const oscillatorRef = useRef(null);
  const gainNodeRef = useRef(null);
  const intervalRef = useRef(null);
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
        console.warn('AudioContext initialization failed:', err);
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
      if (isAlarmActive) {
        if (!gainNodeRef.current) {
          gainNodeRef.current = audioCtxRef.current.createGain();
          gainNodeRef.current.gain.value = 0; // Start silent
          gainNodeRef.current.connect(audioCtxRef.current.destination);

          oscillatorRef.current = audioCtxRef.current.createOscillator();
          oscillatorRef.current.type = 'square';
          oscillatorRef.current.frequency.value = 400; // Base frequency
          oscillatorRef.current.connect(gainNodeRef.current);
          oscillatorRef.current.start();
        }

        // Wobble the frequency to make it sound like a siren
        let isHigh = false;
        intervalRef.current = setInterval(() => {
          if (oscillatorRef.current && gainNodeRef.current && audioCtxRef.current) {
            oscillatorRef.current.frequency.setValueAtTime(
              isHigh ? 600 : 400,
              audioCtxRef.current.currentTime
            );
            gainNodeRef.current.gain.setValueAtTime(0.1, audioCtxRef.current.currentTime);
          }
          isHigh = !isHigh;
        }, 300); // Toggle every 300ms

      } else {
        // Turn off alarm
        if (intervalRef.current) clearInterval(intervalRef.current);
        if (gainNodeRef.current && audioCtxRef.current) {
          gainNodeRef.current.gain.setValueAtTime(0, audioCtxRef.current.currentTime);
        }
      }
    } catch (err) {
      console.warn('Audio alarm error:', err);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isAlarmActive, userInteracted]);
}
