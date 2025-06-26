import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function SoundMonitor() {
  const [dbLevel, setDbLevel] = useState(0);
  const [threshold, setThreshold] = useState(70); // Sensitivity control
  const [status, setStatus] = useState('silent');
  const [topPredictions, setTopPredictions] = useState([]);
  const [log, setLog] = useState([]);
  const [alert, setAlert] = useState(false);

  const modelRef = useRef(null);
  const classNamesRef = useRef([]);
  const audioBufferRef = useRef([]);
  const statusHistory = useRef([]);
  const crowdHoldCount = useRef(0);
  const maxHistory = 3;

  const speechKeywords = [
    "speech", "conversation", 
    // "narration", "monologue",
    // "shout", "yell", "whisper", "rapping", "chant", "mantra", "babbling",
    // "singing", "giggle", "laugh", "snicker", "chuckle", "chortle"
  ];

  const crowdKeywords = [
    "crowd", "cheering",
    //  "children shouting", "shouting",
    // "hubbub", "babble", "applause", "chatter", "screaming",
    // "yell", "whoop", "bellow",//"Vehicle"
  ];

  const silentKeywords = [
    "silence", "quiet", "ambient", "white noise", "pink noise", "background noise"
  ];

  useEffect(() => {
    const init = async () => {
      modelRef.current = await tf.loadGraphModel('/yamnet/model.json');

      const classMap = await fetch(
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
      ).then(res => res.text());

      classNamesRef.current = classMap
        .split('\n')
        .slice(1)
        .map(line => line.split(',')[2]);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      const buffer = new Float32Array(analyser.fftSize);
      source.connect(analyser);

      const updateStatus = (newStatus) => {
        statusHistory.current.push(newStatus);
        if (statusHistory.current.length > maxHistory) statusHistory.current.shift();
        const count = statusHistory.current.reduce((acc, val) => {
          acc[val] = (acc[val] || 0) + 1;
          return acc;
        }, {});
        const stable = Object.entries(count).sort((a, b) => b[1] - a[1])[0][0];
        setStatus(stable);
      };

      const loop = async () => {
        analyser.getFloatTimeDomainData(buffer);
        const rms = Math.sqrt(buffer.reduce((a, b) => a + b * b, 0) / buffer.length);
        const dbRaw = 20 * Math.log10(rms + 1e-8);
        const normalizedDb = dbRaw + 100;
        const sensitivityFactor = (120 - threshold) / 50;
        const adjustedDb = normalizedDb * sensitivityFactor;
        const clampedDb = Math.max(0, Math.min(100, Math.round(adjustedDb)));
        setDbLevel(clampedDb);

        if (rms < 0.001) {
          updateStatus('silent');
          requestAnimationFrame(loop);
          return;
        }

        audioBufferRef.current = [...audioBufferRef.current, ...Array.from(buffer)];

        if (audioBufferRef.current.length >= 16000/2 && modelRef.current) {
          const slice = audioBufferRef.current.slice(0, 16000);
          audioBufferRef.current = [];

          try {
            const input = tf.tensor1d(slice);
            const [scoresTensor] = await modelRef.current.executeAsync(input);
            const scoresArray = await scoresTensor.array();
            const frameScores = scoresArray[0] || scoresArray;

            const classList = classNamesRef.current
              .map((name, idx) => ({
                name,
                score: frameScores[idx]
              }))
              .filter(item => item.name);

            const sorted = classList.sort((a, b) => b.score - a.score);
            const top = sorted.slice(0, 10);
            setTopPredictions(top);

            const groupTotals = { speech: 0, crowd: 0, silent: 0 };
            classList.forEach(item => {
              const label = item.name.toLowerCase();
              if (speechKeywords.some(k => label.includes(k))) groupTotals.speech += item.score ;
              if (crowdKeywords.some(k => label.includes(k))) groupTotals.crowd += item.score *10;
              if (silentKeywords.some(k => label.includes(k))) groupTotals.silent += item.score*0.05;
            });

            const maxGroup = Object.entries(groupTotals).sort((a, b) => b[1] - a[1])[0][0];
            updateStatus(maxGroup);

            // Alert logic
            if (maxGroup === 'crowd' && clampedDb > 70) {
              crowdHoldCount.current++;
              if (crowdHoldCount.current >= 3) setAlert(true); // 3 frames ~3 seconds
            } else {
              crowdHoldCount.current = 0;
              setAlert(false);
            }

            setLog(log => [...log.slice(-9), {
              time: new Date().toLocaleTimeString(),
              ...groupTotals
            }]);

            tf.dispose([input, scoresTensor]);
          } catch (err) {
            console.error('Prediction error:', err);
          }
        }

        requestAnimationFrame(loop);
      };

      loop();
    };

    init();
  }, [threshold]);

  return (
    <div className="max-w-xl mx-auto mt-10 p-6 bg-white shadow-lg rounded-2xl space-y-6">
      <h1 className="text-xl font-bold text-center">Sound Monitor</h1>

      {/* Slider for dB Sensitivity */}
      

      {/* dB Bar */}
      <div className="w-full h-5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            dbLevel < 35 ? 'bg-[green]' : dbLevel < 65 ? 'bg-[orange]' : 'bg-[red]'
          }`}
          style={{ width: `${dbLevel}%` }}
        />
      </div>
      <p className="text-center text-sm text-gray-700">Sound Level: {dbLevel} / 100</p>

      {/* Status */}
      <div className="text-center">
        {status === 'crowd' && <p className="text-red-600 font-semibold">🚨 Crowd detected!</p>}
        {status === 'speech' && <p className="text-yellow-600 font-semibold">🗣️ Speech detected</p>}
        {status === 'silent' && <p className="text-gray-500 font-medium">✅ Quiet / Ambient</p>}
      </div>

      {/* Alert Box */}
      {alert && (
        <div className="bg-red-100 text-red-700 border border-red-300 rounded-md px-4 py-2 text-center font-semibold">
          🔊 Crowd noise is too loud!
        </div>
      )}

      {/* Predictions */}
      <div className="bg-gray-50 rounded-xl p-4 shadow-inner">
        <h2 className="text-sm font-semibold text-gray-700 mb-2">Top Predictions:</h2>
        <ul className="text-sm text-gray-800 space-y-1">
          {topPredictions.slice(0, 5).map((item, idx) => (
            <li key={idx}>
              <span className="font-medium">{item.name}</span>: {(item.score * 100).toFixed(1)}%
            </li>
          ))}
        </ul>
      </div>

      {/* Group Score Log */}
      <div className="bg-gray-100 rounded-xl p-4 shadow-inner">
        <h2 className="text-sm font-semibold text-gray-700 mb-2">Group Score Log:</h2>
        <ul className="text-xs text-gray-700 max-h-40 overflow-y-auto space-y-1">
          {log.map((entry, idx) => (
            <li key={idx}>
              <span className="font-mono text-gray-500">{entry.time}</span>{' '}
              | 🗣️ {entry.speech.toFixed(2)} | 👥 {entry.crowd.toFixed(2)} | 🤫 {entry.silent.toFixed(2)}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
