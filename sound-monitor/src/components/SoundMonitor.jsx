import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function SoundMonitor() {
  const [volume, setVolume] = useState(0);
  const [label, setLabel] = useState('Listening...');
  const [score, setScore] = useState(0);
  const modelRef = useRef(null);
  const classNamesRef = useRef([]);

  useEffect(() => {
    const init = async () => {
      // Load model
      modelRef.current = await tf.loadGraphModel('/yamnet/model.json');

      // Load class labels
      const classMap = await fetch(
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
      ).then(res => res.text());

      classNamesRef.current = classMap
        .split('\n')
        .slice(1)
        .map(line => line.split(',')[2]);

      // Get audio
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      const buffer = new Float32Array(analyser.fftSize);
      source.connect(analyser);

      const loop = async () => {
        analyser.getFloatTimeDomainData(buffer);
        const rms = Math.sqrt(buffer.reduce((a, b) => a + b * b, 0) / buffer.length);
        const db = 20 * Math.log10(rms + 1e-8);
        const norm = Math.max(0, Math.min(100, Math.round((db + 100) * 1.2)));
        setVolume(norm);

        if (Math.random() < 0.05 && modelRef.current) {
        try {
            const input = tf.tensor1d(buffer);
            const [scoresTensor] = await modelRef.current.executeAsync(input);
            const scoresArray = await scoresTensor.array();
            const frameScores = scoresArray[0] || scoresArray;
            const topIdx = frameScores.indexOf(Math.max(...frameScores));
            const name = classNamesRef.current[topIdx];
            setLabel(name);
            setScore((frameScores[topIdx] * 100).toFixed(1));
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
  }, []);

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white shadow-lg rounded-2xl space-y-6">
      <h1 className="text-xl font-bold text-center">Sound Monitor</h1>
      <div className="w-full h-5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            volume < 40 ? 'bg-[green]' : volume < 70 ? 'bg-[orange]' : 'bg-[red]'
          }`}
          style={{ width: `${volume}%` }}
        />
      </div>
      <p className="text-center text-sm text-gray-700">Volume Level: {volume} / 100</p>
      <p className="text-center text-lg font-medium">
        Detected: <span className="text-indigo-600 font-semibold">{label}</span> ({score}%)
      </p>
    </div>
  );
}