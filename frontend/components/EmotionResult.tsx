'use client';

import { motion } from 'framer-motion';

interface EmotionResultProps {
  emotion: string;
  confidence: Record<string, number>;
}

export default function EmotionResult({ emotion, confidence }: EmotionResultProps) {
  const sortedEmotions = Object.entries(confidence).sort((a, b) => b[1] - a[1]);

  return (
    <div className="card-3d rounded-3xl p-8 h-full">
      {/* Main Emotion Display - Compact */}
      <div className="text-center mb-8 pb-6 border-b border-white/10">
        <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">
          Detected Emotion
        </p>
        <h2 className="text-5xl font-bold text-gradient-accent mb-3">
          {emotion.toUpperCase()}
        </h2>
        <div className="inline-flex px-5 py-2 rounded-full bg-white/5 border border-white/10">
          <span className="text-sm text-gray-300 font-medium">
            {(confidence[emotion] * 100).toFixed(1)}% Confidence
          </span>
        </div>
      </div>

      {/* Confidence Distribution - Compact */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">
          All Predictions
        </h3>

        <div className="space-y-3">
          {sortedEmotions.map(([label, score], index) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + index * 0.03 }}
            >
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm font-medium capitalize text-white">
                  {label}
                </span>
                <span className="text-xs font-semibold text-gray-400">
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="progress-3d h-1.5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${score * 100}%` }}
                  transition={{ duration: 0.5, delay: 0.1 + index * 0.03, ease: 'easeOut' }}
                  className="progress-fill h-full rounded-full"
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
