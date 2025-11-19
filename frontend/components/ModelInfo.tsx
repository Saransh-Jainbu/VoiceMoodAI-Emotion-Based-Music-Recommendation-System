'use client';

import { motion } from 'framer-motion';
import { Brain, Database, Layers, Target, TrendingUp, Cpu } from 'lucide-react';

const modelSpecs = [
  {
    category: 'Architecture',
    icon: Layers,
    details: [
      { label: 'Type', value: 'Convolutional Neural Network (CNN)' },
      { label: 'Parameters', value: '7,100,000' },
      { label: 'Layers', value: 'Conv2D, MaxPooling, Dropout, Dense' },
      { label: 'Input', value: 'MFCC Features (40 coefficients)' },
    ]
  },
  {
    category: 'Training',
    icon: Target,
    details: [
      { label: 'Accuracy', value: '97.09%' },
      { label: 'Loss Function', value: 'Categorical Crossentropy' },
      { label: 'Optimizer', value: 'Adam' },
      { label: 'Epochs', value: '100 (Early Stopping)' },
    ]
  },
  {
    category: 'Dataset',
    icon: Database,
    details: [
      { label: 'RAVDESS', value: '1,440 audio files' },
      { label: 'CREMA-D', value: '7,442 audio files' },
      { label: 'TESS', value: '2,800 audio files' },
      { label: 'SAVEE', value: '480 audio files' },
    ]
  },
  {
    category: 'Performance',
    icon: TrendingUp,
    details: [
      { label: 'Inference Time', value: '<100ms' },
      { label: 'Compute', value: 'CUDA / RTX 3060' },
      { label: 'Precision', value: 'FP32' },
      { label: 'Framework', value: 'PyTorch 2.5.1' },
    ]
  },
];

const emotions = [
  { name: 'Angry', description: 'High arousal, negative valence' },
  { name: 'Disgust', description: 'Moderate arousal, negative valence' },
  { name: 'Fear', description: 'High arousal, negative valence' },
  { name: 'Happy', description: 'High arousal, positive valence' },
  { name: 'Neutral', description: 'Low arousal, neutral valence' },
  { name: 'Sad', description: 'Low arousal, negative valence' },
  { name: 'Surprise', description: 'High arousal, neutral valence' },
];

export default function ModelInfo() {
  return (
    <div className="space-y-6">
      {/* Model Specifications Grid - More Compact */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid md:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {modelSpecs.map((spec, index) => {
          const Icon = spec.icon;
          return (
            <motion.div
              key={spec.category}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="card-3d rounded-xl p-5 hover-lift"
            >
              <div className="flex items-center gap-2 mb-3">
                <div className="p-1.5 rounded-lg bg-white/5 border border-white/10">
                  <Icon className="w-4 h-4 text-white" />
                </div>
                <h3 className="text-base font-semibold text-white">{spec.category}</h3>
              </div>
              <div className="space-y-2">
                {spec.details.map((detail) => (
                  <div key={detail.label} className="flex justify-between items-center text-xs">
                    <span className="text-gray-500">{detail.label}</span>
                    <span className="font-medium text-gray-300 text-right">{detail.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Emotion Classes - Compact Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card-3d rounded-xl p-5"
      >
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-white/5 border border-white/10">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <h3 className="text-base font-semibold text-white">Emotion Classes</h3>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
          {emotions.map((emotion, index) => (
            <motion.div
              key={emotion.name}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + index * 0.03 }}
              className="p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-all text-center"
            >
              <div className="text-white font-semibold text-sm mb-0.5">{emotion.name}</div>
              <div className="text-xs text-gray-600">{emotion.description.split(',')[0]}</div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Technical Pipeline - Horizontal Layout */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card-3d rounded-xl p-5"
      >
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-white/5 border border-white/10">
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <h3 className="text-base font-semibold text-white">Processing Pipeline</h3>
        </div>
        <div className="grid md:grid-cols-4 gap-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-white/10 flex items-center justify-center text-white font-semibold text-xs">
              1
            </div>
            <div>
              <h4 className="text-white font-medium text-sm mb-1">Audio Preprocessing</h4>
              <p className="text-xs text-gray-500">
                22kHz sampling, normalization
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-white/10 flex items-center justify-center text-white font-semibold text-xs">
              2
            </div>
            <div>
              <h4 className="text-white font-medium text-sm mb-1">MFCC Extraction</h4>
              <p className="text-xs text-gray-500">
                40 coefficients via librosa
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-white/10 flex items-center justify-center text-white font-semibold text-xs">
              3
            </div>
            <div>
              <h4 className="text-white font-medium text-sm mb-1">Standardization</h4>
              <p className="text-xs text-gray-500">
                Zero mean, unit variance
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-white/10 flex items-center justify-center text-white font-semibold text-xs">
              4
            </div>
            <div>
              <h4 className="text-white font-medium text-sm mb-1">CNN Classification</h4>
              <p className="text-xs text-gray-500">
                Conv2D layers with dropout
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
