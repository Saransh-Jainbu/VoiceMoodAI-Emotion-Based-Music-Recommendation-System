'use client';

import { motion } from 'framer-motion';
import { Brain, Cpu, Database, Layers } from 'lucide-react';

const stats = [
  { icon: Brain, label: 'Architecture', value: 'CNN' },
  { icon: Layers, label: 'Parameters', value: '7.1M' },
  { icon: Cpu, label: 'Accuracy', value: '97.09%' },
  { icon: Database, label: 'Emotions', value: '7 Classes' },
];

export default function Header() {
  return (
    <header className="relative mb-10">
      {/* Main Title Section */}
      <div className="glass-strong rounded-3xl p-8 card-3d">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="relative">
              <div className="absolute inset-0 bg-white/10 rounded-full blur-xl" />
              <div className="relative p-3 rounded-full border border-white/20 bg-gradient-to-br from-white/5 to-transparent">
                <Brain className="w-8 h-8 text-white" />
              </div>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-gradient mb-3">
            VoiceMood AI
          </h1>
          
          <p className="text-lg text-gray-400 font-light mb-4">
            Neural Speech Emotion Recognition System
          </p>
          
          <div className="flex items-center justify-center gap-3 text-sm">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/5">
              <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
              <span className="text-gray-300">Real-time</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/5">
              <Cpu className="w-4 h-4 text-gray-400" />
              <span className="text-gray-300">CUDA</span>
            </div>
          </div>
        </motion.div>

        {/* Stats Grid - Compact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-3"
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.3 + index * 0.1 }}
                className="card-3d rounded-xl p-4 hover-lift stat-glow text-center"
              >
                <Icon className="w-6 h-6 mx-auto mb-2 text-gray-400" />
                <div className="text-xl font-bold text-white mb-0.5">
                  {stat.value}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">
                  {stat.label}
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </header>
  );
}
