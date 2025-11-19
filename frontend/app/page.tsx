'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Brain, Zap, ArrowRight, TrendingUp, Database } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Deep Learning CNN',
    description: '7.1M parameter neural network trained on 12,162 audio samples',
  },
  {
    icon: TrendingUp,
    title: '97.09% Accuracy',
    description: 'State-of-the-art performance on professional emotion datasets',
  },
  {
    icon: Zap,
    title: 'Real-time Processing',
    description: 'Sub-100ms inference time with CUDA acceleration',
  },
  {
    icon: Database,
    title: '7 Emotion Classes',
    description: 'Detects angry, disgust, fear, happy, neutral, sad, surprise',
  },
];

const stats = [
  { value: '97.09%', label: 'Accuracy' },
  { value: '7.1M', label: 'Parameters' },
  { value: '<100ms', label: 'Inference' },
  { value: '12K+', label: 'Training Samples' },
];

export default function LandingPage() {
  return (
    <main className="relative min-h-screen">
      <div className="grid-background fixed inset-0 opacity-30" />
      
      <div className="relative z-10">
        {/* Hero Section */}
        <section className="min-h-[90vh] flex items-center justify-center px-4 py-20">
          <div className="max-w-6xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8">
                <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                <span className="text-sm text-gray-300">Powered by PyTorch & CUDA</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6">
                <span className="text-gradient">Speech Emotion</span>
                <br />
                <span className="text-white">Recognition AI</span>
              </h1>

              {/* Description */}
              <p className="text-xl md:text-2xl text-gray-400 mb-12 max-w-3xl mx-auto">
                Advanced neural network system that analyzes voice recordings to detect emotional states with professional-grade accuracy
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/app">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="button-3d px-8 py-4 rounded-xl text-white font-semibold flex items-center gap-2 border border-white/20"
                  >
                    Try the App
                    <ArrowRight className="w-5 h-5" />
                  </motion.button>
                </Link>
                <Link href="/details">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-8 py-4 rounded-xl text-white font-semibold border border-white/10 hover:bg-white/5 transition-all"
                  >
                    Learn More
                  </motion.button>
                </Link>
              </div>
            </motion.div>

            {/* Stats Bar */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto"
            >
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
                  className="card-3d rounded-xl p-6 text-center hover-lift"
                >
                  <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-500 uppercase tracking-wider">{stat.label}</div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-4">
          <div className="max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl md:text-5xl font-bold text-gradient mb-4">
                Key Features
              </h2>
              <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                Professional-grade emotion recognition powered by cutting-edge AI technology
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="card-3d rounded-2xl p-8 hover-lift"
                  >
                    <div className="flex items-start gap-4">
                      <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                        <Icon className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-xl font-semibold text-white mb-2">
                          {feature.title}
                        </h3>
                        <p className="text-gray-400">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="glass-strong rounded-3xl p-12 text-center"
            >
              <h2 className="text-4xl font-bold text-white mb-4">
                Ready to Analyze Emotions?
              </h2>
              <p className="text-xl text-gray-400 mb-8">
                Upload your audio file and get instant emotion detection results
              </p>
              <Link href="/app">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="button-3d px-8 py-4 rounded-xl text-white font-semibold flex items-center gap-2 mx-auto border border-white/20"
                >
                  Get Started
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-8 px-4 border-t border-white/10">
          <div className="max-w-6xl mx-auto text-center">
            <p className="text-sm text-gray-500">
              © 2025 VoiceMood AI • Neural Speech Emotion Recognition System
            </p>
          </div>
        </footer>
      </div>
    </main>
  );
}
