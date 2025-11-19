'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Brain, Zap, ArrowRight, TrendingUp, Database, Activity } from 'lucide-react';
import { useEffect, useState } from 'react';

// Animated Waveform Visualization
function AudioWaveform() {
  const [heights, setHeights] = useState<number[]>(() =>
    Array.from({ length: 40 }, () => Math.random() * 100 + 20)
  );

  useEffect(() => {
    // Animate bars continuously
    const interval = setInterval(() => {
      setHeights(Array.from({ length: 40 }, () => Math.random() * 100 + 20));
    }, 150);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center justify-center gap-1 h-64">
      {heights.map((height, index) => (
        <motion.div
          key={index}
          animate={{ height: `${height}%` }}
          transition={{ duration: 0.15, ease: "easeInOut" }}
          className="w-2 bg-gradient-to-t from-purple-600 via-pink-500 to-indigo-400 rounded-full"
          style={{
            boxShadow: '0 0 20px rgba(168, 85, 247, 0.5)',
          }}
        />
      ))}
    </div>
  );
}

const features = [
  {
    icon: Brain,
    title: 'Deep Learning CNN',
    description: '7.1M parameter neural network trained on 12,162 audio samples',
    gradient: 'from-purple-500 to-pink-500',
  },
  {
    icon: TrendingUp,
    title: '97.09% Accuracy',
    description: 'State-of-the-art performance on professional emotion datasets',
    gradient: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Zap,
    title: 'Real-time Processing',
    description: 'Sub-100ms inference time with CUDA acceleration',
    gradient: 'from-yellow-500 to-orange-500',
  },
  {
    icon: Database,
    title: '7 Emotion Classes',
    description: 'Detects angry, disgust, fear, happy, neutral, sad, surprise',
    gradient: 'from-green-500 to-emerald-500',
  },
];

const stats = [
  { value: '97.09%', label: 'Accuracy', icon: Activity },
  { value: '7.1M', label: 'Parameters', icon: Brain },
  { value: '<100ms', label: 'Inference', icon: Zap },
  { value: '12K+', label: 'Training Samples', icon: Database },
];

export default function LandingPage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Gradient Overlays */}
      <div className="fixed inset-0 z-0 bg-gradient-to-br from-purple-900/20 via-transparent to-blue-900/20 pointer-events-none" />
      <div className="fixed inset-0 z-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/30 via-transparent to-transparent pointer-events-none" />
      
      <div className="relative z-10">
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center px-4 py-20">
          <div className="max-w-7xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Left Content */}
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 1 }}
              >
                {/* Floating Badge */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 mb-8"
                >
                  <div className="w-2 h-2 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 animate-pulse" />
                  <span className="text-sm text-gray-300 font-medium">Powered by PyTorch & CUDA</span>
                </motion.div>

                {/* Main Heading with 3D Effect */}
                <h1 className="text-6xl md:text-7xl xl:text-8xl font-black mb-6 leading-tight">
                  <motion.span
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.3 }}
                    className="block text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-indigo-500"
                    style={{ 
                      textShadow: '0 0 40px rgba(168, 85, 247, 0.4)',
                      WebkitTextStroke: '1px rgba(168, 85, 247, 0.1)'
                    }}
                  >
                    Speech Emotion
                  </motion.span>
                  <motion.span
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className="block text-white"
                    style={{ textShadow: '0 0 30px rgba(255, 255, 255, 0.1)' }}
                  >
                    Recognition AI
                  </motion.span>
                </h1>

                {/* Description */}
                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.7 }}
                  className="text-xl md:text-2xl text-gray-300 mb-10 leading-relaxed"
                >
                  Advanced neural network system that analyzes voice recordings to detect emotional states with{' '}
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-500 font-bold">
                    97.09% accuracy
                  </span>
                </motion.p>

                {/* CTA Buttons with Enhanced Design */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.9 }}
                  className="flex flex-col sm:flex-row items-start sm:items-center gap-4"
                >
                  <Link href="/app">
                    <motion.button
                      whileHover={{ scale: 1.05, y: -2 }}
                      whileTap={{ scale: 0.95 }}
                      className="group relative px-8 py-4 rounded-2xl text-white font-bold text-lg overflow-hidden"
                    >
                      {/* Animated Background */}
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 transition-all" />
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 opacity-0 group-hover:opacity-100 blur-xl transition-all" />
                      <span className="relative flex items-center gap-2">
                        Try the App
                        <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                      </span>
                    </motion.button>
                  </Link>
                  <Link href="/details">
                    <motion.button
                      whileHover={{ scale: 1.05, y: -2 }}
                      whileTap={{ scale: 0.95 }}
                      className="px-8 py-4 rounded-2xl text-white font-bold text-lg border-2 border-white/20 hover:bg-white/5 backdrop-blur-xl transition-all"
                    >
                      Learn More
                    </motion.button>
                  </Link>
                </motion.div>

                {/* Emotion Tags */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 1.1 }}
                  className="mt-10 flex flex-wrap gap-2"
                >
                  {['Angry', 'Happy', 'Sad', 'Fear', 'Surprise', 'Disgust', 'Neutral'].map((emotion, index) => (
                    <motion.span
                      key={emotion}
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, delay: 1.2 + index * 0.1 }}
                      whileHover={{ scale: 1.1, y: -2 }}
                      className="px-4 py-2 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 text-sm text-gray-300 cursor-default"
                    >
                      {emotion}
                    </motion.span>
                  ))}
                </motion.div>
              </motion.div>

              {/* Right Content - Audio Waveform Visualization */}
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 1, delay: 0.3 }}
                className="relative h-[600px] hidden lg:flex items-center justify-center"
              >
                {/* Glowing Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-purple-600/10 via-pink-600/10 to-indigo-600/10 rounded-3xl backdrop-blur-xl border border-white/10" />
                
                {/* Waveform Container */}
                <div className="relative z-10 w-full px-12">
                  <AudioWaveform />
                  
                  {/* Pulsing Ring Effect */}
                  <motion.div
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 0.2, 0.5],
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full border-4 border-purple-500/30"
                  />
                  <motion.div
                    animate={{
                      scale: [1, 1.3, 1],
                      opacity: [0.3, 0.1, 0.3],
                    }}
                    transition={{
                      duration: 4,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: 0.5
                    }}
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[28rem] h-[28rem] rounded-full border-4 border-pink-500/20"
                  />
                </div>
                
                {/* Floating Gradient Orbs */}
                <motion.div
                  animate={{
                    y: [0, -20, 0],
                    rotate: [0, 360],
                  }}
                  transition={{
                    duration: 8,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  className="absolute top-20 right-20 w-32 h-32 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 blur-3xl opacity-60"
                />
                <motion.div
                  animate={{
                    y: [0, 30, 0],
                    rotate: [360, 0],
                  }}
                  transition={{
                    duration: 10,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  className="absolute bottom-32 left-10 w-40 h-40 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 blur-3xl opacity-60"
                />
              </motion.div>
            </div>
          </div>
        </section>

        {/* Stats Section with 3D Cards */}
        <section className="py-20 px-4">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="grid grid-cols-2 md:grid-cols-4 gap-6"
            >
              {stats.map((stat, index) => {
                const Icon = stat.icon;
                return (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                    whileInView={{ opacity: 1, scale: 1, rotateY: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    whileHover={{ scale: 1.05, rotateY: 5, z: 50 }}
                    className="relative group"
                    style={{ transformStyle: 'preserve-3d' }}
                  >
                    {/* Glowing Background */}
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl group-hover:blur-2xl transition-all opacity-0 group-hover:opacity-100" />
                    
                    {/* Card */}
                    <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 text-center hover:border-white/20 transition-all">
                      <Icon className="w-8 h-8 mx-auto mb-4 text-purple-400" />
                      <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-2">
                        {stat.value}
                      </div>
                      <div className="text-sm text-gray-400 uppercase tracking-wider font-semibold">
                        {stat.label}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </motion.div>
          </div>
        </section>

        {/* Features Section with Enhanced 3D Cards */}
        <section className="py-20 px-4">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                transition={{ duration: 0.8, type: "spring" }}
                viewport={{ once: true }}
                className="inline-block mb-4"
              >
                <div className="px-6 py-2 rounded-full bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 font-bold">
                    ✨ KEY FEATURES
                  </span>
                </div>
              </motion.div>
              <h2 className="text-5xl md:text-6xl font-black text-white mb-4">
                Cutting-Edge AI Technology
              </h2>
              <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Professional-grade emotion recognition powered by state-of-the-art deep learning
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 50 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.15 }}
                    viewport={{ once: true }}
                    whileHover={{ scale: 1.02, y: -5 }}
                    className="group relative"
                    style={{ transformStyle: 'preserve-3d' }}
                  >
                    {/* Animated Gradient Background */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-20 rounded-3xl blur-2xl transition-all duration-500`} />
                    
                    {/* Card */}
                    <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 hover:border-white/20 transition-all">
                      <div className="flex items-start gap-6">
                        {/* Icon with 3D Effect */}
                        <motion.div
                          whileHover={{ rotateY: 360, scale: 1.1 }}
                          transition={{ duration: 0.8 }}
                          className={`p-4 rounded-2xl bg-gradient-to-br ${feature.gradient} shadow-2xl`}
                          style={{ transformStyle: 'preserve-3d' }}
                        >
                          <Icon className="w-8 h-8 text-white" />
                        </motion.div>
                        
                        {/* Content */}
                        <div className="flex-1">
                          <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-purple-400 group-hover:to-pink-400 transition-all">
                            {feature.title}
                          </h3>
                          <p className="text-gray-400 leading-relaxed">
                            {feature.description}
                          </p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </section>

        {/* CTA Section with Floating Elements */}
        <section className="py-32 px-4 relative overflow-hidden">
          {/* Floating Gradient Orbs */}
          <motion.div
            animate={{
              x: [0, 100, 0],
              y: [0, -50, 0],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute top-10 left-10 w-64 h-64 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 blur-3xl opacity-30"
          />
          <motion.div
            animate={{
              x: [0, -100, 0],
              y: [0, 50, 0],
              scale: [1, 1.3, 1],
            }}
            transition={{
              duration: 18,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute bottom-10 right-10 w-80 h-80 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 blur-3xl opacity-30"
          />

          <div className="max-w-5xl mx-auto relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative group"
            >
              {/* Glowing Border Effect */}
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-3xl opacity-75 group-hover:opacity-100 blur-lg transition-all duration-500" />
              
              {/* Card */}
              <div className="relative bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900 backdrop-blur-2xl rounded-3xl p-16 text-center border border-white/10">
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="absolute top-10 right-10 w-20 h-20 border-4 border-purple-500/30 border-t-purple-500 rounded-full"
                />
                <motion.div
                  animate={{ rotate: [360, 0] }}
                  transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                  className="absolute bottom-10 left-10 w-16 h-16 border-4 border-pink-500/30 border-t-pink-500 rounded-full"
                />

                <h2 className="text-5xl md:text-6xl font-black text-white mb-6">
                  Ready to Analyze Emotions?
                </h2>
                <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-2xl mx-auto">
                  Upload your audio file and get instant emotion detection results with{' '}
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-500 font-bold">
                    industry-leading accuracy
                  </span>
                </p>
                
                <Link href="/app">
                  <motion.button
                    whileHover={{ scale: 1.05, y: -5 }}
                    whileTap={{ scale: 0.95 }}
                    className="group relative px-12 py-5 rounded-2xl text-white font-black text-xl overflow-hidden"
                  >
                    {/* Animated Background */}
                    <motion.div
                      animate={{
                        backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                      className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600"
                      style={{ backgroundSize: '200% 100%' }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 opacity-0 group-hover:opacity-100 blur-2xl transition-all" />
                    <span className="relative flex items-center gap-3">
                      Get Started Now
                      <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform" />
                    </span>
                  </motion.button>
                </Link>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 px-4 border-t border-white/10 relative">
          <div className="max-w-7xl mx-auto">
            <div className="text-center">
              <motion.div
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                className="mb-4"
              >
                <span className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                  VoiceMood AI
                </span>
              </motion.div>
              <p className="text-gray-500 mb-2">
                Neural Speech Emotion Recognition System
              </p>
              <p className="text-sm text-gray-600">
                © 2025 • Built with PyTorch, FastAPI, and Next.js
              </p>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
