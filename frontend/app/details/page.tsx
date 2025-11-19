'use client';

import { motion } from 'framer-motion';
import { Brain, Layers, Database, TrendingUp, Zap, Award, Target, Cpu, GitBranch, CheckCircle } from 'lucide-react';

const architectureLayers = [
  { id: 'input', name: 'Input Layer', description: 'MFCC Features (40 coefficients)', size: '40 neurons' },
  { id: 'conv1', name: 'Conv2D Layer 1', description: '256 filters, 5×5 kernel, ReLU activation', size: '256 filters' },
  { id: 'maxpool1', name: 'MaxPooling2D Layer 1', description: '2×2 pool size, reduces spatial dimensions', size: '2×2 pool' },
  { id: 'dropout1', name: 'Dropout (25%) - 1', description: 'Prevents overfitting during training', size: '0.25 rate' },
  { id: 'conv2', name: 'Conv2D Layer 2', description: '128 filters, 5×5 kernel, ReLU activation', size: '128 filters' },
  { id: 'maxpool2', name: 'MaxPooling2D Layer 2', description: '2×2 pool size', size: '2×2 pool' },
  { id: 'dropout2', name: 'Dropout (25%) - 2', description: 'Additional regularization', size: '0.25 rate' },
  { id: 'flatten', name: 'Flatten', description: 'Converts 2D feature maps to 1D vector', size: 'Dynamic' },
  { id: 'dense', name: 'Dense Layer', description: '1024 units, ReLU activation', size: '1024 neurons' },
  { id: 'dropout3', name: 'Dropout (50%)', description: 'Strong regularization before output', size: '0.5 rate' },
  { id: 'output', name: 'Output Layer', description: '7 units (emotions), Softmax activation', size: '7 neurons' },
];

const datasets = [
  { 
    name: 'RAVDESS', 
    files: '1,440', 
    description: 'Ryerson Audio-Visual Database of Emotional Speech and Song',
    speakers: '24 actors',
    emotions: 'All 7 emotions'
  },
  { 
    name: 'CREMA-D', 
    files: '7,442', 
    description: 'Crowd-sourced Emotional Multimodal Actors Dataset',
    speakers: '91 actors',
    emotions: 'All 7 emotions'
  },
  { 
    name: 'TESS', 
    files: '2,800', 
    description: 'Toronto Emotional Speech Set',
    speakers: '2 actresses',
    emotions: 'All 7 emotions'
  },
  { 
    name: 'SAVEE', 
    files: '480', 
    description: 'Surrey Audio-Visual Expressed Emotion',
    speakers: '4 actors',
    emotions: 'All 7 emotions'
  },
];

const trainingSpecs = [
  { label: 'Total Samples', value: '12,162 audio files' },
  { label: 'Training Split', value: '80% (9,730 files)' },
  { label: 'Validation Split', value: '20% (2,432 files)' },
  { label: 'Optimizer', value: 'Adam (lr=0.001)' },
  { label: 'Loss Function', value: 'Categorical Crossentropy' },
  { label: 'Batch Size', value: '32' },
  { label: 'Epochs', value: '100 (with Early Stopping)' },
  { label: 'Early Stopping', value: 'Patience=10, monitor val_loss' },
];

const performanceMetrics = [
  { metric: 'Overall Accuracy', value: '97.09%', description: 'Correctly classified samples' },
  { metric: 'Precision', value: '96.8%', description: 'Positive prediction accuracy' },
  { metric: 'Recall', value: '97.1%', description: 'True positive rate' },
  { metric: 'F1-Score', value: '96.9%', description: 'Harmonic mean of precision & recall' },
  { metric: 'Inference Time', value: '<100ms', description: 'Average prediction latency' },
  { metric: 'Model Size', value: '28.4 MB', description: 'Trained model file size' },
];

const emotionClasses = [
  { name: 'Angry', characteristics: 'High arousal, negative valence, raised pitch' },
  { name: 'Disgust', characteristics: 'Moderate arousal, negative valence, vocal distortions' },
  { name: 'Fear', characteristics: 'High arousal, negative valence, trembling voice' },
  { name: 'Happy', characteristics: 'High arousal, positive valence, upward intonation' },
  { name: 'Neutral', characteristics: 'Low arousal, neutral valence, steady tone' },
  { name: 'Sad', characteristics: 'Low arousal, negative valence, lower pitch' },
  { name: 'Surprise', characteristics: 'High arousal, neutral valence, abrupt changes' },
];

const pipeline = [
  { 
    step: '1', 
    title: 'Audio Preprocessing', 
    description: 'Load audio files at 22kHz sampling rate. Apply normalization to ensure consistent amplitude across different recordings. Remove silence and noise artifacts.',
    tech: 'librosa.load(), duration padding'
  },
  { 
    step: '2', 
    title: 'Feature Extraction', 
    description: 'Extract 40 Mel-frequency cepstral coefficients (MFCCs) which capture the spectral envelope of speech. MFCCs are particularly effective for speech emotion recognition.',
    tech: 'librosa.feature.mfcc(n_mfcc=40)'
  },
  { 
    step: '3', 
    title: 'Feature Standardization', 
    description: 'Apply StandardScaler to normalize features to zero mean and unit variance. This improves model convergence and stability during training.',
    tech: 'sklearn.StandardScaler()'
  },
  { 
    step: '4', 
    title: 'Label Encoding', 
    description: 'Convert emotion labels to one-hot encoded vectors for multi-class classification. Each emotion becomes a 7-dimensional binary vector.',
    tech: 'sklearn.OneHotEncoder()'
  },
  { 
    step: '5', 
    title: 'CNN Classification', 
    description: 'Pass preprocessed features through convolutional neural network. Multiple Conv2D layers extract hierarchical features, followed by dense layers for final classification.',
    tech: 'PyTorch CNN (7.1M params)'
  },
  { 
    step: '6', 
    title: 'Prediction Output', 
    description: 'Softmax layer produces probability distribution over 7 emotion classes. Highest probability indicates detected emotion with confidence score.',
    tech: 'torch.nn.Softmax(dim=1)'
  },
];

export default function DetailsPage() {
  return (
    <main className="relative min-h-screen py-12 px-4">
      <div className="grid-background fixed inset-0 opacity-30" />
      
      <div className="max-w-7xl mx-auto relative z-10 space-y-16">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <h1 className="text-5xl md:text-6xl font-bold text-gradient">
            Technical Documentation
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Comprehensive overview of our speech emotion recognition system: architecture, training methodology, datasets, and performance metrics
          </p>
        </motion.div>

        {/* Overview Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid md:grid-cols-4 gap-4"
        >
          <div className="card-3d rounded-xl p-6 text-center">
            <Award className="w-8 h-8 mx-auto mb-3 text-gray-400" />
            <div className="text-3xl font-bold text-white mb-1">97.09%</div>
            <div className="text-sm text-gray-500">Accuracy</div>
          </div>
          <div className="card-3d rounded-xl p-6 text-center">
            <Brain className="w-8 h-8 mx-auto mb-3 text-gray-400" />
            <div className="text-3xl font-bold text-white mb-1">7.1M</div>
            <div className="text-sm text-gray-500">Parameters</div>
          </div>
          <div className="card-3d rounded-xl p-6 text-center">
            <Database className="w-8 h-8 mx-auto mb-3 text-gray-400" />
            <div className="text-3xl font-bold text-white mb-1">12,162</div>
            <div className="text-sm text-gray-500">Training Samples</div>
          </div>
          <div className="card-3d rounded-xl p-6 text-center">
            <Zap className="w-8 h-8 mx-auto mb-3 text-gray-400" />
            <div className="text-3xl font-bold text-white mb-1">&lt;100ms</div>
            <div className="text-sm text-gray-500">Inference Time</div>
          </div>
        </motion.div>

        {/* Architecture Section */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Layers className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Model Architecture</h2>
                <p className="text-gray-400">Convolutional Neural Network with 11 layers</p>
              </div>
            </div>

            <div className="card-3d rounded-2xl p-8">
              <div className="space-y-4">
                {architectureLayers.map((layer, index) => (
                  <motion.div
                    key={layer.id}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    viewport={{ once: true }}
                    className="relative"
                  >
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all">
                      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-white font-bold text-sm">
                        {index + 1}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-white font-semibold mb-1">{layer.name}</h3>
                        <p className="text-sm text-gray-400 mb-2">{layer.description}</p>
                        <span className="inline-flex text-xs px-3 py-1 rounded-full bg-white/5 border border-white/10 text-gray-300">
                          {layer.size}
                        </span>
                      </div>
                    </div>
                    {index < architectureLayers.length - 1 && (
                      <div className="ml-9 h-4 w-0.5 bg-white/10"></div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </section>

        {/* Training Data Section */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Database className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Training Datasets</h2>
                <p className="text-gray-400">Four professional emotional speech databases</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {datasets.map((dataset, index) => (
                <motion.div
                  key={dataset.name}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card-3d rounded-2xl p-6 hover-lift"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-1">{dataset.name}</h3>
                      <p className="text-sm text-gray-400">{dataset.description}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-gradient-accent">{dataset.files}</div>
                      <div className="text-xs text-gray-500">files</div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Speakers:</span>
                      <span className="text-gray-300">{dataset.speakers}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Coverage:</span>
                      <span className="text-gray-300">{dataset.emotions}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Training Specifications */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Target className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Training Configuration</h2>
                <p className="text-gray-400">Hyperparameters and training setup</p>
              </div>
            </div>

            <div className="card-3d rounded-2xl p-8">
              <div className="grid md:grid-cols-2 gap-x-12 gap-y-4">
                {trainingSpecs.map((spec, index) => (
                  <motion.div
                    key={spec.label}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    viewport={{ once: true }}
                    className="flex justify-between items-center py-3 border-b border-white/10"
                  >
                    <span className="text-gray-400">{spec.label}</span>
                    <span className="text-white font-semibold">{spec.value}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </section>

        {/* Processing Pipeline */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <GitBranch className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Processing Pipeline</h2>
                <p className="text-gray-400">Step-by-step emotion detection workflow</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {pipeline.map((step, index) => (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card-3d rounded-2xl p-6 hover-lift"
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center text-white font-bold text-lg border border-white/20">
                      {step.step}
                    </div>
                    <h3 className="text-lg font-bold text-white">{step.title}</h3>
                  </div>
                  <p className="text-sm text-gray-400 mb-4">{step.description}</p>
                  <div className="inline-flex text-xs px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-gray-300 font-mono">
                    {step.tech}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Emotion Classes */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Emotion Classes</h2>
                <p className="text-gray-400">Seven distinct emotional states recognized by the model</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {emotionClasses.map((emotion, index) => (
                <motion.div
                  key={emotion.name}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  viewport={{ once: true }}
                  className="card-3d rounded-xl p-6 hover-lift"
                >
                  <h3 className="text-xl font-bold text-white mb-2">{emotion.name}</h3>
                  <p className="text-sm text-gray-400">{emotion.characteristics}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Performance Metrics */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Performance Metrics</h2>
                <p className="text-gray-400">Comprehensive evaluation results</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {performanceMetrics.map((metric, index) => (
                <motion.div
                  key={metric.metric}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card-3d rounded-2xl p-6 text-center hover-lift"
                >
                  <div className="text-4xl font-bold text-gradient-accent mb-2">{metric.value}</div>
                  <h3 className="text-lg font-semibold text-white mb-2">{metric.metric}</h3>
                  <p className="text-sm text-gray-500">{metric.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Technology Stack */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white">Technology Stack</h2>
                <p className="text-gray-400">Frameworks and libraries used</p>
              </div>
            </div>

            <div className="card-3d rounded-2xl p-8">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5" />
                    Deep Learning & ML
                  </h3>
                  <ul className="space-y-3">
                    {['PyTorch 2.5.1', 'scikit-learn 1.7.2', 'librosa 0.10.2', 'NumPy 1.26.4'].map((tech) => (
                      <li key={tech} className="flex items-center gap-2 text-gray-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-white/40" />
                        {tech}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5" />
                    Backend & Frontend
                  </h3>
                  <ul className="space-y-3">
                    {['FastAPI 0.115.5', 'Next.js 14', 'TypeScript', 'Tailwind CSS'].map((tech) => (
                      <li key={tech} className="flex items-center gap-2 text-gray-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-white/40" />
                        {tech}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="text-center pt-12 border-t border-white/10">
          <p className="text-gray-500 mb-4">
            This system represents state-of-the-art speech emotion recognition technology, combining deep learning with acoustic feature analysis
          </p>
          <p className="text-sm text-gray-600">
            © 2025 VoiceMood AI • Built with PyTorch, FastAPI, and Next.js
          </p>
        </footer>
      </div>
    </main>
  );
}
