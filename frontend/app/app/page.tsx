'use client';

import { useState } from 'react';
import { Music2 } from 'lucide-react';
import AudioUploader from '@/components/AudioUploader';
import EmotionResult from '@/components/EmotionResult';
import MusicRecommendations from '@/components/MusicRecommendations';
import { detectEmotion, EmotionResponse } from '@/lib/api';

export default function AppPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<EmotionResponse['data'] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await detectEmotion(file);
      setResult(response.data);
    } catch (err) {
      console.error('Error detecting emotion:', err);
      setError('Failed to analyze audio. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen py-8 px-4">
      <div className="grid-background fixed inset-0 opacity-30" />
      
      <div className="max-w-7xl mx-auto relative z-10 space-y-12">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gradient mb-3">
            Emotion Detection App
          </h1>
          <p className="text-xl text-gray-400">
            Upload an audio file to analyze emotional content
          </p>
        </div>

        {/* Upload and Analysis Section */}
        <div className="grid lg:grid-cols-5 gap-6">
          {/* Left - Upload (2 columns) */}
          <div className="lg:col-span-2">
            <AudioUploader onFileSelect={handleFileSelect} isLoading={isLoading} />

            {error && (
              <div className="mt-4 rounded-2xl p-4 bg-red-500/10 border border-red-500/20">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Right - Results (3 columns) */}
          <div className="lg:col-span-3">
            {isLoading && (
              <div className="card-3d rounded-3xl p-12 text-center h-full flex flex-col items-center justify-center">
                <div className="relative w-16 h-16 mx-auto mb-4">
                  <div className="absolute inset-0 rounded-full border-4 border-white/10" />
                  <div className="absolute inset-0 rounded-full border-4 border-t-white animate-spin" />
                </div>
                <p className="text-lg font-medium text-white">
                  Analyzing Emotion...
                </p>
                <p className="text-sm mt-2 text-gray-400">
                  Extracting MFCC features
                </p>
              </div>
            )}

            {result && !isLoading && (
              <EmotionResult 
                emotion={result.emotion} 
                confidence={result.confidence} 
              />
            )}

            {!result && !isLoading && !error && (
              <div className="card-3d rounded-3xl p-12 text-center hover-lift h-full flex flex-col items-center justify-center">
                <div className="mb-6 inline-block">
                  <div className="p-6 rounded-full bg-white/5 border border-white/10">
                    <Music2 className="w-16 h-16 text-gray-400" />
                  </div>
                </div>
                <p className="text-lg font-medium mb-2 text-white">
                  Ready for Analysis
                </p>
                <p className="text-sm text-gray-400">
                  Upload an audio file to begin
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Music Recommendations - Full Width */}
        {result && !isLoading && (
          <MusicRecommendations 
            emotion={result.emotion}
            recommendations={result.recommendations}
          />
        )}
      </div>
    </main>
  );
}
