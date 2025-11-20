'use client';

import { motion } from 'framer-motion';
import { Play, Music2, ExternalLink, Sparkles } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';

interface SpotifyWidgetProps {
  trackName: string;
  artist: string;
  album: string;
  spotifyUrl?: string;
  previewUrl?: string;
  albumArt?: string;
  index: number;
}

export default function SpotifyWidget({ 
  trackName, 
  artist, 
  album, 
  spotifyUrl, 
  previewUrl,
  albumArt,
  index 
}: SpotifyWidgetProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (previewUrl) {
      audioRef.current = new Audio(previewUrl);
      audioRef.current.onended = () => setIsPlaying(false);
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, [previewUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ y: -4 }}
      className="group relative"
    >
      {/* Gradient glow on hover */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl opacity-0 group-hover:opacity-20 blur transition-all" />
      
      {/* Card */}
      <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5 hover:bg-white/10 transition-all">
        <div className="flex items-center gap-4">
          {/* Album Art or Icon */}
          <div className="relative flex-shrink-0">
            {albumArt ? (
              <div className="relative w-16 h-16 rounded-xl overflow-hidden">
                <Image 
                  src={albumArt} 
                  alt={album}
                  width={64}
                  height={64}
                  className="w-full h-full object-cover"
                />
                {/* Play overlay */}
                {previewUrl && (
                  <button
                    onClick={togglePlay}
                    className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity"
                  >
                    <Play className={`w-6 h-6 text-white ${isPlaying ? 'animate-pulse' : ''}`} />
                  </button>
                )}
              </div>
            ) : (
              <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
                <Music2 className="w-8 h-8 text-white" />
              </div>
            )}
          </div>

          {/* Song Info */}
          <div className="flex-1 min-w-0">
            <h4 className="font-semibold text-base mb-1 truncate text-white group-hover:text-green-400 transition-colors">
              {trackName}
            </h4>
            <p className="text-sm text-gray-400 truncate mb-1">{artist}</p>
            <p className="text-xs text-gray-500 truncate">{album}</p>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Preview Play Button */}
            {previewUrl && (
              <button
                onClick={togglePlay}
                className="p-2.5 rounded-xl bg-green-600 hover:bg-green-500 text-white transition-all hover:scale-110"
                title="Play preview"
              >
                <Play className={`w-4 h-4 ${isPlaying ? 'animate-pulse' : ''}`} />
              </button>
            )}

            {/* Spotify Link */}
            {spotifyUrl && (
              <a
                href={spotifyUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 hover:text-white transition-all"
                title="Open in Spotify"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
          </div>
        </div>

        {/* Playing indicator */}
        {isPlaying && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-3 right-3"
          >
            <div className="flex items-center gap-1">
              <div className="w-1 h-3 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
              <div className="w-1 h-4 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
              <div className="w-1 h-3 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

// Spotify Featured Section Component
export function SpotifyFeaturedSection({ emotion }: { emotion: string }) {
  const features = [
    { icon: Sparkles, text: 'Curated by AI', color: 'text-purple-400' },
    { icon: Music2, text: 'Mood-matched', color: 'text-pink-400' },
    { icon: Play, text: 'Preview available', color: 'text-green-400' },
  ];

  return (
    <div className="flex items-center justify-between mb-6 pb-6 border-b border-white/10">
      <div>
        <h3 className="text-2xl font-bold text-white mb-1">
          Music for Your Mood
        </h3>
        <p className="text-sm text-gray-400">
          Personalized tracks matching your <span className="text-white font-semibold">{emotion}</span> emotion
        </p>
      </div>
      
      <div className="hidden md:flex items-center gap-4">
        {features.map((feature, index) => (
          <div key={index} className="flex items-center gap-2 text-xs text-gray-400">
            <feature.icon className={`w-4 h-4 ${feature.color}`} />
            <span>{feature.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
