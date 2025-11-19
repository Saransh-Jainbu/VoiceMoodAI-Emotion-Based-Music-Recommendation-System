'use client';

import { motion } from 'framer-motion';
import { Music, User, Disc } from 'lucide-react';

interface MusicRecommendationsProps {
  emotion: string;
  recommendations: Array<{
    name: string;
    artist: string;
    album: string;
  }>;
}

export default function MusicRecommendations({ emotion, recommendations }: MusicRecommendationsProps) {
  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="card-3d rounded-3xl p-8 text-center">
        <Music className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-gray-400">
          No recommendations found for {emotion}
        </p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="card-3d rounded-3xl p-8"
    >
      <h3 className="text-2xl font-bold mb-2 text-white">
        Music Recommendations
      </h3>
      <p className="text-sm mb-6 text-gray-400">
        Curated tracks matching your <span className="font-semibold text-white">{emotion}</span> emotional state
      </p>

      <div className="grid gap-4">
        {recommendations.slice(0, 8).map((song, index) => (
          <motion.div
            key={`${song.name}-${index}`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 + index * 0.05 }}
            whileHover={{ scale: 1.02, x: 4 }}
            className="rounded-2xl p-6 transition-all cursor-pointer hover-lift bg-white/5 border border-white/10"
          >
            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex-shrink-0">
                <Music className="w-5 h-5 text-white" />
              </div>

              {/* Song Info */}
              <div className="flex-1 min-w-0">
                <h4 className="font-semibold text-lg mb-1 truncate text-white">
                  {song.name}
                </h4>
                
                <div className="flex items-center gap-3 text-sm text-gray-400">
                  <div className="flex items-center gap-1">
                    <User className="w-3.5 h-3.5" />
                    <span className="truncate">{song.artist}</span>
                  </div>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <Disc className="w-3.5 h-3.5" />
                    <span className="truncate">{song.album}</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
