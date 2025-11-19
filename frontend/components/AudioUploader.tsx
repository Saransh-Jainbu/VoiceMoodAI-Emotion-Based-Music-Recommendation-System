'use client';

import { useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileAudio, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AudioUploaderProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

export default function AudioUploader({ onFileSelect, isLoading }: AudioUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.wav')) {
      setSelectedFile(file);
      onFileSelect(file);
    } else {
      alert('Please upload a .wav file');
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const clearFile = useCallback(() => {
    setSelectedFile(null);
  }, []);

  return (
    <div className="card-3d rounded-3xl p-8">
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
          <Upload className="w-5 h-5 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white">
          Upload Audio File
        </h2>
      </div>
      <p className="text-sm mb-6 text-gray-400 ml-13">
        Select a .wav file for emotion analysis
      </p>

      {!selectedFile ? (
        <motion.div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            'relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer',
            isDragging 
              ? 'border-white/40 bg-white/5 scale-105' 
              : 'border-white/10 hover:border-white/30 hover:bg-white/5'
          )}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <input
            type="file"
            accept=".wav"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isLoading}
          />

          <motion.div
            animate={{ y: isDragging ? -10 : 0 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="relative inline-block mb-4">
              <Upload className="relative w-16 h-16 mx-auto text-gray-400" />
            </div>
            
            <h3 className="text-xl font-semibold text-white mb-2">
              Drop your audio file here
            </h3>
            <p className="text-sm text-gray-400">
              or click to browse • <span className="text-white">WAV format only</span>
            </p>
          </motion.div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card-3d rounded-2xl p-6 flex items-center justify-between"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-white/5 border border-white/10">
              <FileAudio className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="font-medium text-white">
                {selectedFile.name}
              </p>
              <p className="text-sm text-gray-400">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>

          {!isLoading && (
            <button
              onClick={clearFile}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors group"
            >
              <X className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors" />
            </button>
          )}

          {isLoading && (
            <div className="relative">
              <div className="w-6 h-6 rounded-full border-2 border-white/20 border-t-white animate-spin" />
            </div>
          )}
        </motion.div>
      )}

      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 rounded-xl p-4 border border-white/20 bg-white/5"
        >
          <p className="text-sm text-gray-300">
            <strong className="font-semibold text-white">Note:</strong> Model trained on professional acted speech datasets (RAVDESS, CREMA-D, TESS, SAVEE). Best results with clear, emotionally expressive recordings.
          </p>
        </motion.div>
      )}
    </div>
  );
}
