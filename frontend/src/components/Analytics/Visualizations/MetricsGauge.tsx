import React from 'react';
import { motion } from 'framer-motion';

interface GaugeProps {
  value: number;
  label: string;
  color: string;
  size?: number;
}

export function MetricsGauge({ value, label, color, size = 120 }: GaugeProps) {
  const radius = size / 2;
  const circumference = radius * Math.PI * 2;
  const progress = (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={radius}
          cy={radius}
          r={radius - 10}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="8"
        />
        <motion.circle
          cx={radius}
          cy={radius}
          r={radius - 10}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span 
          className="text-2xl font-bold"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          key={value}
        >
          {value}%
        </motion.span>
        <span className="text-sm text-gray-500">{label}</span>
      </div>
    </div>
  );
}