import React from 'react';

export const ThreatMap: React.FC = () => {
  return (
    <div>
      <div className="relative h-[300px] bg-slate-100 rounded-lg p-4">
        <div className="absolute inset-0 flex items-center justify-center">
          {/* World map background - simplified representation */}
          <div className="w-full h-full opacity-20 bg-[url('/world-map.svg')] bg-no-repeat bg-center bg-contain" />
        </div>
        
        {/* Threat indicators */}
        <div className="relative z-10">
          <div className="absolute top-1/4 left-1/4 w-3 h-3 bg-red-500 rounded-full animate-ping" />
          <div className="absolute top-1/2 left-1/3 w-3 h-3 bg-yellow-500 rounded-full animate-ping" />
          <div className="absolute bottom-1/3 right-1/4 w-3 h-3 bg-orange-500 rounded-full animate-ping" />
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-white/80 p-2 rounded-md text-sm">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full"></span>
            <span>Critical Threats</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
            <span>High Threats</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-yellow-500 rounded-full"></span>
            <span>Medium Threats</span>
          </div>
        </div>
      </div>
      <h2>Threat Map</h2>
    </div>
  );
};