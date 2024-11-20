import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { ThreatLocation } from '../../types';

interface ThreatMapProps {
  threats: ThreatLocation[];
}

export const ThreatMap: React.FC<ThreatMapProps> = ({ threats }) => {
  return (
    <MapContainer center={[0, 0]} zoom={2} style={{ height: '400px', width: '100%' }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {threats.map((threat, index) => (
        <CircleMarker
          key={index}
          center={[threat.lat, threat.lng]}
          radius={threat.severity * 5}
          color={getThreatColor(threat.severity)}
        >
          <Popup>
            <h3>{threat.type}</h3>
            <p>Severity: {threat.severity}</p>
            <p>Source IP: {threat.sourceIp}</p>
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
};