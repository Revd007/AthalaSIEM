import React from 'react';
import { Timeline, TimelineItem } from '../common/Timeline';
import { Alert } from '../../types';

interface AlertsTimelineProps {
  alerts: Alert[];
}

export const AlertsTimeline: React.FC<AlertsTimelineProps> = ({ alerts }) => {
  return (
    <Timeline>
      {alerts.map((alert) => (
        <TimelineItem
          key={alert.id}
          timestamp={alert.timestamp}
          severity={alert.severity}
          title={alert.title}
        >
          <div className="alert-details">
            <p>{alert.description}</p>
            <div className="alert-metadata">
              <span>Source: {alert.source}</span>
              <span>Category: {alert.category}</span>
            </div>
            {alert.recommendations && (
              <div className="recommendations">
                <h4>Recommended Actions:</h4>
                <ul>
                  {alert.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </TimelineItem>
      ))}
    </Timeline>
  );
};