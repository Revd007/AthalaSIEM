import React from 'react';
import { Globe, Mail, Link2, User, AlertTriangle } from 'lucide-react';
import { ThreatActor } from '../../../types/osint';

interface ThreatActorProfileProps {
  actor: ThreatActor;
}

export function ThreatActorProfile({ actor }: ThreatActorProfileProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold">Threat Actor Profile</h3>
        <span className={`px-3 py-1 rounded-full text-sm ${
          actor.riskLevel === 'high' 
            ? 'bg-red-100 text-red-800' 
            : 'bg-yellow-100 text-yellow-800'
        }`}>
          {actor.riskLevel} risk
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ProfileSection
          icon={Globe}
          title="Location"
          content={actor.location}
        />
        <ProfileSection
          icon={Mail}
          title="Associated Emails"
          content={actor.associatedEmails.join(', ')}
        />
        <ProfileSection
          icon={Link2}
          title="Connected Infrastructure"
          content={actor.infrastructure.join(', ')}
        />
        <ProfileSection
          icon={User}
          title="Aliases"
          content={actor.aliases.join(', ')}
        />
      </div>

      {actor.recentActivities && (
        <div className="mt-6">
          <h4 className="font-medium mb-3">Recent Activities</h4>
          <div className="space-y-2">
            {actor.recentActivities.map((activity, index) => (
              <div key={index} className="flex items-start space-x-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span>{activity}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ProfileSection({ 
  icon: Icon, 
  title, 
  content 
}: { 
  icon: React.ElementType;
  title: string;
  content: string;
}) {
  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
      <div className="flex items-center space-x-2 mb-2">
        <Icon className="h-5 w-5 text-blue-500" />
        <span className="font-medium">{title}</span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-300">{content}</p>
    </div>
  );
}