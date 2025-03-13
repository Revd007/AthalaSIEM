import React from 'react';
import { MessageSquare, Users, Bell, Share2 } from 'lucide-react';

const collaborationItems = [
  {
    id: 1,
    type: 'comment',
    user: 'Sarah Chen',
    content: 'Investigating unusual network patterns in sector 3',
    time: '2 min ago',
    avatar: 'https://i.pravatar.cc/40?img=1',
  },
  {
    id: 2,
    type: 'action',
    user: 'Mike Johnson',
    content: 'Initiated incident response for potential data breach',
    time: '5 min ago',
    avatar: 'https://i.pravatar.cc/40?img=2',
  },
  {
    id: 3,
    type: 'notification',
    user: 'System',
    content: 'Automated playbook "Ransomware Response" triggered',
    time: '10 min ago',
  },
];

export function RealTimeCollaboration() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Users className="h-6 w-6 text-indigo-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Team Collaboration</h2>
        </div>
        <div className="flex space-x-2">
          <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
            <Share2 className="h-5 w-5" />
          </button>
          <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300">
            <Bell className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="space-y-4 mb-6">
        {collaborationItems.map((item) => (
          <div key={item.id} className="flex items-start space-x-3">
            {item.avatar ? (
              <img src={item.avatar} alt={item.user} className="w-8 h-8 rounded-full" />
            ) : (
              <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                <Bell className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              </div>
            )}
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <span className="font-medium text-gray-900 dark:text-white">{item.user}</span>
                <span className="text-xs text-gray-500">{item.time}</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{item.content}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="relative">
        <input
          type="text"
          placeholder="Type your message..."
          className="w-full pl-4 pr-12 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
        />
        <button className="absolute right-2 top-2 text-blue-500 hover:text-blue-600">
          <MessageSquare className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}