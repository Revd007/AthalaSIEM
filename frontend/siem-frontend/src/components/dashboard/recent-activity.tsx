import React from 'react';

export const RecentActivity = () => {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
      <div className="space-y-2">
        <div className="p-2 bg-gray-50 rounded">
          <p className="text-sm">User login attempt - 2 minutes ago</p>
        </div>
        <div className="p-2 bg-gray-50 rounded">
          <p className="text-sm">System update completed - 15 minutes ago</p>
        </div>
      </div>
    </div>
  );
};