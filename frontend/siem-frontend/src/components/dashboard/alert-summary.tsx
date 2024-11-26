import { AlertSummaryData } from '../../types/dashboard'

interface AlertSummaryProps {
  alerts?: AlertSummaryData;
}

export default function AlertSummary({ alerts }: AlertSummaryProps) {
  if (!alerts) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Alert Summary</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-red-100 rounded-lg">
          <p className="text-gray-800 font-medium">Critical Alerts</p>
          <p className="text-2xl font-bold text-gray-900">{alerts.critical}</p>
        </div>
        <div className="p-4 bg-orange-100 rounded-lg">
          <p className="text-orange-800 font-medium">High</p>
          <p className="text-2xl font-bold text-orange-900">{alerts.high}</p>
        </div>
        <div className="p-4 bg-yellow-100 rounded-lg">
          <p className="text-yellow-800 font-medium">Medium</p>
          <p className="text-2xl font-bold text-yellow-900">{alerts.medium}</p>
        </div>
        <div className="p-4 bg-blue-100 rounded-lg">
          <p className="text-blue-800 font-medium">Low</p>
          <p className="text-2xl font-bold text-blue-900">{alerts.low}</p>
        </div>
      </div>
      <div className="p-4 bg-gray-100 rounded-lg">
        <p className="text-gray-800 font-medium">Total Alerts</p>
        <p className="text-2xl font-bold text-gray-900">{alerts.total}</p>
      </div>
    </div>
  );
}