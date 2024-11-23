import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../ui/dialog'
import { Button } from '../../ui/button'

interface CreateReportModalProps {
  isOpen: boolean
  onClose: () => void
}

export function CreateReportModal({ isOpen, onClose }: CreateReportModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Generate New Report</DialogTitle>
        </DialogHeader>
        <form className="space-y-4">
          <div>
            <label htmlFor="reportType" className="block text-sm font-medium text-gray-700">
              Report Type
            </label>
            <select
              id="reportType"
              name="reportType"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              <option value="security">Security Overview</option>
              <option value="alerts">Alert Summary</option>
              <option value="logs">Log Analysis</option>
              <option value="compliance">Compliance Report</option>
            </select>
          </div>

          <div>
            <label htmlFor="timeRange" className="block text-sm font-medium text-gray-700">
              Time Range
            </label>
            <select
              id="timeRange"
              name="timeRange"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="custom">Custom Range</option>
            </select>
          </div>

          <div>
            <label htmlFor="format" className="block text-sm font-medium text-gray-700">
              Export Format
            </label>
            <select
              id="format"
              name="format"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              <option value="pdf">PDF</option>
              <option value="csv">CSV</option>
              <option value="xlsx">Excel</option>
            </select>
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700">
              Description (Optional)
            </label>
            <textarea
              id="description"
              name="description"
              rows={3}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Add any additional notes or context for this report"
            />
          </div>

          <Button type="submit" className="w-full">
            Generate Report
          </Button>
        </form>
        <div className="mt-4 flex justify-end">
          <Button onClick={onClose}>Cancel</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}