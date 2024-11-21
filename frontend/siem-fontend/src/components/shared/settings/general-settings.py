import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

export function GeneralSettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">System Information</h3>
        <div className="mt-4 grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
          <Input
            label="System Name"
            defaultValue="SIEM System"
            placeholder="Enter system name"
          />
          <Input
            label="Time Zone"
            defaultValue="UTC+00:00"
            type="select"
          />
          <Input
            label="Data Retention Period"
            defaultValue="30"
            type="number"
            min="1"
            max="365"
          />
          <Input
            label="Log Format"
            defaultValue="JSON"
            type="select"
          />
        </div>
      </div>

      <div>
        <h3 className="text-lg font-medium leading-6 text-gray-900">Contact Information</h3>
        <div className="mt-4 grid grid-cols-1 gap-y-6 sm:grid-cols-2 sm:gap-x-4">
          <Input
            label="Admin Email"
            type="email"
            placeholder="admin@example.com"
          />
          <Input
            label="Support Phone"
            type="tel"
            placeholder="+1 (555) 000-0000"
          />
        </div>
      </div>

      <div className="flex justify-end">
        <Button variant="outline" className="mr-3">
          Cancel
        </Button>
        <Button variant="primary">
          Save Changes
        </Button>
      </div>
    </div>
  )
}