'use client'

interface PlaybookCreationModalProps {
  onClose: () => void
}

export function PlaybookCreationModal({ onClose }: PlaybookCreationModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-white p-6 rounded-lg">
        <h2 className="text-xl font-bold mb-4">Create New Playbook</h2>
        {/* Implement creation form */}
        <div className="flex justify-end gap-2">
          <button onClick={onClose}>Cancel</button>
          <button>Create</button>
        </div>
      </div>
    </div>
  )
} 