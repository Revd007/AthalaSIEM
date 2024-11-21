interface Tab {
    id: string
    label: string
  }
  
  interface TabsProps {
    tabs: Tab[]
    activeTab: string
    onChange: (tabId: string) => void
  }
  
  export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
    return (
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              className={`
                whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
              `}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    )
  }