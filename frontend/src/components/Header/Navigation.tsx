'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Home, 
  Shield,
  Target,
  Brain, 
  Activity,
  PlayCircle,
  AlertTriangle,
  Users,
  FileCheck,
  Network,
  Menu,
  X,
  ChevronDown,
  ChevronUp,
  Database,
  Link2
} from 'lucide-react'

const categories = [
  {
    name: 'Threat Management',
    items: [
      { name: 'Security Events', href: '/dashboard/events', icon: Shield },
      { name: 'Threat Hunting', href: '/dashboard/threat-hunting', icon: Target },
      { name: 'AI Analysis', href: '/dashboard/ai-analysis', icon: Brain },
      { name: 'Log Normalization', href: '/dashboard/normalization', icon: Database },
      { name: 'Correlation Engine', href: '/dashboard/correlation', icon: Link2 }
    ]
  },
  {
    name: 'Incident & Response',
    items: [
      { name: 'Predictive Analytics', href: '/dashboard/predictive', icon: Activity },
      { name: 'Automated Playbooks', href: '/dashboard/playbooks', icon: PlayCircle },
      { name: 'Active Incidents', href: '/dashboard/incidents', icon: AlertTriangle }
    ]
  },
  {
    name: 'System & Compliance',
    items: [
      { name: 'Team Collaboration', href: '/dashboard/collaboration', icon: Users },
      { name: 'Compliance', href: '/dashboard/compliance', icon: FileCheck },
      { name: 'System Health', href: '/dashboard/health', icon: Network }
    ]
  }
]

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home }
]

export function Navigation() {
  const pathname = usePathname()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [openDropdown, setOpenDropdown] = useState<number | null>(null)
  const dropdownRefs = useRef<Array<HTMLDivElement | null>>([])
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([])

  // Initialize refs
  useEffect(() => {
    dropdownRefs.current = Array(categories.length).fill(null)
    buttonRefs.current = Array(categories.length).fill(null)
  }, [])

  // Handle ESC key to close dropdowns
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpenDropdown(null);
        if (isMobileMenuOpen) setIsMobileMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleEsc);

    return () => {
      window.removeEventListener('keydown', handleEsc);
    };
  }, [isMobileMenuOpen]);

  // Position dropdown below button
  useEffect(() => {
    if (openDropdown !== null) {
      const buttonEl = buttonRefs.current[openDropdown];
      const dropdownEl = dropdownRefs.current[openDropdown];
      
      if (buttonEl && dropdownEl) {
        const buttonRect = buttonEl.getBoundingClientRect();
        dropdownEl.style.top = `${buttonRect.bottom + window.scrollY}px`;
        dropdownEl.style.left = `${buttonRect.left + window.scrollX}px`;
      }
    }
  }, [openDropdown]);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (!target.closest('.dropdown-button') && !target.closest('.dropdown-menu')) {
        setOpenDropdown(null);
      }
    };

    window.addEventListener('mousedown', handleClickOutside);
    return () => {
      window.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const setButtonRef = (index: number) => (el: HTMLButtonElement | null) => {
    buttonRefs.current[index] = el;
  };

  const setDropdownRef = (index: number) => (el: HTMLDivElement | null) => {
    dropdownRefs.current[index] = el;
  };

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden lg:flex items-center space-x-4 ml-3">
        {navigation.map((item) => (
          <Link key={item.name} href={item.href} className={`text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200 ${pathname === item.href ? 'font-bold' : ''}`}>
            <item.icon className="inline-block mr-2 h-5 w-5 text-gray-500" />
            {item.name}
          </Link>
        ))}
        {categories.map((category, index) => (
          <div key={index} className="relative">
            <button
              ref={setButtonRef(index)}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setOpenDropdown(openDropdown === index ? null : index);
              }}
              className="dropdown-button flex items-center text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
            >
              {category.name}
              {openDropdown === index ? <ChevronUp className="ml-1 h-4 w-4" /> : <ChevronDown className="ml-1 h-4 w-4" />}
            </button>
            {openDropdown === index && (
              <div 
                ref={setDropdownRef(index)}
                className="dropdown-menu fixed w-58 bg-white dark:bg-gray-800 shadow-lg rounded-lg p-2 z-50 border border-gray-200 dark:border-gray-700
                           transition-all duration-300 ease-out transform origin-top scale-95 opacity-0
                           animate-dropdown-open"
                style={{ position: 'fixed', marginTop: '5px' }}
                onClick={(e) => e.stopPropagation()}
              >
                {category.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link 
                      key={item.name} 
                      href={item.href} 
                      className="block px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                      onClick={() => setOpenDropdown(null)}
                    >
                      <Icon className="inline-block mr-3 h-5 w-5 text-gray-500" />
                      {item.name}
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </nav>
      
      {/* Mobile Menu Button */}
      <div className="lg:hidden">
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 rounded-lg text-gray-600 hover:text-gray-900">
          {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-50 bg-white dark:bg-gray-800 p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Menu</h2>
            <button onClick={() => setIsMobileMenuOpen(false)} className="p-2 rounded-lg text-gray-600 hover:text-gray-900">
              <X className="h-6 w-6" />
            </button>
          </div>
          {navigation.map((item) => (
            <Link key={item.name} href={item.href} className="block px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700">
              <item.icon className="inline-block mr-3 h-5 w-5 text-gray-500" />
              {item.name}
            </Link>
          ))}
          {categories.map((category, index) => (
            <div key={index} className="mt-4">
              <button
                onClick={() => setOpenDropdown(openDropdown === index ? null : index)}
                className="w-full flex items-center justify-between px-4 py-2 text-sm font-medium bg-gray-100 dark:bg-gray-700"
              >
                {category.name}
                {openDropdown === index ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
              </button>
              {openDropdown === index && (
                <div className="mt-2 space-y-1 border-l-2 border-gray-200 dark:border-gray-700 ml-2 pl-2 transition-all duration-300 ease-in-out transform scale-95 opacity-0 animate-dropdown-open">
                  {category.items.map((item) => {
                    const Icon = item.icon;
                    return (
                      <Link 
                        key={item.name} 
                        href={item.href} 
                        className="block px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                        onClick={() => {
                          setOpenDropdown(null);
                          setIsMobileMenuOpen(false);
                        }}
                      >
                        <Icon className="inline-block mr-3 h-5 w-5 text-gray-500" />
                        {item.name}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  )
}