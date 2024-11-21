export default function Footer() {
    return (
      <footer className="bg-white border-t border-gray-200">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="py-4 flex items-center justify-between">
            <div className="text-sm text-gray-500">
              © {new Date().getFullYear()} SIEM System. All rights reserved.
            </div>
            <div className="flex space-x-6">
              <a
                href="#"
                className="text-sm text-gray-500 hover:text-gray-900"
              >
                Privacy Policy
              </a>
              <a
                href="#"
                className="text-sm text-gray-500 hover:text-gray-900"
              >
                Terms of Service
              </a>
              <a
                href="#"
                className="text-sm text-gray-500 hover:text-gray-900"
              >
                Contact Support
              </a>
            </div>
          </div>
        </div>
      </footer>
    )
  }