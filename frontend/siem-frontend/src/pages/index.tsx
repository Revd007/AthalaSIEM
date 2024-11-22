import { useEffect } from 'react'
import { useRouter } from 'next/router'
import Image from 'next/image'
import { motion } from 'framer-motion'
import { Bell, ChartBar, Shield } from 'lucide-react'

export default function LandingPage() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to dashboard if already logged in
    const token = localStorage.getItem('auth_token')
    if (token) {
      router.push('/dashboard')
    }
  }, [router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <Image
            src="/logo-athala.png"
            alt="AthalaSIEM Logo"
            width={120}
            height={120}
            className="mx-auto"
          />
          <h1 className="mt-8 text-4xl font-bold text-gray-900 md:text-6xl">
            Welcome to AthalaSIEM
          </h1>
          <p className="mt-4 text-xl text-gray-600">
            Advanced Security Information and Event Management System
          </p>
          
          <div className="mt-12 flex justify-center space-x-4">
            <button
              onClick={() => router.push('/login')}
              className="rounded-lg bg-primary-600 px-8 py-3 text-white hover:bg-primary-700"
            >
              Login
            </button>
            <button
              onClick={() => router.push('/register')}
              className="rounded-lg border border-primary-600 px-8 py-3 text-primary-600 hover:bg-primary-50"
            >
              Register
            </button>
          </div>

          <div className="mt-16 grid gap-8 md:grid-cols-3">
            <motion.div
              whileHover={{ y: -5 }}
              className="rounded-xl bg-white p-6 shadow-lg"
            >
              <div className="mb-4 rounded-full bg-primary-100 p-3 inline-block">
                <Shield className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold">Real-time Monitoring</h3>
              <p className="mt-2 text-gray-600">
                Monitor your security events and system metrics in real-time
              </p>
            </motion.div>

            <motion.div
              whileHover={{ y: -5 }}
              className="rounded-xl bg-white p-6 shadow-lg"
            >
              <div className="mb-4 rounded-full bg-primary-100 p-3 inline-block">
                <Bell className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold">Intelligent Alerts</h3>
              <p className="mt-2 text-gray-600">
                Get notified instantly about security threats and anomalies
              </p>
            </motion.div>

            <motion.div
              whileHover={{ y: -5 }}
              className="rounded-xl bg-white p-6 shadow-lg"
            >
              <div className="mb-4 rounded-full bg-primary-100 p-3 inline-block">
                <ChartBar className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold">Advanced Analytics</h3>
              <p className="mt-2 text-gray-600">
                Gain insights through comprehensive security analytics and reporting
              </p>
            </motion.div>
          </div>
        </motion.div>
      </div>

      <footer className="mt-16 border-t bg-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-600">
            © {new Date().getFullYear()} AthalaSIEM. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}