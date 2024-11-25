'use client'

import { useEffect, useState } from 'react'
import type { AppProps } from 'next/app'
import '../styles/globals.css'
import { HashLoader } from 'react-spinners'

export default function App({ Component, pageProps }: AppProps) {
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)
  }, [])

  // Tampilkan loading screen selama hydration
  if (!isHydrated) {
    return (
      <div 
        className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-sky-400 via-blue-400 to-indigo-400 relative overflow-hidden"
      >
        <div className="absolute inset-0 w-full h-full">
          <div className="absolute w-96 h-96 -top-10 -left-10 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
          <div className="absolute w-96 h-96 -bottom-10 -right-10 bg-yellow-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
          <div className="absolute w-96 h-96 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        </div>
        <div className="relative z-10 flex flex-col items-center">
          <HashLoader color="#000B58" size={100} speedMultiplier={1.2} />
          <p className="mt-8 text-white text-xl font-medium animate-pulse">
            AthalaSIEM
          </p>
          <p className='mt-2 text-white text-sm font-medium animate-pulse'>
            Security Information and Event Management System
          </p>
        </div>
      </div>
    )
  }

  return <Component {...pageProps} />
}
