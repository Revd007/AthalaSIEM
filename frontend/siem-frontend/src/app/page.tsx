'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../hooks/use-auth'
import { HashLoader } from 'react-spinners'

export default function Home() {
  const router = useRouter()
  const { token } = useAuthStore()

  useEffect(() => {
    if (token) {
      router.replace('/dashboard/overview')
    } else {
      router.replace('/login')
    }
  }, [router, token])

  return (
    <div 
      className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-sky-400 via-blue-400 to-indigo-400 relative overflow-hidden"
    >
      {/* Animated background elements */}
      <div className="absolute inset-0 w-full h-full">
        <div className="absolute w-96 h-96 -top-10 -left-10 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute w-96 h-96 -bottom-10 -right-10 bg-yellow-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute w-96 h-96 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center">
        <HashLoader 
          color="#000B58" 
          size={100} 
          speedMultiplier={1.2}
        />
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