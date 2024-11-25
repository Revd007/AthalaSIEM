'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../hooks/use-auth'
import { useState } from 'react'
import Link from 'next/link'

export default function LoginPage() {
  const router = useRouter()
  const { token, login } = useAuthStore()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (token) {
      router.push('/dashboard')
    }
  }, [token, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')
    
    try {
      const success = await login(username, password)
      if (success) {
        router.push('/dashboard')
      }
    } catch (err) {
      setError('Invalid username or password')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <div className="form-block-wrapper form-block-wrapper--is-login">
        <section className="form-block">
          <div className="form-block__header">
            <h1>Welcome back!</h1>
            {error && (
              <div className="bg-red-500/20 p-3 rounded">
                <p className="text-sm">{error}</p>
              </div>
            )}
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="form-group form-group--login">
              <input 
                type="text"
                className="form-group__input"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
              <input 
                type="password"
                className="form-group__input"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            
            <button 
              className="button button--primary full-width"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          <div className="mt-4 text-center">
            <Link 
              href="/auth/register" 
              className="text-sm hover:text-white/80"
            >
              Don't have an account? Sign up
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}