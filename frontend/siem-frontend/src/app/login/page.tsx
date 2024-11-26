'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '../../hooks/use-auth'

interface LoginFormData {
  username: string
  password: string
}

export default function Login() {
  const router = useRouter()
  const { login, isLoading, loginError, isAuthenticated } = useAuth()
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [formData, setFormData] = useState<LoginFormData>({
    username: '',
    password: '',
  })

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard/overview')
    }
  }, [isAuthenticated, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordError(null)
    try {
      await login(formData)
    } catch (error: any) {
      setPasswordError(error.message)
    }
  }

  const errorMessage = passwordError || loginError?.message

  return (
    <div className="min-h-screen">
      <div className="form-block-wrapper form-block-wrapper--is-login">
        <section className="form-block">
          <div className="form-block__header">
            <h1>Welcome back!</h1>
            <h2>Please sign in to your account</h2>
            {errorMessage && (
              <div className="bg-red-500/20 p-3 rounded">
                <p className="text-sm">{errorMessage}</p>
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group form-group--login">
              <input
                type="text"
                className="form-group__input"
                placeholder="Username"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                required
              />
              <input
                type="password"
                className="form-group__input"
                placeholder="Password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
              />
            </div>

            <button 
              className="button button--primary full-width"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? 'Signing In...' : 'Sign In'}
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