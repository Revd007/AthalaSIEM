'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useMutation } from '@tanstack/react-query'

interface LoginFormData {
  username: string
  password: string
}

const login = async (data: LoginFormData) => {
  const response = await fetch('/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Login failed')
  }

  return response.json()
}

export default function Login() {
  const router = useRouter()
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [formData, setFormData] = useState<LoginFormData>({
    username: '',
    password: '',
  })

  const loginMutation = useMutation({
    mutationFn: (data: LoginFormData) => login(data),
    onSuccess: () => {
      router.push('/dashboard')
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordError(null)
    loginMutation.mutate(formData)
  }

  const errorMessage = passwordError || loginMutation.error?.message

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
              disabled={loginMutation.isPending}
            >
              {loginMutation.isPending ? 'Signing In...' : 'Sign In'}
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