'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'

interface RegisterFormData {
  username: string
  email: string
  password: string
  confirmPassword: string
  full_name: string
  role: 'ADMIN' | 'ANALYST' | 'OPERATOR' | 'VIEWER'
}

const register = async (data: Omit<RegisterFormData, 'confirmPassword'>) => {
  const response = await fetch('/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...data,
      is_active: true
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Registration failed')
  }

  return response.json()
}

export default function Register() {
  const router = useRouter()
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [formData, setFormData] = useState<RegisterFormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    role: 'VIEWER'
  })

  const registerMutation = useMutation({
    mutationFn: (data: Omit<RegisterFormData, 'confirmPassword'>) => register(data),
    onSuccess: () => {
      router.push('/login')
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordError(null)

    if (formData.password !== formData.confirmPassword) {
      setPasswordError('Passwords do not match')
      return
    }

    const { confirmPassword, ...registerData } = formData
    registerMutation.mutate(registerData)
  }

  const errorMessage = passwordError || registerMutation.error?.message

  return (
    <div className="min-h-screen">
      <div className="form-block-wrapper form-block-wrapper--is-signup">
        <section className="form-block">
          <div className="form-block__header">
            <h1>Create Account</h1>
            {errorMessage && (
              <div className="bg-red-500/20 p-3 rounded">
                <p className="text-sm">{errorMessage}</p>
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group form-group--signup">
              <input
                type="text"
                className="form-group__input"
                placeholder="Username"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                required
              />
              <input
                type="email"
                className="form-group__input"
                placeholder="Email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                required
              />
              <input
                type="text"
                className="form-group__input"
                placeholder="Full Name"
                value={formData.full_name}
                onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                required
              />
              <select
                className="form-group__input"
                value={formData.role}
                onChange={(e) => setFormData({ ...formData, role: e.target.value as RegisterFormData['role'] })}
                required
              >
                <option value="VIEWER">Viewer</option>
                <option value="OPERATOR">Operator</option>
                <option value="ANALYST">Analyst</option>
                <option value="ADMIN">Admin</option>
              </select>
              <input
                type="password"
                className="form-group__input"
                placeholder="Password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
              />
              <input
                type="password"
                className="form-group__input"
                placeholder="Confirm Password"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                required
              />
            </div>

            <button 
              className="button button--primary full-width"
              type="submit"
              disabled={registerMutation.isPending}
            >
              {registerMutation.isPending ? 'Creating Account...' : 'Create Account'}
            </button>
          </form>

          <div className="mt-4 text-center">
            <Link 
              href="/login" 
              className="text-sm hover:text-white/80"
            >
              Already have an account? Sign in
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}