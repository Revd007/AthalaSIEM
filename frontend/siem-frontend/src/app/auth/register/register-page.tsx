'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { axiosInstance } from '../../../lib/axios'

interface RegisterFormData {
  username: string
  email: string
  password: string
  confirmPassword: string
  full_name: string
  role: 'ADMIN' | 'ANALYST' | 'OPERATOR' | 'VIEWER'
}

export default function Register() {
  const router = useRouter()
  const [formData, setFormData] = useState<RegisterFormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    role: 'VIEWER'
  })

  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setLoading(false)
      return
    }

    try {
      const response = await axiosInstance.post('/auth/register', {
        username: formData.username,
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name,
        role: formData.role,
        is_active: true
      })

      if (response.status === 200 || response.status === 201) {
        router.push('/login')
      }
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <div className="form-block-wrapper form-block-wrapper--is-signup">
        <section className="form-block">
          <div className="form-block__header">
            <h1>Create Account</h1>
            {error && (
              <div className="bg-red-500/20 p-3 rounded">
                <p className="text-sm">{error}</p>
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
              disabled={loading}
            >
              {loading ? 'Creating Account...' : 'Create Account'}
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