import type { AppProps } from 'next/app'
import { useAuth } from '../hooks/use-auth'
import '../styles/globals.css'

export default function App({ Component, pageProps }: AppProps) {
  const { isLoading } = useAuth()

  if (isLoading) {
    return <div>Loading...</div>
  }

  return <Component {...pageProps} />
}
