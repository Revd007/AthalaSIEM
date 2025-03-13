import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>AthalaSIEM</title>
        <meta name="description" content="Security Information and Event Management System" />
      </head>
      <body className={inter.className}>
        {children}
      </body>
    </html>
  )
} 