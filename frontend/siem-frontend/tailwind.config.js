/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // Mengaktifkan dark mode
    theme: {
      extend: {
        colors: {
          // Warna utama
          primary: {
            50: '#f0f9ff',
            100: '#e0f2fe',
            200: '#bae6fd',
            300: '#7dd3fc',
            400: '#38bdf8',
            500: '#0ea5e9',
            600: '#0284c7',
            700: '#0369a1',
            800: '#075985',
            900: '#0c4a6e',
          },
          // Warna untuk severity/tingkat bahaya
          severity: {
            low: '#22c55e',     // Hijau
            medium: '#f59e0b',   // Kuning
            high: '#ef4444',     // Merah
            critical: '#7f1d1d', // Merah Gelap
          },
          // Warna untuk status
          status: {
            active: '#22c55e',    // Hijau
            pending: '#f59e0b',   // Kuning
            resolved: '#64748b',  // Abu-abu
            failed: '#ef4444',    // Merah
          },
          // Warna background
          background: {
            light: '#ffffff',
            dark: '#1e293b',
            card: {
              light: '#f8fafc',
              dark: '#334155',
            }
          },
          // Warna teks
          text: {
            light: '#1e293b',
            dark: '#f8fafc',
            muted: {
              light: '#64748b',
              dark: '#94a3b8',
            }
          }
        },
        // Spacing khusus
        spacing: {
          'sidebar': '280px',
          'header': '64px',
          'card': '24px',
        },
        // Border radius
        borderRadius: {
          'card': '8px',
          'button': '6px',
        },
        // Shadow
        boxShadow: {
          'card': '0 2px 4px rgba(0, 0, 0, 0.05)',
          'card-hover': '0 4px 6px rgba(0, 0, 0, 0.1)',
          'dropdown': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        },
        // Font
        fontFamily: {
          sans: ['Inter', 'sans-serif'],
          mono: ['JetBrains Mono', 'monospace'],
        },
        // Animasi
        animation: {
          'spin-slow': 'spin 2s linear infinite',
          'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        },
        // Grid template untuk dashboard
        gridTemplateColumns: {
          'dashboard': 'repeat(auto-fit, minmax(300px, 1fr))',
          'metrics': 'repeat(auto-fit, minmax(200px, 1fr))',
        },
      },
    },
    plugins: [
      require('@tailwindcss/forms'),
      require('@tailwindcss/typography'),
    ],
  }