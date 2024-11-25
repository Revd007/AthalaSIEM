/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
      './src/components/**/*.{js,ts,jsx,tsx,mdx}',
      './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
      extend: {
        colors: {
          primary: {
            DEFAULT: '#3b82f6', // blue-500
            hover: '#2563eb', // blue-600
          },
          background: {
            DEFAULT: '#ffffff',
            secondary: '#f3f4f6', // gray-100
          },
          text: {
            DEFAULT: '#111827', // gray-900
            secondary: '#6b7280', // gray-500
          },
          border: {
            DEFAULT: '#e5e7eb', // gray-200
          },
          accent: {
            DEFAULT: '#818cf8', // indigo-400
            hover: '#6366f1', // indigo-500
          }
        },
      },
    },
    plugins: [],
  }