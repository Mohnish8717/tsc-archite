/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#FFFFFF', // Pure White
        brand: '#FF4500', // International Orange (Accent)
        card: '#FFFFFF', // Pure White
        textMain: '#000000', // Pure Black
        textLight: '#000000',
        textMuted: '#666666',
        textDark: '#FFFFFF', // Pure White
        primary: '#000000',
        secondary: '#111111',
        accent: '#FF4500', // International Orange
        tension: '#FF4500',
        // shadcn semantic colors
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        foreground: "hsl(var(--foreground))",
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      fontFamily: {
        sans: ['"Fira Sans"', 'sans-serif'],
        mono: ['"Fira Code"', 'monospace'],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ripple': 'ripple 1s ease-out forwards',
        'marquee': 'marquee 30s linear infinite',
      },
      keyframes: {
        ripple: {
          '0%': { transform: 'scale(0.8)', opacity: '1' },
          '100%': { transform: 'scale(2.5)', opacity: '0' },
        },
        marquee: {
          '0%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(-100%)' },
        },
      },
      boxShadow: {
        'neo-black': '8px 8px 0px 0px rgba(0,0,0,1)',
        'neo-white': '8px 8px 0px 0px rgba(255,255,255,1)',
        'neo-brand': '8px 8px 0px 0px #FF4500',
        'neo-pressed': '0px 0px 0px 0px rgba(0,0,0,1)',
      },
      borderWidth: {
        '3': '3px',
        '4': '4px',
        '8': '8px',
      }
    },
  },
  plugins: [],
}
