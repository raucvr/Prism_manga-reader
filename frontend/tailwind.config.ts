import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    // class names also live in helpers (speaker colours) and stores (reader text sizes)
    "./src/lib/**/*.{js,ts,jsx,tsx}",
    "./src/store/**/*.{js,ts,jsx,tsx}",
    "./src/hooks/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 主色：柔和的粉色
        primary: {
          50: "#fef7f7",
          100: "#fdeef0",
          200: "#fbd5db",
          300: "#f8b4bf",
          400: "#f48a9c",
          500: "#eb6079",
          600: "#d63d5a",
          700: "#b42d48",
          800: "#962841",
          900: "#80263c",
          DEFAULT: "#f8b4bf",
          foreground: "#4a4a4a",
        },
        secondary: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          200: "#b9e5fe",
          300: "#7cd1fd",
          400: "#36bbfa",
          500: "#0ca2eb",
          600: "#0081c9",
          700: "#0167a3",
          800: "#065786",
          900: "#0b496f",
          DEFAULT: "#7cd1fd",
          foreground: "#4a4a4a",
        },
        // Design tokens used across components (mirror the CSS variables in globals.css)
        background: "#FFF8E7",
        foreground: "#4A4A4A",
        border: "#E8E0F0",
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#4A4A4A",
        },
        muted: {
          DEFAULT: "#F0F0F0",
          foreground: "#6B6B6B",
        },
        accent: {
          DEFAULT: "#FFE4D6",
          foreground: "#4A4A4A",
        },
        cream: "#FFF8E7",
        peach: "#FFE4D6",
        mint: "#D4F0E7",
        lavender: "#E8E0F0",
      },
      fontFamily: {
        // --font-inter is set by next/font in layout.tsx; the var() fallbacks keep the
        // declaration valid if a variable is missing (otherwise browsers fall back to serif).
        sans: ["var(--font-inter, ui-sans-serif)", "Noto Sans SC", "system-ui", "sans-serif"],
        cute: ["var(--font-cute, 'Comic Neue')", "Comic Neue", "cursive"],
      },
      borderWidth: {
        3: "3px",
      },
      borderRadius: {
        "4xl": "2rem",
        "5xl": "2.5rem",
      },
      animation: {
        "bounce-slow": "bounce 2s infinite",
        "wiggle": "wiggle 0.5s ease-in-out infinite",
        "float": "float 3s ease-in-out infinite",
      },
      keyframes: {
        wiggle: {
          "0%, 100%": { transform: "rotate(-3deg)" },
          "50%": { transform: "rotate(3deg)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
