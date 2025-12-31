import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Chiikawa 风格配色
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
        },
        cream: "#FFF8E7",
        peach: "#FFE4D6",
        mint: "#D4F0E7",
        lavender: "#E8E0F0",
      },
      fontFamily: {
        sans: ["var(--font-noto)", "Noto Sans SC", "sans-serif"],
        cute: ["var(--font-cute)", "Comic Neue", "cursive"],
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
