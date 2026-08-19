import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        vault: {
          bg: "#0B0D10",
          card: "rgba(26, 26, 29, 0.72)",
          subtle: "rgba(20, 20, 23, 0.55)",
          border: "rgba(236, 231, 220, 0.08)",
          borderGold: "rgba(201, 169, 97, 0.28)",
          primary: "#ECE7DC",
          secondary: "#8A8578",
          accent: "#C9A961",
          supported: "#1F6F54",
          abstain: "#7A2E2E",
        },
        parchment: {
          bg: "#F5F0E6",
          card: "rgba(255, 252, 245, 0.85)",
          subtle: "rgba(240, 234, 222, 0.7)",
          border: "rgba(30, 25, 20, 0.12)",
          borderGold: "rgba(168, 126, 42, 0.35)",
          primary: "#181614",
          secondary: "#6A6458",
          accent: "#9E7828",
          supported: "#1F6F54",
          abstain: "#7A2E2E",
        },
      },
      fontFamily: {
        display: ["'Fraunces'", "Georgia", "serif"],
        serif: ["'Fraunces'", "Georgia", "Cambria", "serif"],
        sans: ["'Inter'", "'Public Sans'", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["'IBM Plex Mono'", "JetBrains Mono", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
