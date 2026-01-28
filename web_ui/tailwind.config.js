/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom colors for speakers
        interviewer: {
          bg: '#e0f2fe',      // sky-100
          border: '#0284c7',   // sky-600
          text: '#0c4a6e',     // sky-900
        },
        participant: {
          bg: '#fce7f3',      // pink-100
          border: '#db2777',   // pink-600
          text: '#831843',     // pink-900
        },
      },
    },
  },
  plugins: [],
}
