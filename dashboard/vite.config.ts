import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    sourcemap: true, // Add sourcemaps for easier debugging
    rollupOptions: {
      output: {
        // Using a simpler chunking strategy
        manualChunks: {
          // React core
          "vendor-react": ["react", "react-dom", "react-router-dom"],

          // UI libraries (all Radix UI components)
          "vendor-ui": [
            "@radix-ui/react-accordion",
            "@radix-ui/react-alert-dialog",
            "@radix-ui/react-avatar",
            "@radix-ui/react-checkbox",
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-label",
            "@radix-ui/react-popover",
            "@radix-ui/react-progress",
            "@radix-ui/react-radio-group",
            "@radix-ui/react-select",
            "@radix-ui/react-separator",
            "@radix-ui/react-slider",
            "@radix-ui/react-slot",
            "@radix-ui/react-switch",
            "@radix-ui/react-tabs",
            "@radix-ui/react-toggle",
            "@radix-ui/react-toggle-group",
            "@radix-ui/react-tooltip",
            "lucide-react",
          ],

          // State management and data fetching
          "vendor-state": ["zustand", "swr", "axios"],

          // Animation and UI effects
          "vendor-animation": ["framer-motion", "sonner"],

          // Chunk for the charting library
          "vendor-charts": ["recharts"],

          // Chunk for the code editor
          "vendor-codemirror": [
            "@uiw/codemirror-theme-vscode",
            "@uiw/react-codemirror",
            "@codemirror/lang-javascript",
            "@codemirror/lang-json",
            "@codemirror/lang-python",
            "@codemirror/language",
            "@codemirror/lint",
          ],
        },
      },
    },
    chunkSizeWarningLimit: 1200, // Increase warning limit slightly
  },
});
