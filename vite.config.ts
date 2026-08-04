import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Split the heavy vendors out of the app chunk. Combined with the
        // dynamic import()s in App.tsx this keeps three.js, the CSG evaluator
        // and jszip out of the initial page load — and gives each its own
        // long-lived cache entry, so shipping app code no longer invalidates
        // 550KB of three.js.
        manualChunks(id) {
          if (id.includes('three-bvh-csg') || id.includes('three-mesh-bvh')) return 'vendor-csg';
          if (id.includes('node_modules/three/')) return 'vendor-three';
          if (id.includes('jszip')) return 'vendor-jszip';
          if (id.includes('node_modules/react')) return 'vendor-react';
        },
      },
    },
  },
  server: {
    port: 3000,
  },
});
