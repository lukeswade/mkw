// Everything that pulls in three.js, the CSG evaluator, or jszip.
//
// App.tsx imports this module *dynamically* so those ~176KB (gzipped) stay out
// of the initial bundle. Anything added here must not be needed before first
// paint — if it is, it belongs in the eager graph instead.

import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import type { ModelParams } from '../types';

export { buildProceduralGeometry, calculatePrintAnalytics } from './procedural-3d';
export { measureMesh, weldVertices } from './mesh';

// NOTE: the exporters are deliberately NOT re-exported here. ./3mf-exporter
// imports jszip, and this barrel is loaded to build the very first model — so
// re-exporting it pulled 30KB of zip machinery into the initial render path.
// App.tsx imports each exporter with its own import() at click time instead.

/**
 * Loads an external STL (the famous-models library, e.g. 3DBenchy), scales it
 * to the requested envelope and drops it on the bed.
 *
 * Promisified from STLLoader's callback API so the caller can await it
 * alongside the dynamic import.
 */
export function loadExternalStl(
  url: string,
  params: Pick<ModelParams, 'width' | 'height' | 'depth'>
): Promise<THREE.BufferGeometry> {
  return new Promise((resolve, reject) => {
    new STLLoader().load(
      url,
      (geometry) => {
        geometry.computeBoundingBox();
        if (geometry.boundingBox) {
          const sizeX = geometry.boundingBox.max.x - geometry.boundingBox.min.x;
          const sizeY = geometry.boundingBox.max.y - geometry.boundingBox.min.y;
          const sizeZ = geometry.boundingBox.max.z - geometry.boundingBox.min.z;
          if (sizeX > 0 && sizeY > 0 && sizeZ > 0) {
            geometry.scale(params.width / sizeX, params.height / sizeY, params.depth / sizeZ);
          }
          geometry.computeBoundingBox();
          geometry.translate(0, -geometry.boundingBox!.min.y, 0); // sit on the bed
        }
        resolve(geometry);
      },
      undefined,
      (err) => reject(err)
    );
  });
}
