// Mesh measurement, welding and validation.
//
// Split out so the geometry generators and both exporters can share it.

import * as THREE from 'three';

/**
 * Exact solid volume and surface area of a triangle mesh.
 *
 * Volume uses the divergence theorem: the signed volumes of the tetrahedra
 * formed by each triangle and the origin sum to the enclosed volume. This is
 * exact for a closed, non-self-intersecting mesh of any shape.
 *
 * The caveat matters: it is only exact if the mesh does not overlap itself.
 * Concatenating two solids that intersect counts the shared region twice,
 * which is why the generators boolean-union rather than just appending.
 */
export function measureMesh(geometry: THREE.BufferGeometry): {
  volumeMm3: number;
  areaMm2: number;
  /**
   * Signed volume. Negative means every face is wound inward — the solid is
   * inside-out, which a slicer may read as a void rather than a body.
   */
  signedVolumeMm3: number;
} {
  const geo = geometry.index ? geometry.toNonIndexed() : geometry;
  const pos = geo.attributes.position;

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const cross = new THREE.Vector3();

  let volume = 0;
  let area = 0;

  for (let i = 0; i < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);

    volume += a.dot(cross.crossVectors(b, c)) / 6;

    ab.subVectors(b, a);
    ac.subVectors(c, a);
    area += cross.crossVectors(ab, ac).length() / 2;
  }

  return { volumeMm3: Math.abs(volume), areaMm2: area, signedVolumeMm3: volume };
}

/**
 * Merges vertices within `tolerance` of each other and returns an indexed
 * geometry.
 *
 * CSG output duplicates a vertex per triangle corner, so the mesh looks
 * closed but shares no edges — a slicer sees thousands of loose triangles
 * rather than a solid. Welding turns it into a real topological surface and
 * drops the zero-area slivers that trigger "non-manifold" warnings.
 *
 * The lookup scans the 3x3x3 cell neighbourhood, because two vertices closer
 * than the tolerance can still land in adjacent cells when they straddle a
 * boundary — rounding alone silently leaves those pairs unwelded.
 */
export function weldVertices(
  geometry: THREE.BufferGeometry,
  tolerance = 1e-4
): THREE.BufferGeometry {
  const source = geometry.index ? geometry.toNonIndexed() : geometry;
  const position = source.attributes.position;

  const cellOf = (v: number) => Math.floor(v / tolerance);
  const lookup = new Map<string, number[]>();
  const vertices: number[] = [];
  const indices: number[] = [];
  const toleranceSq = tolerance * tolerance;

  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i);
    const y = position.getY(i);
    const z = position.getZ(i);
    const cx = cellOf(x);
    const cy = cellOf(y);
    const cz = cellOf(z);

    let index = -1;
    outer: for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = lookup.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!bucket) continue;
          for (const candidate of bucket) {
            const ox = vertices[candidate * 3];
            const oy = vertices[candidate * 3 + 1];
            const oz = vertices[candidate * 3 + 2];
            const distSq = (ox - x) ** 2 + (oy - y) ** 2 + (oz - z) ** 2;
            if (distSq <= toleranceSq) {
              index = candidate;
              break outer;
            }
          }
        }
      }
    }

    if (index === -1) {
      index = vertices.length / 3;
      vertices.push(x, y, z);
      const key = `${cx},${cy},${cz}`;
      const bucket = lookup.get(key);
      if (bucket) bucket.push(index);
      else lookup.set(key, [index]);
    }
    indices.push(index);
  }

  // Drop triangles that collapsed to a line or point during welding — a
  // slicer treats a zero-area facet as a defect.
  const cleaned: number[] = [];
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    if (a !== b && b !== c && a !== c) cleaned.push(a, b, c);
  }

  const welded = new THREE.BufferGeometry();
  welded.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  welded.setIndex(cleaned);
  welded.computeVertexNormals();
  welded.computeBoundingBox();
  return welded;
}

/**
 * Ensures triangles are wound counter-clockwise as seen from outside, which is
 * what STL and 3MF both expect.
 *
 * A boolean subtraction can leave a result whose faces all point inward. It
 * still previews fine — three.js will render backfaces — but a slicer reading
 * the winding sees solid and void swapped. Detected from the sign of the
 * divergence-theorem volume and fixed by reversing each triangle.
 */
export function orientOutward(geometry: THREE.BufferGeometry): {
  geometry: THREE.BufferGeometry;
  flipped: boolean;
} {
  const { signedVolumeMm3 } = measureMesh(geometry);
  if (signedVolumeMm3 >= 0) return { geometry, flipped: false };

  const geo = geometry.index ? geometry.toNonIndexed() : geometry.clone();
  const pos = geo.attributes.position as THREE.BufferAttribute;

  for (let i = 0; i < pos.count; i += 3) {
    const bx = pos.getX(i + 1), by = pos.getY(i + 1), bz = pos.getZ(i + 1);
    pos.setXYZ(i + 1, pos.getX(i + 2), pos.getY(i + 2), pos.getZ(i + 2));
    pos.setXYZ(i + 2, bx, by, bz);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  return { geometry: geo, flipped: true };
}
