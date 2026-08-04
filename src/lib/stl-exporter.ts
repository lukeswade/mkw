import * as THREE from 'three';
import { weldVertices, orientOutward } from './mesh';

/**
 * Serialises a geometry to binary STL.
 *
 * The scene is Y-up (three.js convention) but STL for 3D printing is Z-up.
 * Exporting the scene orientation verbatim dropped every model into the
 * slicer lying on its side; the rotation below fixes that. It is applied to a
 * clone, so the previewed geometry is untouched.
 *
 * The mesh is also welded first, which collapses the duplicate vertex per
 * triangle corner that CSG emits and drops the zero-area slivers that make
 * slicers report a "non-manifold" model.
 */
export function exportBinarySTL(
  geometry: THREE.BufferGeometry,
  options: { title?: string } = {}
): ArrayBuffer {
  const rotated = geometry.clone();
  rotated.rotateX(Math.PI / 2);

  // Normalise winding so a boolean result whose faces ended up pointing inward
  // is not written out as an inside-out solid.
  const nonIndexedGeo = orientOutward(weldVertices(rotated)).geometry.toNonIndexed();
  const posAttr = nonIndexedGeo.attributes.position;

  if (!posAttr) {
    throw new Error('Geometry missing position attributes');
  }

  const triangleCount = posAttr.count / 3;
  const bufferSize = 80 + 4 + triangleCount * (4 * 12 + 2);
  const buffer = new ArrayBuffer(bufferSize);
  const dataView = new DataView(buffer);

  // 80-byte header
  const headerText = `MKW 3D: ${options.title ?? 'model'}`.slice(0, 80);
  for (let i = 0; i < 80; i++) {
    dataView.setUint8(i, i < headerText.length ? headerText.charCodeAt(i) : 32);
  }

  // 4-byte triangle count
  dataView.setUint32(80, triangleCount, true);

  let offset = 84;
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const v3 = new THREE.Vector3();
  const normal = new THREE.Vector3();

  for (let i = 0; i < posAttr.count; i += 3) {
    v1.fromBufferAttribute(posAttr, i);
    v2.fromBufferAttribute(posAttr, i + 1);
    v3.fromBufferAttribute(posAttr, i + 2);

    // Always the geometric face normal. Reading it from the normal attribute
    // instead (as this used to) hands back a *smoothed* vertex normal — the
    // average across adjacent faces from computeVertexNormals — which does not
    // describe the facet's own plane on any curved surface.
    const ab = new THREE.Vector3().subVectors(v2, v1);
    const ac = new THREE.Vector3().subVectors(v3, v1);
    normal.crossVectors(ab, ac);
    if (normal.lengthSq() > 0) normal.normalize();

    // Normal vector
    dataView.setFloat32(offset, normal.x, true); offset += 4;
    dataView.setFloat32(offset, normal.y, true); offset += 4;
    dataView.setFloat32(offset, normal.z, true); offset += 4;

    // Vertex 1
    dataView.setFloat32(offset, v1.x, true); offset += 4;
    dataView.setFloat32(offset, v1.y, true); offset += 4;
    dataView.setFloat32(offset, v1.z, true); offset += 4;

    // Vertex 2
    dataView.setFloat32(offset, v2.x, true); offset += 4;
    dataView.setFloat32(offset, v2.y, true); offset += 4;
    dataView.setFloat32(offset, v2.z, true); offset += 4;

    // Vertex 3
    dataView.setFloat32(offset, v3.x, true); offset += 4;
    dataView.setFloat32(offset, v3.y, true); offset += 4;
    dataView.setFloat32(offset, v3.z, true); offset += 4;

    // Attribute byte count
    dataView.setUint16(offset, 0, true); offset += 2;
  }

  return buffer;
}

// downloadFile moved to ./download so App.tsx can use it without importing
// three.js. Re-exported here for backwards compatibility.
export { downloadFile } from './download';
