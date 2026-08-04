import * as THREE from 'three';
import { weldVertices } from './mesh';

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
export function exportBinarySTL(geometry: THREE.BufferGeometry): ArrayBuffer {
  const rotated = geometry.clone();
  rotated.rotateX(Math.PI / 2);

  const nonIndexedGeo = weldVertices(rotated).toNonIndexed();
  const posAttr = nonIndexedGeo.attributes.position;

  if (!posAttr) {
    throw new Error('Geometry missing position attributes');
  }

  nonIndexedGeo.computeVertexNormals();
  const normalAttr = nonIndexedGeo.attributes.normal;

  const triangleCount = posAttr.count / 3;
  const bufferSize = 80 + 4 + triangleCount * (4 * 12 + 2);
  const buffer = new ArrayBuffer(bufferSize);
  const dataView = new DataView(buffer);

  // 80-byte header
  const headerText = 'Exported from MKW 3D AI Studio (mattkwade.com)';
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

    if (normalAttr) {
      normal.fromBufferAttribute(normalAttr, i);
    } else {
      // Calculate face normal manually
      const cb = new THREE.Vector3().subVectors(v3, v2);
      const ab = new THREE.Vector3().subVectors(v1, v2);
      normal.crossVectors(cb, ab).normalize();
    }

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

export function downloadFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
