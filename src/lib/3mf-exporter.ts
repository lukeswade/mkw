import * as THREE from 'three';
import JSZip from 'jszip';

export async function export3MF(geometry: THREE.BufferGeometry, modelTitle = 'Model'): Promise<Blob> {
  const nonIndexedGeo = geometry.index ? geometry.toNonIndexed() : geometry.clone();
  const posAttr = nonIndexedGeo.attributes.position;

  if (!posAttr) {
    throw new Error('Geometry missing position attribute');
  }

  const vertexMap = new Map<string, number>();
  const vertices: Array<{ x: number; y: number; z: number }> = [];
  const triangles: Array<{ v1: number; v2: number; v3: number }> = [];

  const getOrAddVertex = (x: number, y: number, z: number) => {
    // Round to 4 decimal places for precision & deduplication
    const key = `${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)}`;
    if (vertexMap.has(key)) {
      return vertexMap.get(key)!;
    }
    const idx = vertices.length;
    vertices.push({ x, y, z });
    vertexMap.set(key, idx);
    return idx;
  };

  for (let i = 0; i < posAttr.count; i += 3) {
    const idx1 = getOrAddVertex(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
    const idx2 = getOrAddVertex(posAttr.getX(i + 1), posAttr.getY(i + 1), posAttr.getZ(i + 1));
    const idx3 = getOrAddVertex(posAttr.getX(i + 2), posAttr.getY(i + 2), posAttr.getZ(i + 2));

    triangles.push({ v1: idx1, v2: idx2, v3: idx3 });
  }

  const verticesXML = vertices
    .map(v => `<vertex x="${v.x.toFixed(4)}" y="${v.y.toFixed(4)}" z="${v.z.toFixed(4)}" />`)
    .join('\n');

  const trianglesXML = triangles
    .map(t => `<triangle v1="${t.v1}" v2="${t.v2}" v3="${t.v3}" />`)
    .join('\n');

  const modelXML = `<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">${escapeXml(modelTitle)}</metadata>
  <metadata name="Application">MKW 3D AI Studio (mattkwade.com)</metadata>
  <metadata name="CreationDate">${new Date().toISOString()}</metadata>
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
${verticesXML}
        </vertices>
        <triangles>
${trianglesXML}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" />
  </build>
</model>`;

  const contentTypesXML = `<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>`;

  const relsXML = `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>`;

  const zip = new JSZip();
  zip.file('[Content_Types].xml', contentTypesXML);
  zip.file('_rels/.rels', relsXML);
  zip.file('3D/3dmodel.model', modelXML);

  return await zip.generateAsync({ type: 'blob', mimeType: 'application/vnd.ms-package.3dmanufacturing-3dmodel' });
}

function escapeXml(unsafe: string): string {
  return unsafe.replace(/[<>&'"]/g, c => {
    switch (c) {
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '&': return '&amp;';
      case '\'': return '&apos;';
      case '"': return '&quot;';
      default: return c;
    }
  });
}
