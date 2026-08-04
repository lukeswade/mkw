import * as THREE from 'three';
import JSZip from 'jszip';
import { weldVertices, orientOutward } from './mesh';

/**
 * Serialises a geometry to a 3MF package.
 *
 * Like the STL writer, the mesh is rotated from the scene's Y-up to the Z-up
 * convention 3D printing uses — without it the model arrives in the slicer
 * lying on its side.
 *
 * Welding is delegated to weldVertices(), which compares actual distances
 * across neighbouring grid cells. Deduplicating on a `toFixed(4)` string key
 * (as this did) silently misses any pair that straddles a rounding boundary,
 * leaving phantom cracks in the surface.
 */
/**
 * `createdAt` is opt-in on purpose. Stamping `new Date()` unconditionally — in
 * the metadata AND in every zip entry header, which is JSZip's default — meant
 * exporting the same model twice produced different bytes. That defeats
 * checksum comparison and cache dedupe for no benefit the user asked for.
 * Omitted, the output is byte-identical for identical geometry; pass a date if
 * you genuinely want the export stamped.
 */
export async function export3MF(
  geometry: THREE.BufferGeometry,
  modelTitle = 'Model',
  options: { createdAt?: Date } = {}
): Promise<Blob> {
  const rotated = geometry.clone();
  rotated.rotateX(Math.PI / 2);

  // Normalise winding before indexing, same reason as the STL writer.
  const oriented = orientOutward(weldVertices(rotated)).geometry;
  const welded = oriented.index ? oriented : weldVertices(oriented);
  const posAttr = welded.attributes.position;
  const index = welded.index;

  if (!posAttr || !index) {
    throw new Error('Geometry missing position attribute');
  }

  const vertexRows: string[] = [];
  for (let i = 0; i < posAttr.count; i++) {
    vertexRows.push(
      `<vertex x="${posAttr.getX(i).toFixed(4)}" y="${posAttr.getY(i).toFixed(4)}" z="${posAttr.getZ(i).toFixed(4)}" />`
    );
  }

  const triangleRows: string[] = [];
  for (let i = 0; i < index.count; i += 3) {
    triangleRows.push(
      `<triangle v1="${index.getX(i)}" v2="${index.getX(i + 1)}" v3="${index.getX(i + 2)}" />`
    );
  }

  const verticesXML = vertexRows.join('\n');
  const trianglesXML = triangleRows.join('\n');

  const modelXML = `<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">${escapeXml(modelTitle)}</metadata>
  <metadata name="Application">MKW 3D AI Studio (mattkwade.com)</metadata>${options.createdAt ? `\n  <metadata name="CreationDate">${options.createdAt.toISOString()}</metadata>` : ''}
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
  // Fixed entry date unless one was supplied: JSZip stamps `new Date()` into
  // each local header otherwise, so the archive bytes differ run to run even
  // when the mesh is identical.
  const date = options.createdAt ?? new Date(Date.UTC(2025, 0, 1, 12, 0, 0));

  // createFolders:false matters as much as the date. JSZip otherwise
  // auto-creates `_rels/` and `3D/` directory entries and stamps them with
  // new Date() — the `date` option only reaches the entries you name. Because
  // DOS timestamps tick every 2 seconds, that made the output differ only
  // *sometimes*, which is worse than always. OPC does not need the directory
  // entries, so the simplest fix is not to emit them.
  const entry = { date, createFolders: false };
  zip.file('[Content_Types].xml', contentTypesXML, entry);
  zip.file('_rels/.rels', relsXML, entry);
  zip.file('3D/3dmodel.model', modelXML, entry);

  // DEFLATE, not JSZip's default STORE. The model part is highly repetitive
  // XML — measured ~85% smaller compressed — so storing it uncompressed made
  // every export several times larger than it needed to be.
  return await zip.generateAsync({
    type: 'blob',
    mimeType: 'application/vnd.ms-package.3dmanufacturing-3dmodel',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 },
  });
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
