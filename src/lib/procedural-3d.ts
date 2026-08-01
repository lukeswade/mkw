import * as THREE from 'three';
import { ModelParams, PrintAnalytics, MaterialType } from '../types';

export function buildProceduralGeometry(params: ModelParams): THREE.BufferGeometry {
  const {
    type,
    width = 60,
    depth = 40,
    height = 30,
    wallThickness = 2.5,
    holeDiameter = 4.5,
    roundedRadius = 4,
    baseShape,
    isHollow,
  } = params;

  let geometry: THREE.BufferGeometry;

  switch (type) {
    case 'box':
      geometry = createHollowBox(width, depth, height, wallThickness, roundedRadius);
      break;

    case 'sd_holder':
      geometry = createSDHolderTray(width, depth, height, wallThickness);
      break;

    case 'cable_clip':
      geometry = createCableClip(width, depth, height, holeDiameter, wallThickness);
      break;

    case 'keychain':
      geometry = createKeychainTag(width, depth, height);
      break;

    case 'wall_hook':
      geometry = createWallHook(width, depth, height, wallThickness);
      break;

    case 'phone_stand':
      geometry = createPhoneStand(width, depth, height, wallThickness);
      break;

    case 'hex_tray':
      geometry = createHexagonTray(width, depth, height, wallThickness);
      break;

    case 'custom':
    default:
      geometry = createCustomParametricShape(width, depth, height, wallThickness, roundedRadius, baseShape, isHollow);
      break;
  }

  // Ensure origin is centered on print bed Z=0
  geometry.computeBoundingBox();
  if (geometry.boundingBox) {
    const minY = geometry.boundingBox.min.y;
    geometry.translate(0, -minY, 0);
  }

  return geometry;
}

// 1. Hollow Box / Storage Container
function createHollowBox(w: number, d: number, h: number, wall: number, radius: number): THREE.BufferGeometry {
  const outerShape = new THREE.Shape();
  const r = Math.min(radius, w / 4, d / 4);
  const hw = w / 2;
  const hd = d / 2;

  // Outer rounded rectangle path
  outerShape.moveTo(-hw + r, -hd);
  outerShape.lineTo(hw - r, -hd);
  outerShape.quadraticCurveTo(hw, -hd, hw, -hd + r);
  outerShape.lineTo(hw, hd - r);
  outerShape.quadraticCurveTo(hw, hd, hw - r, hd);
  outerShape.lineTo(-hw + r, hd);
  outerShape.quadraticCurveTo(-hw, hd, -hw, hd - r);
  outerShape.lineTo(-hw, -hd + r);
  outerShape.quadraticCurveTo(-hw, -hd, -hw + r, -hd);

  // Inner cutout shape (wall thickness)
  const iw = Math.max(2, hw - wall);
  const id = Math.max(2, hd - wall);
  const ir = Math.max(1, r - wall / 2);

  const holePath = new THREE.Path();
  holePath.moveTo(-iw + ir, -id);
  holePath.lineTo(iw - ir, -id);
  holePath.quadraticCurveTo(iw, -id, iw, -id + ir);
  holePath.lineTo(iw, id - ir);
  holePath.quadraticCurveTo(iw, id, iw - ir, id);
  holePath.lineTo(-iw + ir, id);
  holePath.quadraticCurveTo(-iw, id, -iw, id - ir);
  holePath.lineTo(-iw, -hd + r);
  holePath.quadraticCurveTo(-iw, -id, -iw + ir, -id);

  // Base floor extrusion
  const floorGeo = new THREE.ExtrudeGeometry(outerShape, {
    depth: wall,
    bevelEnabled: true,
    bevelSegments: 6,
    steps: 1,
    bevelSize: 0.5,
    bevelThickness: 0.5,
    curveSegments: 32,
  });
  floorGeo.rotateX(Math.PI / 2);

  // Walls extrusion with hollow cavity
  outerShape.holes.push(holePath);
  const wallGeo = new THREE.ExtrudeGeometry(outerShape, {
    depth: h - wall,
    bevelEnabled: false,
    steps: 1,
    curveSegments: 32,
  });
  wallGeo.rotateX(Math.PI / 2);
  wallGeo.translate(0, wall, 0);

  return mergeBufferGeometries([floorGeo, wallGeo]);
}

// 2. SD & MicroSD Card Organizer Tray
function createSDHolderTray(w: number, d: number, h: number, wall: number): THREE.BufferGeometry {
  const baseGeo = createHollowBox(w, d, h, wall, 3);
  const slots: THREE.BufferGeometry[] = [baseGeo];

  // Add internal divider slots for SD cards
  const slotCount = Math.floor((d - wall * 2) / 6);
  const slotWidth = w - wall * 2 - 4;

  for (let i = 0; i < slotCount; i++) {
    const divider = new THREE.BoxGeometry(slotWidth, h * 0.75, wall);
    const zPos = -d / 2 + wall + 4 + i * 6;
    divider.translate(0, (h * 0.75) / 2 + wall, zPos);
    slots.push(divider);
  }

  return mergeBufferGeometries(slots);
}

// 3. Wall Mountable Cable Clip
function createCableClip(w: number, d: number, h: number, holeDia: number, wall: number): THREE.BufferGeometry {
  const outerRadius = Math.max(h / 2, 8);
  const clipShape = new THREE.Shape();

  // Draw clip arch
  clipShape.absarc(0, outerRadius, outerRadius, Math.PI, 0, false);
  clipShape.lineTo(outerRadius, 0);
  clipShape.lineTo(-outerRadius, 0);

  // Hole for cables
  const innerRadius = Math.max(3, outerRadius - wall);
  const holePath = new THREE.Path();
  holePath.absarc(0, outerRadius, innerRadius, 0, Math.PI * 2, true);
  clipShape.holes.push(holePath);

  const mainGeo = new THREE.ExtrudeGeometry(clipShape, { depth: d, bevelEnabled: true, bevelSize: 0.8, bevelThickness: 0.8, bevelSegments: 6, curveSegments: 32 });

  // Base mounting tab
  const tabGeo = new THREE.BoxGeometry(w, wall * 2, d);
  tabGeo.translate(0, wall, d / 2);

  // Screw hole cutout simulation
  const screwHole = new THREE.CylinderGeometry(holeDia / 2, holeDia / 2, wall * 3, 32);
  screwHole.translate(w / 3, wall, d / 2);

  return mergeBufferGeometries([mainGeo, tabGeo]);
}

// 4. Custom Keychain Tag with Ring Hole
function createKeychainTag(w: number, d: number, h: number): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const hw = w / 2;
  const hd = d / 2;
  const r = 5;

  shape.moveTo(-hw + r, -hd);
  shape.lineTo(hw - r, -hd);
  shape.quadraticCurveTo(hw, -hd, hw, -hd + r);
  shape.lineTo(hw, hd - r);
  shape.quadraticCurveTo(hw, hd, hw - r, hd);
  shape.lineTo(-hw + r, hd);
  shape.quadraticCurveTo(-hw, hd, -hw, hd - r);
  shape.lineTo(-hw, -hd + r);
  shape.quadraticCurveTo(-hw, -hd, -hw + r, -hd);

  // Keychain ring hole
  const holePath = new THREE.Path();
  const holeRadius = 2.5;
  holePath.absarc(-hw + r + 2, 0, holeRadius, 0, Math.PI * 2, true);
  shape.holes.push(holePath);

  const tagGeo = new THREE.ExtrudeGeometry(shape, {
    depth: h,
    bevelEnabled: true,
    bevelSize: 0.6,
    bevelThickness: 0.6,
    bevelSegments: 6,
    curveSegments: 32,
  });
  tagGeo.rotateX(Math.PI / 2);

  // Raised rim for premium finish
  const rimShape = new THREE.Shape();
  rimShape.moveTo(-hw + r, -hd);
  rimShape.lineTo(hw - r, -hd);
  rimShape.quadraticCurveTo(hw, -hd, hw, -hd + r);
  rimShape.lineTo(hw, hd - r);
  rimShape.quadraticCurveTo(hw, hd, hw - r, hd);
  rimShape.lineTo(-hw + r, hd);
  rimShape.quadraticCurveTo(-hw, hd, -hw, hd - r);
  rimShape.lineTo(-hw, -hd + r);
  rimShape.quadraticCurveTo(-hw, -hd, -hw + r, -hd);

  const innerRimPath = new THREE.Path();
  const rimWall = 1.8;
  innerRimPath.moveTo(-hw + r + rimWall, -hd + rimWall);
  innerRimPath.lineTo(hw - r - rimWall, -hd + rimWall);
  innerRimPath.quadraticCurveTo(hw - rimWall, -hd + rimWall, hw - rimWall, -hd + r + rimWall);
  innerRimPath.lineTo(hw - rimWall, hd - r - rimWall);
  innerRimPath.quadraticCurveTo(hw - rimWall, hd - rimWall, hw - r - rimWall, hd - rimWall);
  innerRimPath.lineTo(-hw + r + rimWall, hd - rimWall);
  innerRimPath.quadraticCurveTo(-hw + rimWall, hd - rimWall, -hw + rimWall, hd - r - rimWall);
  innerRimPath.lineTo(-hw + rimWall, -hd + r + rimWall);
  innerRimPath.quadraticCurveTo(-hw + rimWall, -hd + rimWall, -hw + r + rimWall, -hd + rimWall);
  rimShape.holes.push(innerRimPath);

  const rimGeo = new THREE.ExtrudeGeometry(rimShape, { depth: 1.2, bevelEnabled: false, curveSegments: 32 });
  rimGeo.rotateX(Math.PI / 2);
  rimGeo.translate(0, h, 0);

  return mergeBufferGeometries([tagGeo, rimGeo]);
}

// 5. Heavy Duty Wall Mount Hook
function createWallHook(w: number, d: number, h: number, wall: number): THREE.BufferGeometry {
  const backplate = new THREE.BoxGeometry(w, h, wall * 1.5);
  backplate.translate(0, h / 2, wall * 0.75);

  const hookArch = new THREE.TorusGeometry(d / 2, wall, 32, 64, Math.PI * 0.85);
  hookArch.rotateY(Math.PI / 2);
  hookArch.translate(0, wall * 2, d / 2 + wall);

  const tip = new THREE.SphereGeometry(wall * 1.2, 32, 32);
  tip.translate(0, d / 2 + wall * 2, d * 0.8);

  return mergeBufferGeometries([backplate, hookArch, tip]);
}

// 6. Angled Desktop Phone Stand
function createPhoneStand(w: number, d: number, h: number, wall: number): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const angle = Math.PI / 3; // 60 degree incline

  shape.moveTo(0, 0);
  shape.lineTo(d, 0);
  shape.lineTo(d, wall * 1.5);
  shape.lineTo(d * 0.3, wall * 1.5);
  shape.lineTo(d * 0.3 + Math.cos(angle) * h, Math.sin(angle) * h);
  shape.lineTo(d * 0.3 + Math.cos(angle) * h - wall, Math.sin(angle) * h);
  shape.lineTo(d * 0.15, wall * 2);
  shape.lineTo(0, wall * 2);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, { depth: w, bevelEnabled: true, bevelSize: 0.5, bevelThickness: 0.5, bevelSegments: 6, curveSegments: 32 });
  geo.rotateY(-Math.PI / 2);
  geo.translate(w / 2, 0, 0);

  return geo;
}

// 7. Modular Hexagonal Drawer Tray
function createHexagonTray(w: number, d: number, h: number, wall: number): THREE.BufferGeometry {
  const hexShape = new THREE.Shape();
  const radius = Math.min(w, d) / 2;

  for (let i = 0; i < 6; i++) {
    const a = (i * Math.PI) / 3;
    const x = radius * Math.cos(a);
    const y = radius * Math.sin(a);
    if (i === 0) hexShape.moveTo(x, y);
    else hexShape.lineTo(x, y);
  }
  hexShape.closePath();

  const innerRadius = radius - wall;
  const holePath = new THREE.Path();
  for (let i = 0; i < 6; i++) {
    const a = (i * Math.PI) / 3;
    const x = innerRadius * Math.cos(a);
    const y = innerRadius * Math.sin(a);
    if (i === 0) holePath.moveTo(x, y);
    else holePath.lineTo(x, y);
  }
  holePath.closePath();

  // Floor
  const floor = new THREE.ExtrudeGeometry(hexShape, { depth: wall, bevelEnabled: true, bevelSize: 0.5, bevelThickness: 0.5 });
  floor.rotateX(Math.PI / 2);

  // Walls
  hexShape.holes.push(holePath);
  const walls = new THREE.ExtrudeGeometry(hexShape, { depth: h - wall, bevelEnabled: false });
  walls.rotateX(Math.PI / 2);
  walls.translate(0, wall, 0);

  return mergeBufferGeometries([floor, walls]);
}

// 8. Custom Parametric Fallback Shape
function createCustomParametricShape(w: number, d: number, h: number, wall: number, radius: number, baseShape?: string, isHollow?: boolean): THREE.BufferGeometry {
  if (baseShape === 'cylinder') {
    if (isHollow) {
      const shape = new THREE.Shape();
      const hw = Math.max(w / 2, 1);
      shape.absarc(0, 0, hw, 0, Math.PI * 2, false);
      const holePath = new THREE.Path();
      holePath.absarc(0, 0, Math.max(0.1, hw - wall), 0, Math.PI * 2, true);
      shape.holes.push(holePath);
      const geo = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false, curveSegments: 32 });
      geo.rotateX(Math.PI / 2);
      geo.translate(0, h, 0);
      return geo;
    } else {
      const geo = new THREE.CylinderGeometry(w / 2, w / 2, h, 32);
      geo.translate(0, h / 2, 0);
      return geo;
    }
  }
  if (baseShape === 'sphere') {
    const geo = new THREE.SphereGeometry(w / 2, 32, 32);
    geo.translate(0, w / 2, 0);
    return geo;
  }
  if (baseShape === 'cone') {
    const geo = new THREE.ConeGeometry(w / 2, h, 32);
    geo.translate(0, h / 2, 0);
    return geo;
  }
  if (baseShape === 'torus') {
    const geo = new THREE.TorusGeometry(w / 2, wall, 32, 64);
    geo.rotateX(Math.PI / 2);
    geo.translate(0, wall, 0);
    return geo;
  }
  if (baseShape === 'pyramid') {
    const geo = new THREE.CylinderGeometry(0, w / 2, h, 4);
    geo.rotateY(Math.PI / 4);
    geo.translate(0, h / 2, 0);
    return geo;
  }

  // Default box fallback
  if (isHollow === false) {
    const geo = new THREE.BoxGeometry(w, h, d);
    geo.translate(0, h / 2, 0);
    return geo;
  } else {
    // Default to a hollow box if not specified otherwise
    return createHollowBox(w, d, h, wall, radius);
  }
}

// Utility to combine multiple buffer geometries safely
function mergeBufferGeometries(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const validGeos = geometries.map(g => (g.index ? g.toNonIndexed() : g.clone()));
  
  let totalVertices = 0;
  validGeos.forEach(g => {
    totalVertices += g.attributes.position.count;
  });

  const mergedPositions = new Float32Array(totalVertices * 3);
  let offset = 0;

  validGeos.forEach(g => {
    const pos = g.attributes.position.array;
    mergedPositions.set(pos, offset);
    offset += pos.length;
  });

  const mergedGeo = new THREE.BufferGeometry();
  mergedGeo.setAttribute('position', new THREE.BufferAttribute(mergedPositions, 3));
  mergedGeo.computeVertexNormals();
  mergedGeo.computeBoundingBox();

  return mergedGeo;
}

// Calculate slicer analytics
export function calculatePrintAnalytics(geometry: THREE.BufferGeometry, material: MaterialType = 'PLA'): PrintAnalytics {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox || new THREE.Box3();

  const sizeX = Math.abs(box.max.x - box.min.x);
  const sizeY = Math.abs(box.max.y - box.min.y);
  const sizeZ = Math.abs(box.max.z - box.min.z);

  // Approximate solid mesh volume (cm³) based on bounding box shell ratio
  const boundingVolumeCm3 = (sizeX * sizeY * sizeZ) / 1000;
  const estimatedSolidVolumeCm3 = Math.max(0.5, boundingVolumeCm3 * 0.35);

  // Material densities in g/cm³
  const densities: Record<MaterialType, number> = {
    PLA: 1.24,
    PETG: 1.27,
    TPU: 1.21,
    ABS: 1.04,
  };

  const weightGrams = Math.round(estimatedSolidVolumeCm3 * densities[material] * 10) / 10;

  // Standard 1.75mm filament length calculation (area = PI * (0.875)^2 mm² = 2.405 mm²)
  const filamentLengthMeters = Math.round((estimatedSolidVolumeCm3 * 1000 / 2.405) / 1000 * 10) / 10;

  // Print time estimation heuristic (approx 18 cm³ per hour on modern high-speed printers)
  const estimatedTimeMin = Math.max(10, Math.round((estimatedSolidVolumeCm3 / 18) * 60));

  return {
    volumeCm3: Math.round(estimatedSolidVolumeCm3 * 10) / 10,
    weightGrams,
    estimatedTimeMin,
    filamentLengthMeters,
    bedCompatibility: {
      bambuLab: sizeX <= 256 && sizeZ <= 256 && sizeY <= 256,
      ender3: sizeX <= 220 && sizeZ <= 220 && sizeY <= 250,
      mini: sizeX <= 180 && sizeZ <= 180 && sizeY <= 180,
    },
  };
}
