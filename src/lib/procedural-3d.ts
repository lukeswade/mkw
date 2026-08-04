import * as THREE from 'three';
import { Evaluator, Brush, SUBTRACTION, ADDITION, INTERSECTION } from 'three-bvh-csg';
import { ModelParams, PrintAnalytics, MaterialType, CSGOperation } from '../types';

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

    case 'swatch':
      geometry = createFilamentSwatch(width, depth, height);
      break;

    case 'spool_tag':
      geometry = createSpoolTag(width, depth, height);
      break;

    case 'csg':
      geometry = buildCSGGeometry(params.operations || []);
      break;

    case 'custom':
    default:
      geometry = createCustomParametricShape(width, depth, height, wallThickness, roundedRadius, baseShape, isHollow);
      break;
  }

  // Apply Twist Modifier if specified!
  if (params.twist && params.twist !== 0) {
    applyTwist(geometry, params.twist);
  }

  // Ensure origin is centered on print bed Z=0
  geometry.computeBoundingBox();
  if (geometry.boundingBox) {
    const minY = geometry.boundingBox.min.y;
    geometry.translate(0, -minY, 0);
  }

  return geometry;
}

/**
 * Global Twist Modifier!
 * Modifies the final BufferGeometry by rotating vertices along the Y axis 
 * proportional to their height. This allows for spirals, twirls, threads, etc.
 */
function applyTwist(geometry: THREE.BufferGeometry, twistDegrees: number) {
  geometry.computeBoundingBox();
  if (!geometry.boundingBox) return;
  
  const minY = geometry.boundingBox.min.y;
  const maxY = geometry.boundingBox.max.y;
  const height = maxY - minY;
  if (height === 0) return;
  
  const twistRadians = twistDegrees * (Math.PI / 180);
  const positionAttribute = geometry.attributes.position;
  const vertex = new THREE.Vector3();

  // If geometry isn't non-indexed, you'd usually want to convert it to non-indexed 
  // for flat shading, but CSG geometries are usually fine, and basic shapes 
  // can twist fine if vertices are shared correctly.
  
  for (let i = 0; i < positionAttribute.count; i++) {
    vertex.fromBufferAttribute(positionAttribute, i);
    
    const yFraction = (vertex.y - minY) / height; 
    const angle = yFraction * twistRadians;
    
    const cosAngle = Math.cos(angle);
    const sinAngle = Math.sin(angle);
    
    const x = vertex.x;
    const z = vertex.z;
    
    // Rotate around Y axis
    vertex.x = x * cosAngle - z * sinAngle;
    vertex.z = x * sinAngle + z * cosAngle;
    
    positionAttribute.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  
  // Recompute normals since the surface has curled
  geometry.computeVertexNormals();
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

// 8b. Filament Calibration Swatch (FilTracker Component)
export function createFilamentSwatch(w: number = 85, d: number = 54, h: number = 2): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const r = 4;
  const hw = w / 2;
  const hd = d / 2;

  // Outer rounded card
  shape.moveTo(-hw + r, -hd);
  shape.lineTo(hw - r, -hd);
  shape.quadraticCurveTo(hw, -hd, hw, -hd + r);
  shape.lineTo(hw, hd - r);
  shape.quadraticCurveTo(hw, hd, hw - r, hd);
  shape.lineTo(-hw + r, hd);
  shape.quadraticCurveTo(-hw, hd, -hw, hd - r);
  shape.lineTo(-hw, -hd + r);
  shape.quadraticCurveTo(-hw, -hd, -hw + r, -hd);

  // Top-left keyring hole (5mm)
  const hole = new THREE.Path();
  const holeRadius = 2.5;
  const holeX = -hw + 10;
  const holeY = hd - 10;
  hole.absarc(holeX, holeY, holeRadius, 0, Math.PI * 2, true);
  shape.holes.push(hole);

  const extrudeSettings = {
    depth: h,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.5,
    bevelThickness: 0.5,
  };

  const cardGeo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  cardGeo.rotateX(-Math.PI / 2);

  // Add 3 stepped thickness transparency windows (0.8mm, 1.4mm, 2.0mm)
  const evaluator = new Evaluator();
  evaluator.useGroups = false;

  const cardBrush = new Brush(cardGeo, new THREE.MeshStandardMaterial());

  // Step 1: 0.8mm window cutout
  const win1Geo = new THREE.BoxGeometry(15, 10, 10);
  const win1Brush = new Brush(win1Geo, new THREE.MeshStandardMaterial());
  win1Brush.position.set(-hw + 30, 0.4, 0);
  win1Brush.updateMatrixWorld();

  const step1 = evaluator.evaluate(cardBrush, win1Brush, SUBTRACTION);

  // Step 2: 1.4mm window cutout
  const win2Geo = new THREE.BoxGeometry(15, 10, 10);
  const win2Brush = new Brush(win2Geo, new THREE.MeshStandardMaterial());
  win2Brush.position.set(-hw + 50, 0.7, 0);
  win2Brush.updateMatrixWorld();

  const finalBrush = evaluator.evaluate(step1, win2Brush, SUBTRACTION);

  return finalBrush.geometry;
}

// 8c. Spool Rim Tag Clip (FilTracker Component)
export function createSpoolTag(w: number = 70, d: number = 25, h: number = 3): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const hw = w / 2;
  const hd = d / 2;
  const r = 3;

  shape.moveTo(-hw + r, -hd);
  shape.lineTo(hw - r, -hd);
  shape.quadraticCurveTo(hw, -hd, hw, -hd + r);
  shape.lineTo(hw, hd - r);
  shape.quadraticCurveTo(hw, hd, hw - r, hd);
  shape.lineTo(-hw + r, hd);
  shape.quadraticCurveTo(-hw, hd, -hw, hd - r);
  shape.lineTo(-hw, -hd + r);
  shape.quadraticCurveTo(-hw, -hd, -hw + r, -hd);

  // Center clip slot (to snap onto spool rim)
  const slot = new THREE.Path();
  slot.moveTo(-hw + 15, -hd + 6);
  slot.lineTo(hw - 15, -hd + 6);
  slot.lineTo(hw - 15, -hd + 10);
  slot.lineTo(-hw + 15, -hd + 10);
  slot.closePath();
  shape.holes.push(slot);

  const extrudeSettings = {
    depth: h,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.4,
    bevelThickness: 0.4,
  };

  const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  geo.rotateX(-Math.PI / 2);
  return geo;
}

interface ComputedOpBounds {
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  d: number;
}

function resolveCSGCoordinate(
  val: number | string | undefined,
  axis: 'x' | 'y' | 'z',
  currentDim: { w: number; h: number; d: number },
  history: ComputedOpBounds[]
): number {
  if (val === undefined || val === null) return 0;
  if (typeof val === 'number') return val;
  
  const strVal = String(val).trim().toLowerCase();
  
  // Try parsing numeric string
  const num = parseFloat(strVal);
  if (!isNaN(num) && strVal === String(num)) {
    return num;
  }

  // Parse pattern like "top_of(0)" or "center_of(1)"
  const match = strVal.match(/^([a-z_]+)\((\d+)\)$/);
  if (match) {
    const anchor = match[1];
    const refIndex = parseInt(match[2], 10);

    if (refIndex >= 0 && refIndex < history.length) {
      const ref = history[refIndex];

      switch (anchor) {
        case 'top_of':
          // Stack on top with 1mm overlap
          return ref.y + ref.h / 2 + currentDim.h / 2 - 1;
        case 'bottom_of':
          // Hang under bottom with 1mm overlap
          return ref.y - ref.h / 2 - currentDim.h / 2 + 1;
        case 'top_surface':
          // Center directly on top surface (ideal for top hole cutouts)
          return ref.y + ref.h / 2;
        case 'bottom_surface':
          return ref.y - ref.h / 2;
        case 'right_of':
          return ref.x + ref.w / 2 + currentDim.w / 2 - 1;
        case 'left_of':
          return ref.x - ref.w / 2 - currentDim.w / 2 + 1;
        case 'front_of':
          return ref.z + ref.d / 2 + currentDim.d / 2 - 1;
        case 'back_of':
          return ref.z - ref.d / 2 - currentDim.d / 2 + 1;
        case 'center_of':
          return ref[axis];
      }
    }
  }

  return !isNaN(num) ? num : 0;
}

// 9. True Constructive Solid Geometry Evaluator
function buildCSGGeometry(operations: CSGOperation[]): THREE.BufferGeometry {
  if (!operations || operations.length === 0) {
    return new THREE.BoxGeometry(20, 20, 20);
  }

  const evaluator = new Evaluator();
  evaluator.useGroups = false;
  let resultBrush: Brush | null = null;
  const history: ComputedOpBounds[] = [];

  for (const op of operations) {
    let geo: THREE.BufferGeometry;
    const w = op.width || (op.radius ? op.radius * 2 : 20);
    const d = op.depth || (op.radius ? op.radius * 2 : 20);
    const h = op.height || (op.radius ? op.radius * 2 : 20);
    const r = op.radius || Math.max(w, d) / 2;
    const wall = op.wallThickness || 2;
    const segments = op.segments || 64;

    if (op.shape === 'cylinder') geo = new THREE.CylinderGeometry(r, r, h, segments);
    else if (op.shape === 'sphere') geo = new THREE.SphereGeometry(r, segments, segments);
    else if (op.shape === 'cone') geo = new THREE.ConeGeometry(r, h, segments);
    else if (op.shape === 'torus') geo = new THREE.TorusGeometry(r, wall, segments, segments * 2);
    else if (op.shape === 'pyramid') geo = new THREE.CylinderGeometry(0, r, h, 4, 1, false, Math.PI / 4);
    else geo = new THREE.BoxGeometry(w, h, d);

    const currentDim = { w, h, d };
    const posX = resolveCSGCoordinate(op.x, 'x', currentDim, history);
    const posY = resolveCSGCoordinate(op.y, 'y', currentDim, history);
    const posZ = resolveCSGCoordinate(op.z, 'z', currentDim, history);

    history.push({ x: posX, y: posY, z: posZ, w, h, d });

    const material = new THREE.MeshStandardMaterial();
    const brush = new Brush(geo, material);
    brush.position.set(posX, posY, posZ);
    
    if (op.rotationX) brush.rotation.x = op.rotationX;
    if (op.rotationY) brush.rotation.y = op.rotationY;
    if (op.rotationZ) brush.rotation.z = op.rotationZ;

    brush.updateMatrixWorld();

    if (!resultBrush) {
      resultBrush = brush;
    } else {
      let operationType = ADDITION;
      if (op.op === 'subtract') operationType = SUBTRACTION;
      else if (op.op === 'intersect') operationType = INTERSECTION;

      resultBrush = evaluator.evaluate(resultBrush, brush, operationType);
    }
  }

  return resultBrush ? resultBrush.geometry : new THREE.BoxGeometry(20, 20, 20);
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
