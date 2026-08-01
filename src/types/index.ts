export type ModelType = 
  | 'box' 
  | 'sd_holder' 
  | 'cable_clip' 
  | 'keychain' 
  | 'wall_hook' 
  | 'phone_stand' 
  | 'hex_tray' 
  | 'custom'
  | 'external'
  | 'csg';

export interface CSGOperation {
  op: 'add' | 'subtract' | 'intersect';
  shape: 'box' | 'cylinder' | 'sphere' | 'cone' | 'torus' | 'pyramid';
  width?: number;
  depth?: number;
  height?: number;
  radius?: number;
  wallThickness?: number;
  x: number;
  y: number;
  z: number;
  rotationX?: number;
  rotationY?: number;
  rotationZ?: number;
  twist?: number;
  segments?: number;
}

export interface ModelParams {
  type: ModelType;
  title: string;
  description: string;
  rationale?: string; // AI's internal thought process for the design
  width: number; // in mm
  depth: number; // in mm
  height: number; // in mm
  wallThickness: number; // in mm
  holeDiameter: number; // in mm
  roundedRadius: number; // in mm
  textLabel?: string;
  subType?: string;
  baseShape?: 'box' | 'cylinder' | 'sphere' | 'cone' | 'torus' | 'pyramid';
  isHollow?: boolean;
  customDetails?: {
    slotsCount?: number;
    angle?: number;
    pattern?: string;
  };
  twist?: number; // Global twist applied to the final shape (degrees)
  externalUrl?: string; // URL for loading an external STL model (like Benchy)
  operations?: CSGOperation[];
}

export interface PrintAnalytics {
  volumeCm3: number;
  weightGrams: number;
  estimatedTimeMin: number;
  filamentLengthMeters: number;
  bedCompatibility: {
    bambuLab: boolean; // 256x256
    ender3: boolean;   // 220x220
    mini: boolean;     // 180x180
  };
}

export type MaterialType = 'PLA' | 'PETG' | 'TPU' | 'ABS';

export interface PresetIdea {
  id: string;
  title: string;
  prompt: string;
  category: string;
  icon: string;
  params: ModelParams;
}
