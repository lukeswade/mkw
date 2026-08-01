import { PresetIdea } from '../types';

export const PRESET_IDEAS: PresetIdea[] = [
  {
    id: 'sd_holder',
    title: 'SD & MicroSD Card Organizer',
    prompt: 'A compact desk organizer tray designed to store 10 SD cards and 8 MicroSD cards with clean dividers.',
    category: 'Desk & Office',
    icon: 'Grid',
    params: {
      type: 'sd_holder',
      title: 'SD Card Organizer Tray',
      description: 'Holds 10 standard SD cards vertically with 2.5mm walls.',
      width: 75,
      depth: 50,
      height: 25,
      wallThickness: 2.5,
      holeDiameter: 0,
      roundedRadius: 4,
    }
  },
  {
    id: 'wall_hook',
    title: 'Headphone Wall Mount Hook',
    prompt: 'A heavy duty curved wall hook for over-ear headphones with a countersunk M4 screw mounting hole.',
    category: 'Home & Storage',
    icon: 'Anchor',
    params: {
      type: 'wall_hook',
      title: 'Headphone Wall Mount Hook',
      description: 'Ergonomic headphone hook with countersunk screw hole.',
      width: 35,
      depth: 60,
      height: 70,
      wallThickness: 4,
      holeDiameter: 4.5,
      roundedRadius: 5,
    }
  },
  {
    id: 'cable_clip',
    title: 'Desk Cable Management Clip',
    prompt: 'A snap-fit cable clip for holding USB-C and Lightning charging cables neatly along the side of a desk.',
    category: 'Cable Tech',
    icon: 'Paperclip',
    params: {
      type: 'cable_clip',
      title: 'Snap Cable Management Clip',
      description: 'Flex-clip for securing USB & power cords.',
      width: 45,
      depth: 25,
      height: 20,
      wallThickness: 3,
      holeDiameter: 4,
      roundedRadius: 3,
    }
  },
  {
    id: 'keychain',
    title: 'Custom Text Keychain Tag',
    prompt: 'A rounded rectangular keychain tag with a 5mm ring hole and raised outer border labeled MKW.',
    category: 'Personal Accessories',
    icon: 'Key',
    params: {
      type: 'keychain',
      title: 'MKW Custom Keychain Tag',
      description: 'Personalized keychain with reinforced ring hole.',
      width: 65,
      depth: 30,
      height: 4,
      wallThickness: 2,
      holeDiameter: 5,
      roundedRadius: 6,
      textLabel: 'MATT K WADE',
    }
  },
  {
    id: 'phone_stand',
    title: 'Desktop Phone & Tablet Stand',
    prompt: 'An angled minimalist desktop stand for smartphones with a bottom notch for charging cable pass-through.',
    category: 'Workspace',
    icon: 'Smartphone',
    params: {
      type: 'phone_stand',
      title: 'Minimalist Phone Stand',
      description: '60-degree angle stand compatible with modern iPhones & Android devices.',
      width: 70,
      depth: 85,
      height: 80,
      wallThickness: 3.5,
      holeDiameter: 0,
      roundedRadius: 4,
    }
  },
  {
    id: 'hex_tray',
    title: 'Modular Hexagon Gear Tray',
    prompt: 'A honeycomb hexagonal catch-all tray for screws, keys, and small hardware components.',
    category: 'Organization',
    icon: 'Hexagon',
    params: {
      type: 'hex_tray',
      title: 'Hexagon Catch-All Tray',
      description: 'Interlocking honeycomb organizer tray.',
      width: 90,
      depth: 90,
      height: 30,
      wallThickness: 3,
      holeDiameter: 0,
      roundedRadius: 0,
    }
  }
];
