import { Hono } from 'hono';

type Bindings = {
  AI: any;
  ASSETS: {
    fetch: (request: Request) => Promise<Response>;
  };
};

const app = new Hono<{ Bindings: Bindings }>();

// API route for Cloudflare Workers AI 3D Model Generation
app.post('/api/generate', async (c) => {
  try {
    const body = await c.req.json();
    const messages = body?.messages || [];
    const userPrompt = body?.prompt || '';

    if (!userPrompt.trim() && messages.length === 0) {
      return c.json({ error: 'Prompt cannot be empty' }, 400);
    }

    // Intercept famous standard models (like 3DBenchy)
    const allText = (userPrompt + ' ' + messages.map((m: any) => m.content).join(' ')).toLowerCase();
    if (allText.includes('benchy')) {
      return c.json({
        success: true,
        source: 'famous-models-library',
        params: {
          type: 'external',
          externalUrl: '/models/3DBenchy.stl',
          title: '3DBenchy (Official)',
          description: 'The jolly 3D printing torture-test by CreativeTools.se.',
          rationale: 'The 3DBenchy is a highly complex mesh designed specifically to benchmark 3D printers. Generating this procedurally with CSG would ruin its precise overhangs and bridges, so I have loaded the pristine official STL from the Famous Models Library.',
          width: 60,
          depth: 31,
          height: 48,
          wallThickness: 0,
          holeDiameter: 0,
          roundedRadius: 0
        }
      });
    }

    // Check if Cloudflare Workers AI binding is available
    if (c.env && c.env.AI) {
      const systemPrompt = `You are an expert 3D CAD parametric modeling engineer and 3D printing specialist.
Analyze the user's 3D printing request and map it into a precise, valid JSON specification for 3D model generation.

Output strictly raw JSON with NO markdown formatting, NO backticks, NO extra text.

JSON Schema:
{
  "type": "sd_holder" | "cable_clip" | "keychain" | "wall_hook" | "phone_stand" | "hex_tray" | "csg",
  "title": "A short descriptive name (2-5 words)",
  "description": "A 1-sentence summary of the design and print advice",
  "rationale": "Chain of thought: Explain step-by-step how you will break down the object into primitive shapes, calculate their exact (x,y,z) coordinates using the stacking rules, and compose them.",
  "width": number in mm (range 15-200),
  "depth": number in mm (range 15-200),
  "height": number in mm (range 3-150),
  "wallThickness": number in mm (range 1.5-6),
  "holeDiameter": number in mm (range 0-12),
  "roundedRadius": number in mm (range 0-10),
  "textLabel": "optional text string",
  "twist": "optional number of degrees to twist the entire model (e.g. 360)",
  "operations": [
    {
      "op": "add" | "subtract" | "intersect",
      "shape": "box" | "cylinder" | "sphere" | "cone" | "torus" | "pyramid",
      "width": number, "depth": number, "height": number, "radius": number, "wallThickness": number,
      "segments": number, // Optional (e.g. 5, 6, 8) to make prisms/stars for twisting. Defaults to 64 for high-resolution smooth curves.
      "x": number or string, "y": number or string, "z": number or string, // Numbers OR Semantic Anchors: "top_of(0)", "bottom_of(0)", "top_surface(0)", "center_of(0)", "right_of(0)", "left_of(0)"
      "rotationX": number, "rotationY": number, "rotationZ": number // Rotations are in RADIANS! (e.g. 1.5708 for 90 deg, 3.14159 for 180 deg)
    }
  ]
}

Select the closest matching "type". If it is a generic object, box, case, enclosure, or anything requiring holes/combinations, YOU MUST USE "csg".
- "sd_holder": SD/MicroSD card organizers
- "cable_clip": cable management, wire holders
- "keychain": keychain tags
- "wall_hook": coat hooks, hangers
- "phone_stand": phone stands
- "hex_tray": honeycomb organizers
- "csg": ALL OTHER REQUESTS (enclosures, cases, pipes, boxes with holes, custom shapes, etc).

CRITICAL CSG MODELING RULES (If type is "csg"):
You are an expert CAD engineer. You build models by combining 3D primitives (Constructive Solid Geometry).
1. Semantic Anchors (USE THESE FOR EXACT ALIGNMENT!):
   Instead of guessing raw numbers for x, y, z, PREFER using semantic anchor strings referencing operation index N (0-indexed):
   - "top_surface(N)": Centers a shape/cutout directly on the top face of shape N (perfect for cutting top holes).
   - "top_of(N)": Stacks a shape directly on top of shape N with automatic 1mm CSG union overlap.
   - "bottom_of(N)": Attaches a shape directly under shape N.
   - "center_of(N)": Aligns center along the specified axis with shape N.
   - "right_of(N)" / "left_of(N)": Positions shape to the right/left of shape N.

2. Coordinate System: Center of every primitive is at its (x, y, z) position.
   - X-axis = Width (left/right)
   - Y-axis = Height (up/down). A shape of height H centered at y=0 extends from Y = -H/2 to +H/2.
   - Z-axis = Depth (front/back).

3. Hole & Cutout Rules:
   - Hole in TOP/BOTTOM (vertical): shape="cylinder", rotationX=0, y="top_surface(0)".
   - Hole in FRONT/BACK (horizontal depth): shape="cylinder", rotationX=1.5708 (90 degrees), z="center_of(0)".
   - Hole in LEFT/RIGHT (horizontal width): shape="cylinder", rotationZ=1.5708 (90 degrees), x="center_of(0)".
   - Make subtracting cylinders significantly longer than the wall they are piercing to guarantee a clean cut!

4. FEW-SHOT EXAMPLES:
Study these examples to learn the correct JSON structure using semantic anchors.

Example 1: "A rectangular 3d box with a centered top hole"
{
  "type": "csg",
  "title": "Box with Top Hole",
  "description": "A 50x50x50 box with a 20mm diameter cylinder subtracted from the top surface.",
  "rationale": "Chain of thought: Base box is 50x50x50 at (0,0,0). For the top hole, I subtract a cylinder with radius 10 and height 30. Using y='top_surface(0)' centers the cut directly on the box top, while x='center_of(0)' and z='center_of(0)' keep it perfectly centered.",
  "width": 50, "depth": 50, "height": 50, "wallThickness": 2, "holeDiameter": 20, "roundedRadius": 0,
  "operations": [
    { "op": "add", "shape": "box", "width": 50, "height": 50, "depth": 50, "x": 0, "y": 0, "z": 0 },
    { "op": "subtract", "shape": "cylinder", "radius": 10, "height": 30, "x": "center_of(0)", "y": "top_surface(0)", "z": "center_of(0)" }
  ]
}

Example 2: "Stacked Cylinder on Box"
{
  "type": "csg",
  "title": "Pedestal Tower",
  "description": "A cylinder stacked on top of a square base.",
  "rationale": "Base box 60x60x20 at y=0. Using y='top_of(0)' for the top cylinder automatically stacks it flush on the box top with 1mm overlap for a clean CSG union.",
  "width": 60, "depth": 60, "height": 70, "wallThickness": 2, "holeDiameter": 0, "roundedRadius": 0,
  "operations": [
    { "op": "add", "shape": "box", "width": 60, "height": 20, "depth": 60, "x": 0, "y": 0, "z": 0 },
    { "op": "add", "shape": "cylinder", "radius": 15, "height": 50, "x": "center_of(0)", "y": "top_of(0)", "z": "center_of(0)" }
  ]
}

Example 3: "Hollow Box Case"
{
  "type": "csg",
  "title": "Hollow Enclosure",
  "description": "A hollow container with 2mm walls.",
  "rationale": "Solid box 50x50x50 at y=0. Subtract inner box 46x50x46 at y=2 so it leaves a 2mm solid floor.",
  "width": 50, "depth": 50, "height": 50, "wallThickness": 2, "holeDiameter": 0, "roundedRadius": 0,
  "operations": [
    { "op": "add", "shape": "box", "width": 50, "height": 50, "depth": 50, "x": 0, "y": 0, "z": 0 },
    { "op": "subtract", "shape": "box", "width": 46, "height": 50, "depth": 46, "x": 0, "y": 2, "z": 0 }
  ]
}

5. Twist Modifier: You can apply a global "twist" (in degrees) to the final model. Example: "twist": 360. If twisting a cylinder, MUST set "segments" to 5, 6, or 8 so the twist is visible!`;

      const apiMessages = [{ role: 'system', content: systemPrompt }];
      if (messages.length > 0) {
        apiMessages.push(...messages);
      } else {
        apiMessages.push({ role: 'user', content: userPrompt });
      }

      const aiResponse = await c.env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
        messages: apiMessages,
        temperature: 0.2,
        max_tokens: 500,
      });

      let rawText = '';
      if (typeof aiResponse === 'string') {
        rawText = aiResponse;
      } else if (aiResponse?.response) {
        rawText = typeof aiResponse.response === 'string' ? aiResponse.response : JSON.stringify(aiResponse.response);
      } else {
        throw new Error("UNEXPECTED_AI_RESPONSE: " + JSON.stringify(aiResponse));
      }
      rawText = String(rawText);

      // Clean up markdown wrapping if present
      const jsonMatch = rawText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return c.json({ success: true, params: parsed, source: 'cloudflare-ai' });
      } else {
        throw new Error("AI did not return JSON: " + rawText);
      }
    }

    // Fallback heuristic engine if AI binding is offline or running locally
    const fallbackParams = parsePromptFallback(userPrompt);
    return c.json({ success: true, params: fallbackParams, source: 'local-heuristic' });

  } catch (error: any) {
    console.error('AI Generation Error:', error);
    const body = await c.req.json().catch(() => ({ prompt: '' }));
    const fallbackParams = parsePromptFallback(body?.prompt || '');
    return c.json({ success: true, params: fallbackParams, source: 'fallback-error', error: error.message || String(error) });
  }
});

// Heuristic prompt parser fallback
function parsePromptFallback(prompt: string) {
  const lower = prompt.toLowerCase();
  
  if (lower.includes('swatch') || lower.includes('filament card') || lower.includes('sample')) {
    return {
      type: 'swatch',
      title: 'Filament Calibration Swatch',
      description: 'Precision filament sample card with 3 stepped transparency windows (0.8mm, 1.4mm, 2.0mm).',
      width: 85,
      depth: 54,
      height: 2,
      wallThickness: 2,
      holeDiameter: 5,
      roundedRadius: 4,
    };
  }

  if (lower.includes('spool tag') || lower.includes('spool clip') || lower.includes('label tag')) {
    return {
      type: 'spool_tag',
      title: 'Spool Rim Label Tag',
      description: 'Snap-fit spool rim clip for organizing and labeling filament inventory.',
      width: 70,
      depth: 25,
      height: 3,
      wallThickness: 2,
      holeDiameter: 0,
      roundedRadius: 3,
    };
  }
  if (lower.includes('sd') || lower.includes('micro sd') || lower.includes('memory card')) {
    return {
      type: 'sd_holder',
      title: 'SD & MicroSD Card Organizer',
      description: 'Structured desk tray with precision slots for SD cards.',
      width: 70,
      depth: 45,
      height: 25,
      wallThickness: 2.5,
      holeDiameter: 0,
      roundedRadius: 3,
    };
  }

  if (lower.includes('cable') || lower.includes('wire') || lower.includes('cord')) {
    return {
      type: 'cable_clip',
      title: 'Desk Cable Clip',
      description: 'Snap-fit cable clip designed to hold desktop charging cords.',
      width: 40,
      depth: 25,
      height: 18,
      wallThickness: 3,
      holeDiameter: 4,
      roundedRadius: 3,
    };
  }

  if (lower.includes('key') || lower.includes('tag') || lower.includes('nameplate')) {
    const textWords = prompt.split(' ').filter(w => w.length > 2);
    const textLabel = textWords[0] ? textWords[0].toUpperCase() : 'MKW';
    return {
      type: 'keychain',
      title: 'Custom Keychain Tag',
      description: 'Personalized keychain tag with reinforced ring mounting hole.',
      width: 60,
      depth: 28,
      height: 4,
      wallThickness: 2,
      holeDiameter: 5,
      roundedRadius: 5,
      textLabel,
    };
  }

  if (lower.includes('hook') || lower.includes('headphone') || lower.includes('hang') || lower.includes('mount')) {
    return {
      type: 'wall_hook',
      title: 'Wall Mount Utility Hook',
      description: 'Heavy duty wall-mountable hook with countersunk screw hole.',
      width: 30,
      depth: 55,
      height: 65,
      wallThickness: 4,
      holeDiameter: 4.5,
      roundedRadius: 4,
    };
  }

  if (lower.includes('phone') || lower.includes('stand') || lower.includes('tablet') || lower.includes('desk')) {
    return {
      type: 'phone_stand',
      title: 'Angled Smartphone Stand',
      description: '60-degree desktop phone stand with cable pass-through channel.',
      width: 65,
      depth: 80,
      height: 75,
      wallThickness: 3.5,
      holeDiameter: 0,
      roundedRadius: 4,
    };
  }

  if (lower.includes('hex') || lower.includes('screw') || lower.includes('honeycomb') || lower.includes('hardware')) {
    return {
      type: 'hex_tray',
      title: 'Hexagonal Catch-All Tray',
      description: 'Interlocking honeycomb hex tray for desktop hardware.',
      width: 85,
      depth: 85,
      height: 28,
      wallThickness: 3,
      holeDiameter: 0,
      roundedRadius: 0,
    };
  }

  // Primitive Shape Fallbacks
  if (lower.includes('tube') || lower.includes('cup') || lower.includes('vase') || lower.includes('pipe') || lower.includes('mug')) {
    return {
      type: 'csg',
      title: 'Custom Hollow Tube/Cup',
      description: 'A cylindrical geometry with a hollowed out center built using CSG.',
      width: 60, depth: 60, height: 80, wallThickness: 3, holeDiameter: 0, roundedRadius: 0,
      operations: [
        { op: 'add', shape: 'cylinder', radius: 30, height: 80, x: 0, y: 40, z: 0 },
        { op: 'subtract', shape: 'cylinder', radius: 27, height: 80, x: 0, y: 43, z: 0 }
      ]
    };
  }

  if (lower.includes('cylinder') || lower.includes('rod') || lower.includes('peg')) {
    return {
      type: 'custom',
      baseShape: 'cylinder',
      isHollow: false,
      title: 'Solid Cylinder',
      description: 'A solid cylindrical object.',
      width: 60, depth: 60, height: 80, wallThickness: 2, holeDiameter: 0, roundedRadius: 0,
    };
  }

  if (lower.includes('sphere') || lower.includes('ball') || lower.includes('orb')) {
    return {
      type: 'custom',
      baseShape: 'sphere',
      isHollow: false,
      title: 'Solid Sphere',
      description: 'A solid spherical object.',
      width: 50, depth: 50, height: 50, wallThickness: 2, holeDiameter: 0, roundedRadius: 0,
    };
  }

  if (lower.includes('ring') || lower.includes('torus') || lower.includes('donut')) {
    return {
      type: 'custom',
      baseShape: 'torus',
      isHollow: false,
      title: 'Torus Ring',
      description: 'A toroidal ring structure.',
      width: 60, depth: 60, height: 10, wallThickness: 10, holeDiameter: 0, roundedRadius: 0,
    };
  }

  if (lower.includes('cone') || lower.includes('pyramid') || lower.includes('spike')) {
    return {
      type: 'custom',
      baseShape: lower.includes('pyramid') ? 'pyramid' : 'cone',
      isHollow: false,
      title: 'Conical Object',
      description: 'A conical or pyramidal structure.',
      width: 50, depth: 50, height: 70, wallThickness: 2, holeDiameter: 0, roundedRadius: 0,
    };
  }

  // Default box bin container
  return {
    type: 'box',
    title: 'Custom Storage Container',
    description: 'Hollow parametric box container with chamfered edges.',
    width: 65,
    depth: 45,
    height: 35,
    wallThickness: 2.5,
    holeDiameter: 0,
    roundedRadius: 5,
  };
}

export default {
  async fetch(request: Request, env: Bindings, ctx: any): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      return app.fetch(request, env, ctx);
    }
    // Serve static assets
    return env.ASSETS.fetch(request);
  },
};
