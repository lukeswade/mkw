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
  "rationale": "Chain of thought: Explain step-by-step how you will break down the object into primitive shapes and compose it.",
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
      "segments": number, // Optional (e.g. 5, 6, 8) to make prisms/stars for twisting
      "x": number, "y": number, "z": number,
      "rotationX": number, "rotationY": number, "rotationZ": number
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
1. Coordinate System (Three.js): Center is (0,0,0). 
   - X-axis = Width (left/right)
   - Y-axis = Height (up/down). A box of height H goes from Y = -H/2 to +H/2.
   - Z-axis = Depth (front/back).
2. Primitives:
   - "box": requires 'width' (X), 'height' (Y), 'depth' (Z).
   - "cylinder": requires 'radius' and 'height' (Y). Default orientation stands vertically (along Y axis).
3. Orienting Cylinders for Holes:
   - Hole in TOP/BOTTOM (vertical): shape="cylinder", rotationX=0.
   - Hole in FRONT/BACK (horizontal depth): shape="cylinder", rotationX=1.5708.
   - Hole in LEFT/RIGHT (horizontal width): shape="cylinder", rotationZ=1.5708.
4. Hollow Enclosures & Trays:
   - CLOSED HOLLOW BOX (has a floor and a lid): Add outer box (w, h, d). Subtract inner box with (w - wallThickness*2, h - wallThickness*2, d - wallThickness*2). Set inner box 'y': 0. This leaves solid walls on all sides, including top and bottom!
   - OPEN TRAY (no lid): Add outer box (w, h, d). Subtract inner box (w - wallThickness*2, h, d - wallThickness*2). Shift inner box 'y' up by 'wallThickness' (e.g. 'y': 2) so it cuts through the top but leaves a floor!
5. Drilling Holes:
   - To make a hole in the "top center", subtract a vertical cylinder at 'x': 0, 'z': 0, and 'y' shifted to the top face (e.g. 'y': height/2). 
   - To make a hole in the front face, subtract a horizontal cylinder (rotationX=1.5708) at 'z': depth/2.
   - Make subtracting hole cylinders longer than the wall they are piercing to guarantee a clean cut!
6. Complex Objects & Multi-Part Assembly: You are not limited to just boxes! You can build cars, boats (Benchy), buildings, or robots by assembling multiple primitives. Use 'add' to combine hulls, cabins, noses, and wheels. Example: A boat has a hull (box), a bow (pyramid), a cabin (smaller box), and a smokestack (cylinder). Think creatively!
7. Twist Modifier: You can apply a global "twist" (in degrees) to the final model. Use this for generating spirals, frozen yogurt twirls, screw threads, or organic shapes. Example: "twist": 720. If twisting a cylinder, MUST set "segments" to 5, 6, or 8 so the twist is visible (a perfectly round cylinder looks the same when twisted)!`;

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
