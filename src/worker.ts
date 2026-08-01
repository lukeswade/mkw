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
    const userPrompt = body?.prompt || '';

    if (!userPrompt.trim()) {
      return c.json({ error: 'Prompt cannot be empty' }, 400);
    }

    // Check if Cloudflare Workers AI binding is available
    if (c.env && c.env.AI) {
      const systemPrompt = `You are an expert 3D CAD parametric modeling engineer and 3D printing specialist.
Analyze the user's 3D printing request and map it into a precise, valid JSON specification for 3D model generation.

Output strictly raw JSON with NO markdown formatting, NO backticks, NO extra text.

JSON Schema:
{
  "type": "box" | "sd_holder" | "cable_clip" | "keychain" | "wall_hook" | "phone_stand" | "hex_tray" | "custom",
  "title": "A short descriptive name (2-5 words)",
  "description": "A 1-sentence summary of the design and print advice",
  "width": number in mm (range 15-200),
  "depth": number in mm (range 15-200),
  "height": number in mm (range 3-150),
  "wallThickness": number in mm (range 1.5-6),
  "holeDiameter": number in mm (range 0-12),
  "roundedRadius": number in mm (range 0-10),
  "textLabel": "optional text string for tags/keychains",
  "baseShape": "box" | "cylinder" | "sphere" | "cone" | "torus" | "pyramid",
  "isHollow": boolean
}

Select the closest matching "type":
- "sd_holder" for SD/MicroSD card trays or organizers
- "cable_clip" for cable management, wire holders, clips
- "keychain" for keychain tags, nameplates, zipper pulls
- "wall_hook" for coat hooks, headphone wall mounts, tool hangers
- "phone_stand" for phone, tablet, or desk device stands
- "hex_tray" for hex trays, screw catch-alls, honeycomb organizers
- "box" for boxes, trays, desk bins, hollow containers
- "custom" for any other solid or parametric 3D object.

CRITICAL: If "type" is "custom", you MUST select an appropriate "baseShape" (e.g. cylinder for cups/tubes, sphere for balls, torus for rings, box for bricks) and set "isHollow" (true for cups/tubes/vases, false for solid objects).`;

      const aiResponse = await c.env.AI.run('@cf/meta/llama-3.3-70b-instruct', {
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.2,
        max_tokens: 500,
      });

      let rawText = '';
      if (typeof aiResponse === 'string') {
        rawText = aiResponse;
      } else if (aiResponse?.response) {
        rawText = aiResponse.response;
      } else {
        rawText = JSON.stringify(aiResponse);
      }

      // Clean up markdown wrapping if present
      const jsonMatch = rawText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return c.json({ success: true, params: parsed, source: 'cloudflare-ai' });
      }
    }

    // Fallback heuristic engine if AI binding is offline or running locally
    const fallbackParams = parsePromptFallback(userPrompt);
    return c.json({ success: true, params: fallbackParams, source: 'local-heuristic' });

  } catch (error: any) {
    console.error('AI Generation Error:', error);
    const body = await c.req.json().catch(() => ({ prompt: '' }));
    const fallbackParams = parsePromptFallback(body?.prompt || '');
    return c.json({ success: true, params: fallbackParams, source: 'fallback-error' });
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
  if (lower.includes('cylinder') || lower.includes('tube') || lower.includes('cup') || lower.includes('vase')) {
    return {
      type: 'custom',
      baseShape: 'cylinder',
      isHollow: lower.includes('tube') || lower.includes('cup') || lower.includes('vase'),
      title: 'Custom Cylindrical Object',
      description: 'A cylindrical geometry generated based on your prompt.',
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
