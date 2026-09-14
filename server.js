// server.js
const express = require('express');
const app = express();

app.use(express.json({ limit: '10mb' }));

const NVIDIA_BASE = 'https://integrate.api.nvidia.com/v1';

// Map short/friendly names (what you type into JanitorAI's model field)
// to the actual NVIDIA model IDs. Add/edit entries as needed — find exact
// IDs at https://build.nvidia.com or via GET /v1/models on this proxy.
const MODEL_MAP = {
  'kimi-k3': 'moonshotai/kimi-k3',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'llama-3.1-70b': 'nvidia/llama-3.1-70b-instruct',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'deepseek-r1-distill-llama-8b': 'deepseek-ai/deepseek-r1-distill-llama-8b',
};

function resolveModel(name) {
  return MODEL_MAP[name] || name; // fall through to raw name if not mapped
}

// Visit https://your-deployed-url.com/ in a browser to confirm the server
// is alive and that your API key env var is actually set.
app.get('/', (req, res) => {
  res.json({ status: 'ok', hasKey: !!process.env.NIM_API_KEY });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const body = { ...req.body, model: resolveModel(req.body.model) };

    const nvidiaRes = await fetch(`${NVIDIA_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.NIM_API_KEY}`,
      },
      body: JSON.stringify(body),
    });

    // If NVIDIA itself rejected the request, log the exact reason
    if (!nvidiaRes.ok && !body.stream) {
      const errText = await nvidiaRes.text();
      console.error('NVIDIA API error:', nvidiaRes.status, errText);
      return res.status(nvidiaRes.status).send(errText);
    }

    if (body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      for await (const chunk of nvidiaRes.body) {
        res.write(chunk);
      }
      res.end();
    } else {
      const data = await nvidiaRes.json();
      res.status(nvidiaRes.status).json(data);
    }
  } catch (err) {
    console.error('Proxy error:', err);
    res.status(500).json({ error: 'Proxy error', detail: err.message });
  }
});

// Visit this to see your configured short-name -> NVIDIA model ID mapping
app.get('/v1/model-map', (req, res) => {
  res.json(MODEL_MAP);
});

app.get('/v1/models', async (req, res) => {
  try {
    const r = await fetch(`${NVIDIA_BASE}/models`, {
      headers: { 'Authorization': `Bearer ${process.env.NIM_API_KEY}` },
    });
    const data = await r.json();
    res.status(r.status).json(data);
  } catch (err) {
    console.error('Models fetch error:', err);
    res.status(500).json({ error: 'Models fetch error', detail: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Proxy running on port ${PORT}`));
