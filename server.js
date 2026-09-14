// server.js
const express = require('express');
const app = express();

app.use(express.json({ limit: '10mb' }));

const NVIDIA_BASE = 'https://integrate.api.nvidia.com/v1';

// Visit https://your-deployed-url.com/ in a browser to confirm the server
// is alive and that your API key env var is actually set.
app.get('/', (req, res) => {
  res.json({ status: 'ok', hasKey: !!process.env.NIM_API_KEY });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const nvidiaRes = await fetch(`${NVIDIA_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.NIM_API_KEY}`,
      },
      body: JSON.stringify(req.body),
    });

    // If NVIDIA itself rejected the request, log the exact reason
    if (!nvidiaRes.ok && !req.body.stream) {
      const errText = await nvidiaRes.text();
      console.error('NVIDIA API error:', nvidiaRes.status, errText);
      return res.status(nvidiaRes.status).send(errText);
    }

    if (req.body.stream) {
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
