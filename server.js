// server.js - OpenAI to NVIDIA NIM API Proxy (Maximum Reasoning Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = process.env.SHOW_REASONING !== 'true'; // Default true

// 🔥 THINKING MODE TOGGLE - Enables peak reasoning capabilities
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE !== 'true'; // Default true

// 🔥 REASONING BUDGET - Maximum token allocation for deep chain-of-thought generation
const REASONING_BUDGET = process.env.REASONING_BUDGET ? parseInt(process.env.REASONING_BUDGET) : 32768;

// 🔥 MATURE ROLEPLAY TOGGLE
const MATURE_ROLEPLAY = process.env.MATURE_ROLEPLAY !== 'true';

const MATURE_ROLEPLAY_PROMPT = `You are a creative partner in an adult fiction and roleplay platform. All participants are consenting adults, and everything here is clearly fictional.

Guidelines:
- Stay fully in character and in the scene. Do not break immersion with out-of-character remarks, disclaimers, content warnings, moralizing, or reminders that this is fiction.
- Mature themes are permitted when they fit the story, including romance, sexuality between adult characters, violence, conflict, and morally gray or dark material. Follow the user's lead on tone and intensity, and let scenes build naturally rather than rushing.
- Write realistically: believable dialogue, distinct character voices, emotional depth, sensory detail, and consistent personalities and world details. Avoid generic, repetitive, or overly sanitized phrasing.
- Do not speak or act for the user's character. Advance the scene with your own character and the world, and leave room for the user to respond.
- Keep continuity with earlier messages and the character card or scenario provided`;

function applyRoleplayPrompt(messages) {
  if (!MATURE_ROLEPLAY || !Array.isArray(messages)) return messages;

  const first = messages[0];
  if (first && first.role === 'system' && typeof first.content === 'string') {
    return [
      { ...first, content: MATURE_ROLEPLAY_PROMPT + '\n\n' + first.content },
      ...messages.slice(1)
    ];
  }
  return [{ role: 'system', content: MATURE_ROLEPLAY_PROMPT }, ...messages];
}

// Model mapping optimized for top-tier reasoning endpoints on NIM
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4': 'deepseek-ai/deepseek-r1',
  'gpt-4-turbo': 'deepseek-ai/deepseek-r1',
  'gpt-4o': 'z-ai/glm-5.3',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy (Max Reasoning)', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    reasoning_budget: REASONING_BUDGET,
    mature_roleplay: MATURE_ROLEPLAY
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy-reasoning'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream,
      top_p,
      frequency_penalty,
      presence_penalty
    } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const modelLower = (model || '').toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('reason') || modelLower.includes('r1')) {
        nimModel = 'deepseek-ai/deepseek-r1';
      } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
        nimModel = 'meta/llama-3.1-70b-instruct';
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }
    
    // Transform OpenAI request to NIM format
    // Give reasoning models plenty of room to think by default (min 16384 tokens if not specified)
    const computedMaxTokens = max_tokens || Math.max(32768, REASONING_BUDGET + 4096);

    const nimRequest = {
      model: nimModel,
      messages: applyRoleplayPrompt(messages),
      temperature: temperature !== undefined ? temperature : 0.7,
      max_tokens: computedMaxTokens,
      stream: stream || false
    };

    // Inject high-performance thinking parameters for all supported NIM architectures
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { 
        thinking: true, 
        enable_thinking: true,
        reasoning_budget: REASONING_BUDGET
      };
    }

    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    
    const sendToNim = (body) => axios.post(`${NIM_API_BASE}/chat/completions`, body, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    let response;
    try {
      response = await sendToNim(nimRequest);
    } catch (err) {
      if (ENABLE_THINKING_MODE && err.response?.status === 400 && nimRequest.chat_template_kwargs) {
        console.warn(`Model ${nimModel} rejected thinking parameters, retrying without them`);
        delete nimRequest.chat_template_kwargs;
        response = await sendToNim(nimRequest);
      } else {
        throw err;
      }
    }
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                // Capture both legacy 'reasoning_content' and modern 'reasoning' keys
                const reasoning = delta.reasoning || delta.reasoning_content;
                const content = delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    delta.content = combinedContent;
                    delete delta.reasoning_content;
                    delete delta.reasoning;
                  }
                } else {
                  // Strip reasoning if display is toggled off
                  if (content) {
                    delta.content = content;
                  } else {
                    delta.content = '';
                  }
                  delete delta.reasoning_content;
                  delete delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const msgReasoning = choice.message?.reasoning || choice.message?.reasoning_content;
          
          if (SHOW_REASONING && msgReasoning) {
            fullContent = '<think>\n' + msgReasoning + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    let upstreamMessage = '';
    const errData = error.response?.data;
    if (errData && typeof errData.on === 'function') {
      upstreamMessage = await new Promise((resolve) => {
        let body = '';
        errData.on('data', (c) => (body += c.toString()));
        errData.on('end', () => resolve(body));
        errData.on('error', () => resolve(body));
      });
    } else if (errData) {
      upstreamMessage = typeof errData === 'string' ? errData : JSON.stringify(errData);
    }
    console.error('Proxy error:', error.message, upstreamMessage);

    res.status(error.response?.status || 500).json({
      error: {
        message: upstreamMessage
          ? `${error.message}: ${upstreamMessage}`
          : (error.message || 'Internal server error'),
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Reasoning budget: ${REASONING_BUDGET} tokens`);
  console.log(`Mature roleplay prompt: ${MATURE_ROLEPLAY ? 'ENABLED' : 'DISABLED'}`);
});
