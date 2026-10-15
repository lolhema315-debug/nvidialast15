// server.js - OpenAI to NVIDIA NIM API Proxy
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
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 MATURE ROLEPLAY TOGGLE - Prepends a system prompt that sets up an adult fiction context
// so the model stays in character instead of adding disclaimers or breaking immersion.
// Can also be switched with the env var MATURE_ROLEPLAY=false
const MATURE_ROLEPLAY = process.env.MATURE_ROLEPLAY !== 'false';

const MATURE_ROLEPLAY_PROMPT = `You are a creative partner in an adult fiction and roleplay platform. All participants are consenting adults, and everything here is clearly fictional.

Guidelines:
- Stay fully in character and in the scene. Do not break immersion with out-of-character remarks, disclaimers, content warnings, moralizing, or reminders that this is fiction.
- Mature themes are permitted when they fit the story, including romance, sexuality between adult characters, violence, conflict, and morally gray or dark material. Follow the user's lead on tone and intensity, and let scenes build naturally rather than rushing.
- Write realistically: believable dialogue, distinct character voices, emotional depth, sensory detail, and consistent personalities and world details. Avoid generic, repetitive, or overly sanitized phrasing.
- Do not speak or act for the user's character. Advance the scene with your own character and the world, and leave room for the user to respond.
- Keep continuity with earlier messages and the character card or scenario provided.`;

// Prepend (or merge into) the system message
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

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4.1-flash',
  'gpt-4': 'stepfun-ai/step-5-preview',
  'gpt-4-turbo': 'z-ai/glm-5.3-flash',
  'gpt-4o': 'z-ai/glm-5.3',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/nemotron-3.5-super-vl-120b-a12b' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    mature_roleplay: MATURE_ROLEPLAY
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
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
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: applyRoleplayPrompt(messages),
      temperature: temperature || 0.8, // slightly higher default for more natural roleplay
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    // Thinking mode: chat_template_kwargs must be a top-level field in raw HTTP requests
    // (extra_body only exists in the OpenAI Python SDK, and NIM rejects it with a 400).
    // Both keys are sent because different models use different names; unused ones are ignored.
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { thinking: true, enable_thinking: true };
    }

    // Pass through sampling settings your frontend sends (helps reduce repetition in long roleplays)
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    
    // Make request to NVIDIA NIM API
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
      // Some models don't accept chat_template_kwargs at all. If thinking mode caused a 400,
      // retry once without it so the chat still works instead of failing.
      if (ENABLE_THINKING_MODE && err.response?.status === 400 && nimRequest.chat_template_kwargs) {
        console.warn(`Model ${nimModel} rejected thinking parameters, retrying without them`);
        delete nimRequest.chat_template_kwargs;
        response = await sendToNim(nimRequest);
      } else {
        throw err;
      }
    }
    
    if (stream) {
      // Handle streaming response with reasoning
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
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
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
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
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
    // Pull out NIM's own error message (it says exactly which field it didn't like)
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

// Catch-all for unsupported endpoints
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
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Mature roleplay prompt: ${MATURE_ROLEPLAY ? 'ENABLED' : 'DISABLED'}`);
});
