// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry forwards each request
// to a provider on that provider's own wire — OpenAI, Anthropic or Gemini — and translates the answer
// back into the wire the client used.
//
// The translation is faithful by construction: the provider echoes what it parsed (system text, user
// text, tool names, max_tokens, stop sequences, image, JSON mode), and the client compares that echo
// with what it sent, so every echoed field is carried across unchanged, in the shape the receiving wire
// parses it from. A field the translation rewrites or drops loses the request.
//
// Stop reasons are mapped by family (stop/end_turn/STOP, length/max_tokens/MAX_TOKENS,
// tool_calls/tool_use, content_filter/refusal/SAFETY), so the client always hears the family its
// provider answered with. Usage counts are copied from the provider's answer, never recomputed, so the
// client's usage check sees the provider's own numbers.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The 13-character pieces the simulated providers split streamed text into; reused for parity. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One stop family: 0 stop, 1 length, 2 tool use, 3 refusal. Null when the reason names none. */
export function stopFamilyOf(reason: unknown): number | null {
  if (typeof reason !== 'string') return null;
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** A provider's stop reason as an OpenAI finish_reason, by family (stop when the family is unknown). */
export function openAIFinishOf(providerStop: unknown): string {
  const family = stopFamilyOf(providerStop);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** A provider's stop reason as an Anthropic stop_reason, by family (end_turn when unknown). */
export function anthropicStopOf(providerStop: unknown): string {
  const family = stopFamilyOf(providerStop);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  return anthropicStopOf(finish);
}

/** A number, or 0 when the value is not one. */
export function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The usage of a provider answer, in {input, output} tokens; null when it carries none. */
export interface Usage {
  input: number;
  output: number;
}

/** Usage from an OpenAI-shaped answer body. */
export function openAIUsageOf(body: Json): Usage | null {
  const usage = isRecord(body.usage) ? body.usage : null;
  if (usage === null) return null;
  return { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) };
}

/** Usage from an Anthropic-shaped answer body. */
export function anthropicUsageOf(body: Json): Usage | null {
  const usage = isRecord(body.usage) ? body.usage : null;
  if (usage === null) return null;
  return { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) };
}

/** Usage from a Gemini-shaped answer body (usageMetadata). */
export function geminiUsageOf(body: Json): Usage | null {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  if (meta === null) return null;
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** The assistant text of an OpenAI-shaped answer body. */
export function openAITextOf(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  return textOf(message.content);
}

/** The assistant text of an Anthropic-shaped answer body (its text blocks joined). */
export function anthropicTextOf(body: Json): string {
  if (!Array.isArray(body.content)) return '';
  const parts: string[] = [];
  for (const block of body.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** The model text of a Gemini-shaped answer body (its candidates' parts joined). */
export function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const parts: string[] = [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
  }
  return parts.join('');
}

/** The finish reason of a Gemini-shaped answer body (its first candidate's), or null. */
export function geminiStopOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  return first !== null && typeof first.finishReason === 'string' ? first.finishReason : null;
}

/** The tool calls of an Anthropic-shaped answer body, as OpenAI tool_calls; [] when it holds none. */
function anthropicToolCallsOf(body: Json): Json[] {
  const out: Json[] = [];
  if (!Array.isArray(body.content)) return out;
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    out.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  return out;
}

/** The tool calls of a Gemini-shaped answer body, as OpenAI tool_calls; [] when it holds none. */
function geminiToolCallsOf(body: Json): Json[] {
  const out: Json[] = [];
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (!isRecord(part) || !isRecord(part.functionCall)) continue;
      const call = part.functionCall;
      out.push({
        id: `call_${out.length}`,
        type: 'function',
        function: { name: typeof call.name === 'string' ? call.name : '', arguments: JSON.stringify(isRecord(call.args) ? call.args : {}) },
      });
    }
  }
  return out;
}

/** A Chat Completions answer object from plain text, a finish reason and the provider's usage. */
export function openAICompletion(model: string, text: string, finishReason: string, usage: Usage | null, toolCalls: Json[] = []): Json {
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason, logprobs: null }],
    usage:
      usage === null
        ? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        : { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Messages answer object from plain text, a stop reason and the provider's usage. */
export function anthropicMessage(model: string, text: string, stopReason: string, usage: Usage | null, toolCalls: Json[] = []): Json {
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of toolCalls) {
    if (!isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: usage === null ? { input_tokens: 0, output_tokens: 0 } : { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as a Chat Completions object (text, tool calls, stop, usage). */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const usage = anthropicUsageOf(body);
  const text = anthropicTextOf(body);
  const calls = anthropicToolCallsOf(body);
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  return openAICompletion(model, text, calls.length > 0 ? 'tool_calls' : openAIFinishOf(stop), usage, calls);
}

/** A Gemini answer as a Chat Completions object (text, tool calls, stop, usage). */
export function geminiToOpenAIAnswer(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  const text = geminiTextOf(body);
  const calls = geminiToolCallsOf(body);
  const stop = geminiStopOf(body);
  return openAICompletion(model, text, calls.length > 0 ? 'tool_calls' : openAIFinishOf(stop), usage, calls);
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Gemini answer as an Anthropic Messages object (text, tool calls, stop, usage). */
export function geminiToAnthropicAnswer(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  const text = geminiTextOf(body);
  const calls = geminiToolCallsOf(body);
  const stop = geminiStopOf(body);
  return anthropicMessage(model, text, calls.length > 0 ? 'tool_use' : anthropicStopOf(stop), usage, calls);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system prompt, image_url parts become image blocks, assistant tool_calls become
 * tool_use blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and
 * the output cap, temperature, top_p, stop sequences and streaming pass through. JSON mode has no
 * Anthropic spelling, so a JSON-mode request is never sent down this path (see wireCarriesJson).
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : textOf(content) }],
      });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Gemini content part for a text string, or null when it holds no text. */
function geminiTextPart(text: string): Json | null {
  return text === '' ? null : { text };
}

/** One Gemini content part for an image URL: inline base64 data, or a file reference. */
function geminiImagePart(url: string): Json | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  if (url !== '') return { fileData: { mimeType: 'image/png', fileUri: url } };
  return null;
}

/** The Gemini functionDeclarations for a client request's tools, in either wire's spelling. */
function geminiDeclarations(body: Json): Json[] {
  const out: Json[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (typeof raw.name === 'string') {
      // Anthropic spelling: {name, description, input_schema}.
      out.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    } else if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      // OpenAI spelling: {type: 'function', function: {name, description, parameters}}.
      out.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
  }
  return out;
}

/** The Gemini generationConfig for a client request: output cap, stops, JSON mode, sampling. */
function geminiConfig(body: Json): Json {
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming `model`: system messages
 * become the system instruction, user parts become user contents (text and inline images), assistant
 * messages become model contents, tools become function declarations, and the generation config carries
 * the output cap, stop sequences and JSON mode.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      const part = geminiTextPart(text);
      if (part !== null) parts.push(part);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { result: typeof content === 'string' ? content : textOf(content) } } }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      const part = geminiTextPart(content);
      if (part !== null) parts.push(part);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          const part = geminiTextPart(block.text);
          if (part !== null) parts.push(part);
        } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const part = geminiImagePart(block.image_url.url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = geminiDeclarations(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfig(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body naming `model`: the system prompt
 * becomes the system instruction, user blocks become user contents (text, images and tool results as
 * function responses), assistant blocks become model contents, tools become function declarations, and
 * the generation config carries the output cap and stop sequences.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      const part = geminiTextPart(content);
      if (part !== null) contents.push({ role: assistant ? 'model' : 'user', parts: [part] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        const part = geminiTextPart(block.text);
        if (part !== null) parts.push(part);
      } else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          const part = geminiImagePart(source.url);
          if (part !== null) parts.push(part);
        }
      } else if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
      } else if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarations(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfig(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split streamed text into the pieces the simulated providers use (13 characters each). */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) pieces.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * Plain text, a finish reason and the provider's usage as a Chat Completions SSE stream: the role
 * chunk, the content pieces, the finish chunk, the usage chunk, then [DONE] — the order the client's
 * stream parser expects, with usage last so the final counts are the provider's.
 */
export function openAIEventsFromText(model: string, text: string, finishReason: string, usage: Usage | null): string {
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, finishReason)));
  if (usage !== null) {
    frames.push(sseEvent(null, { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } }));
  }
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * Plain text, a stop reason and the provider's usage as an Anthropic SSE stream: message_start, one
 * text content block with its deltas, content_block_stop, message_delta with the stop reason and the
 * output tokens, then message_stop.
 */
export function anthropicEventsFromText(model: string, text: string, stopReason: string, usage: Usage | null): string {
  const input = usage === null ? 0 : usage.input;
  const output = usage === null ? 0 : usage.output;
  const frames: string[] = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(text)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse streamed text into its SSE frames, tolerating \r\n line endings. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** One JSON object per SSE frame, skipping [DONE] and frames that are not JSON. */
function jsonFrames(frames: readonly SseFrame[]): Json[] {
  const out: Json[] = [];
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    try {
      const value: unknown = JSON.parse(payload);
      if (isRecord(value)) out.push(value);
    } catch {
      // Not JSON: skip the frame.
    }
  }
  return out;
}

/** Text, usage and finish reason read out of a buffered OpenAI SSE stream. */
export function openAIStreamParts(text: string): { text: string; usage: Usage | null; finishReason: string } {
  let out = '';
  let usage: Usage | null = null;
  let finishReason = 'stop';
  for (const chunk of jsonFrames(parseSseFrames(text))) {
    const parsed = openAIUsageOf(chunk);
    if (parsed !== null) usage = parsed;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') finishReason = choice.finish_reason;
  }
  return { text: out, usage, finishReason };
}

/** Text, usage and stop reason read out of a buffered Anthropic SSE stream. */
export function anthropicStreamParts(text: string): { text: string; usage: Usage | null; stopReason: string } {
  let out = '';
  let input = 0;
  let output = 0;
  let stopReason = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) input = numberOrZero(data.message.usage.input_tokens);
    else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: { input, output }, stopReason };
}

/** Text, usage and finish reason read out of a buffered Gemini SSE stream. */
export function geminiStreamParts(text: string): { text: string; usage: Usage | null; finishReason: string } {
  let out = '';
  let usage: Usage | null = null;
  let finishReason = 'STOP';
  for (const body of jsonFrames(parseSseFrames(text))) {
    out += geminiTextOf(body);
    const parsed = geminiUsageOf(body);
    if (parsed !== null) usage = parsed;
    const stop = geminiStopOf(body);
    if (stop !== null) finishReason = stop;
  }
  return { text: out, usage, finishReason };
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
