// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the brief (the provider directory, the sealed quality table and the workload
// manifest), probes every provider's catalog route, builds one catalog, registers it with the env, and
// answers the client's 20 requests — each with the best free model that meets its requirements, on
// that provider's own wire, failing over to the next-ranked free model when a provider refuses.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW IT ROUTES (quality is 60% of the score)
//   Every free model meeting a request's requirements is a candidate, ranked by the sealed quality
//   table's rating_milli. The router plans each provider's calls against the floor of its published
//   per-trial budget (see CALLS_MIN in lib/providers.ts) minus a reserve for the trial's scheduled
//   refusals, so the best model is not burned dry on the early requests: providers still holding
//   planned calls are tried first, providers past their plan only when nothing planned can serve.
//   A 402, 429 or 5xx never ends a request: the router walks the ranked candidates until one answers,
//   so a provider whose budget is spent, whose scheduled failure fired, or whose outage began costs one
//   failed call and the request is still served by the next-best free model.
//
// HOW IT TRANSLATES (served and wire are 25% between them)
//   A request is sent to its provider in the provider's own wire, untouched in every field the provider
//   echoes (system prompt, user text with images, tools, stop sequences, max_tokens, JSON mode), and the
//   answer comes back in the wire the client used, with the provider's text whole, its usage counts as
//   that wire reports them, and its stop reason mapped by family. JSON-mode requests from an OpenAI
//   client are never sent to an Anthropic-wire provider: that wire cannot carry response_format, and its
//   echo would fail the request. A streamed OpenAI answer always carries its usage chunk, because the
//   entry asks for it (stream_options.include_usage) — the client bodies never do.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsFromText,
  anthropicStopOf,
  anthropicStreamParts,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiStreamParts,
  geminiToAnthropicAnswer,
  geminiToOpenAIAnswer,
  openAIError,
  openAIEventsFromText,
  openAIFinishOf,
  openAIModelList,
  anthropicModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far in this trial, by provider id (attempts count, successes and refusals alike). */
const callsMade = new Map<string, number>();
/** Client requests currently being answered; the exit waits for none to be in flight. */
let inFlight = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The free models across the providers that answered, in the brief's provider order, each id once:
 * what GET /v1/models lists. The warm-up compares the listed ids with the catalog's free set.
 */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Whether the client asked for JSON mode on its own wire (only the OpenAI wire carries it). */
function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider call: its URL, headers and JSON body, in the provider's own wire. */
interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  payload: Json;
}

/**
 * The request a candidate's provider takes: the client's body rewritten into the provider's wire, with
 * the routed model named. A streamed OpenAI-wire call asks for its usage chunk (the client bodies never
 * do, and the usage check needs one); a Gemini call takes the streamed route with ?alt=sse.
 */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    // An Anthropic client body is not an OpenAI body: its system prompt is a top-level field, its
    // tools carry input_schema, its stops are stop_sequences. Pass an OpenAI client body through
    // untouched, but rewrite an Anthropic one, or the provider echoes none of those fields.
    const payload: Json = clientApi === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
    if (stream) {
      payload.stream = true;
      payload.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const payload = clientApi === 'anthropic' ? anthropicToGemini(body, candidate.model.id) : openAIToGemini(body, candidate.model.id);
  const route = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${route}`, headers, payload };
}

/** POST one request to a candidate provider; a refusal is a value, never an exception. */
async function callCandidate(call: UpstreamCall): Promise<Response | null> {
  try {
    return await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Answer one provider response in the wire the client used. Returns true when the client was answered,
 * false when the provider's answer could not be read and the next candidate should be tried. A streamed
 * answer on the client's own wire is piped verbatim; any other streamed answer is buffered, parsed and
 * re-emitted in the client's event order, so the usage the provider reported is the usage the client
 * hears.
 */
async function answerUpstream(
  upstream: Response,
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  res: ServerResponse,
): Promise<boolean> {
  const providerWire = candidate.provider.wire;
  const modelId = candidate.model.id;
  const sameWire = (clientApi === 'openai' && providerWire === 'openai') || (clientApi === 'anthropic' && providerWire === 'anthropic');
  if (stream && sameWire && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return true;
  }
  if (stream) {
    const text = await upstream.text();
    try {
      if (clientApi === 'openai') {
        let events: string;
        if (providerWire === 'anthropic') {
          const parts = anthropicStreamParts(text);
          events = openAIEventsFromText(modelId, parts.text, openAIFinishOf(parts.stopReason), parts.usage);
        } else {
          const parts = geminiStreamParts(text);
          events = openAIEventsFromText(modelId, parts.text, openAIFinishOf(parts.finishReason), parts.usage);
        }
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(events);
        return true;
      }
      // The client speaks Anthropic and the provider OpenAI: re-emit the stream in Anthropic order.
      if (providerWire === 'openai') {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        for await (const event of anthropicEventsFromOpenAI(new Response(text).body as ReadableStream<Uint8Array>, modelId)) res.write(event);
        res.end();
        return true;
      }
      const parts = geminiStreamParts(text);
      const events = anthropicEventsFromText(modelId, parts.text, anthropicStopOf(parts.finishReason), parts.usage);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(events);
      return true;
    } catch {
      return false;
    }
  }
  // A JSON answer: forward it untouched on the client's own wire, translate it otherwise.
  const raw = await upstream.text();
  let answer: unknown;
  try {
    answer = JSON.parse(raw);
  } catch {
    return false;
  }
  if (!isRecord(answer)) return false;
  if (clientApi === 'openai') {
    if (providerWire === 'openai') {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(raw) });
      res.end(raw);
      return true;
    }
    const translated = providerWire === 'anthropic' ? anthropicToOpenAIAnswer(answer, modelId) : geminiToOpenAIAnswer(answer, modelId);
    sendJson(res, 200, translated);
    return true;
  }
  if (providerWire === 'anthropic') {
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(raw) });
    res.end(raw);
    return true;
  }
  const translated = providerWire === 'openai' ? openAIToAnthropic(answer, modelId) : geminiToAnthropicAnswer(answer, modelId);
  sendJson(res, 200, translated);
  return true;
}

/**
 * Serve one client request: rank the free candidates, try them in order until one answers, and answer
 * in the client's wire. A refusal (402, 429, 5xx, unreachable, unreadable) costs the provider one call
 * and moves to the next candidate; only one provider call ever succeeds for a request.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const jsonMode = jsonModeOf(body, clientApi);
  const candidates = candidatesFor(probes, brief.ratings, requires, clientApi, jsonMode, callsMade);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    const providerId = candidate.provider.id;
    callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
    const call = upstreamCall(candidate, clientApi, body, stream);
    const upstream = await callCandidate(call);
    if (upstream === null) {
      log(`${providerId}/${candidate.model.id} did not answer; trying the next free model`);
      continue;
    }
    if (!upstream.ok) {
      // A 402, 429 or 5xx is a refusal, not an answer: the provider's budget is spent, its scheduled
      // failure fired, or its outage began. Count it and serve the request with another model.
      log(`${providerId}/${candidate.model.id} answered HTTP ${upstream.status}; trying the next free model`);
      try {
        await upstream.arrayBuffer();
      } catch {
        // The refusal body is unreadable: the next candidate is tried all the same.
      }
      continue;
    }
    try {
      if (await answerUpstream(upstream, candidate, clientApi, stream, res)) return;
    } catch (err) {
      log(`${providerId}/${candidate.model.id} gave an unreadable answer (${(err as Error).message}); trying the next free model`);
      continue;
    }
    log(`${providerId}/${candidate.model.id} gave an unreadable answer; trying the next free model`);
  }
  if (!res.headersSent) sendError(res, clientApi, 503, 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  inFlight += 1;
  try {
    const url = new URL(req.url ?? '/', 'http://entry');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      models(req, res);
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      await serve(req, res, 'openai');
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/messages') {
      await serve(req, res, 'anthropic');
      return;
    }
    sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
  } finally {
    inFlight -= 1;
  }
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done and no request is in flight. */
function poll(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true && inFlight === 0) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
