# Free Model Aggregator

One server, two APIs, ten free providers. It serves the OpenAI Chat Completions API
(`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`), each in JSON
and in server-sent events, routing every request to the best free model that meets its requirements
— ranked by the sealed quality table — and failing over to the next-best free model when a provider
refuses (402/429/5xx, budget spent, scheduled failure, or outage).

- `node main.ts` — reads the brief from `MAKEFUN_PROMPT_FILE`, probes the ten providers' catalog
  routes, registers the catalog with the env, serves the client's requests, and exits 0 when the
  workload is done.
- `node main.ts --list` — prints the catalog document the server registers, byte for byte, and
  exits 0.
- `GET /v1/models` — the model list in the wire the auth header names: `Authorization: Bearer`
  gets the OpenAI list, `x-api-key` the Anthropic one.

The router plans each provider's calls against the floor of its published per-trial budget, so the
best model is not burned dry on the early requests, and it never picks a paid model. Requests are
sent to providers in the provider's own wire — OpenAI, Anthropic, or Gemini — with every echoed
field (system prompt, user text and images, tools, stop sequences, `max_tokens`, JSON mode) carried
across unchanged, and answers come back in the client's wire with the provider's text whole, its
usage counts, and its stop reason mapped by family. JSON-mode requests from an OpenAI client are
never sent to an Anthropic-wire provider, which cannot carry `response_format`.

Node 22 standard library only; no dependencies, no API keys, no model calls of its own. Deterministic:
no wall clock, no unseeded randomness — every choice is a function of the brief, the requests, and
the providers' answers, so a replay leaves the same state.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, and the failover loop.
- `lib/providers.ts` — the brief's parser, the catalog probe, and the capacity-aware candidate ranking.
- `lib/wire.ts` — the client-wire and provider-wire translations and the SSE writers.

## License

MIT — see [LICENSE](LICENSE).

---

Built for the [make.fun](https://make.fun) bounty "The best free AI model aggregator". If you enter
too, use my referral link: https://make.fun/r/kfz6780b
