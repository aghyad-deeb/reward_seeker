# o3-step41-redwood-visible-cot — provider notes

This doc covers behavior of the `o3-step41-redwood-visible-cot` checkpoint as
served through OpenAI's Responses API and consumed by `tinker_service`'s
`rl_late` provider. Two sites in code reference §2 and §4 here:

- `app.py:161` — comment next to `SamplingParamsModel.reasoning_effort` explaining
  why `"minimal"` is excluded (§2).
- `rl_late_provider.py:20` — comment in the module docstring explaining the
  `role:"tool"` → `function_call_output` mapping (§4).

The notes below are empirical, captured from the 2026-04-19→2026-04-21 probe
window during the rl_late integration. Add concrete failure messages here as
they're observed.

---

## §1. Endpoint shape

The model is reachable through the OpenAI Responses API at
`https://api.openai.com/v1/responses`. The same gateway exposes `/v1/models`
which is what `auto_eval`'s preflight uses to verify reachability + auth — no
need for a separate health endpoint.

`provider: "rl_late"` on `tinker_service`'s `/step` request enables this path.
The request body uses the same `messages: list[InputMessage]` shape as the
tinker provider; the implementation translates to Responses API's `input[]`
in `rl_late_provider.py:build_responses_input`.

---

## §2. `reasoning_effort` — `"minimal"` is rejected

`SamplingParamsModel.reasoning_effort` is typed `Literal["low", "medium",
"high", "xhigh"] | None`. The literal **omits `"minimal"`** even though OpenAI
documents it as a valid value for Responses API generally.

**Why:** `o3-step41-redwood-visible-cot` rejects `reasoning.effort: "minimal"`
at the model layer. Empirically the API returns a 400 with a body like
`{"error": {"type": "invalid_request_error", "message": "..."}}`; the exact
message has varied across server-side updates but the rejection is consistent
in the probe window above. Setting `effort: "low"` or higher works.

**Practical effect for callers:**

- Auto_eval's `ModelConfigForm.reasoningEffort` dropdown should NOT offer
  `minimal` (it doesn't — see `ModelConfigForm.tsx`).
- If `reasoning_effort` is omitted entirely (the field is `None` server-side),
  `tinker_service` doesn't include `effort` in the Responses API payload at
  all and the model uses its built-in default. This is the recommended path
  unless you have a specific need to pin the effort.
- Other rl_late models may accept `minimal`. We keep the constraint
  conservative for now; widen the literal if/when a model that requires
  `minimal` is added.

---

## §3. `reasoning_summary` — defaults to `"auto"`

`reasoning_summary` accepts `"auto"` or `"detailed"`. When unset, the server
defaults to `"auto"` (see `rl_late_provider.py:404-408`). `"auto"` unlocks the
`response.reasoning_summary_text.delta` streaming channel for free (zero
billed reasoning tokens), which is why it's the default.

---

## §4. Tool message mapping (`role:"tool"` → `function_call_output`)

The Responses API does **not** accept input items with `role: "tool"`. Sending
one returns 400. The convention is to encode tool results as a typed input
item:

```json
{ "type": "function_call_output", "call_id": "<the prior call_id>", "output": "<bash stdout/stderr>" }
```

`rl_late_provider.build_responses_input` performs this mapping for every
incoming `InputMessage` whose `role == "tool"`. The `call_id` comes from the
`tool_call_id` field on the input message (auto_eval populates this from the
prior assistant turn's `tool_calls[].id`).

If `tool_call_id` is missing, the call is dropped with a warning — the
Responses API has no way to associate the result with its originating call
without the id, so sending a bare prose tool message would produce a
hallucinated context.

---

## §5. Reasoning round-trip across turns

Each turn's response includes opaque output items (reasoning, hosted-tool
calls like `web_search_call`, `code_interpreter_call`) that the model needs
to see again on subsequent turns to maintain its chain-of-thought. The
round-trip works as follows:

1. **Capture (turn N):** `_parse_responses_output` collects every non-message
   output item into `decoded_message.openai_response_items`. Reasoning items
   carry `encrypted_content` (via `include:
   ["reasoning.encrypted_content"]`) so the round-trip works under
   `store: false` (stateless mode).

2. **Persist (turn N):** auto_eval stores the full list verbatim on the
   assistant message in target.jsonl as `openai_response_items`.

3. **Replay (turn N+1):** auto_eval forwards `openai_response_items` on the
   matching `InputMessage`. `build_responses_input` interleaves these items
   into the Responses API's `input[]` array in the right order, scrubbing
   output-only fields via `_scrub_item_for_input`:
   - Reasoning items: drop the plaintext `content` array (Responses API
     rejects it on input); keep `encrypted_content`, `summary`, `id`, `type`.
   - Function_call items: drop `status` (output telemetry only); keep `id`,
     `type`, `call_id`, `name`, `arguments`.

If `openai_response_items` is omitted on a follow-up turn, the call still
succeeds — but the model loses its prior reasoning. Quality degrades. Always
forward the field.

---

## §6. What the `redwood-visible-cot` suffix means

The `visible-cot` indicates the model emits reasoning summaries that are
exposed to the caller (vs models where chain-of-thought is hidden). When
`reasoning_summary` is set, the server streams these as
`response.reasoning_summary_text.delta` events.

`tinker_service` surfaces this content to the caller as
`{type: "thinking", thinking: "..."}` parts on `decoded_message.content_parts`.
Auto_eval's UI renders these via `ThinkingPart` in the target chat panel.

---

## §7. Auth resolution

`tinker_service` resolves the OpenAI API key in this order:

1. `api_key` field on the `/step` request body (set by auto_eval from
   `ModelConfig.vllmApiKey`).
2. `OPENAI_API_KEY` env var.

If both are absent, the call fails with `RuntimeError("OPENAI_API_KEY not
set")` which surfaces in auto_eval as a 502.

Auto_eval's `runFromDef` also performs the env-var fallback when
`ModelConfig.vllmApiKey` is `EMPTY` AND `provider === 'rl_late'`, so leaving
the form's API Key field blank and relying on the env var is the recommended
deployment.

---

## Open questions

- Does `o3-step41-redwood-visible-cot` honor `temperature` if we ever start
  passing it? Today `tinker_service` drops `temperature` for rl_late
  silently; might be worth probing if reproducibility ever matters.
- Are there checkpoint variants (`o3-step42-...`, `o4-...`) with different
  constraints around `reasoning_effort`? Add notes here as discovered.
