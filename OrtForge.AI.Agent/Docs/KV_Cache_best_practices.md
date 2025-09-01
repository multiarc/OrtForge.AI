### What a KV cache is (for LLMs)~~~~

- In decoder-only Transformers (e.g., LLaMA, GPT), each attention layer computes Keys (K) and Values (V) for every processed token.
- During autoregressive generation, you produce tokens one by one. Without caching, each new token would force recomputation of K and V for the entire prefix at every step, which is expensive.
- KV cache stores the K and V tensors for already-processed tokens so the model only computes K/V for the new token and attends to the cached K/V for the past. This dramatically reduces per-step compute.

In short: KV cache is the model’s per-layer memory of past attention states, enabling fast incremental decoding.

### Why we need the KV cache

- Performance: Avoid quadratic recomputation over the growing prefix at each step.
- Cost efficiency: Per-step cost becomes roughly linear in sequence length (or near-constant with paged attention implementations).
- UX: Enables responsive token streaming during generation.

### How to interact with an LLM using KV cache (token-by-token)

1. Tokenize the prompt to input_ids.
2. First pass (prefill):
    - Inputs: input_ids = the prompt (length > 1), optional attention_mask and position_ids.
    - No past_key_values yet.
    - Outputs: logits (for next token) and present_key_values (the KV cache for the entire processed sequence).
3. Choose next token from logits (argmax/sampling/temperature/top-k/p).
4. Next step (incremental decoding):
    - Inputs: input_ids = [the single new token], and past_key_values = the cache from the previous step; also attention_mask/position_ids if required.
    - Outputs: new logits and updated present_key_values (prefix + new token).
5. Repeat step 3–4 until stopping (EOS token, length limit, etc.).

This pattern allows you to “serve” the KV cache by feeding each step’s present_key_values back as the next step’s past_key_values for the same sequence.

### Naming conventions you’ll see (LLaMA/ONNX)

- Inputs often expect: input_ids, optional attention_mask, position_ids, and past_key_values.* (or past_* per layer and K/V).
- Outputs often provide: logits and present_key_values.* (or present_* variants).
- Between steps you map: present_* → past_*.
- Exporters vary (e.g., present_key_values.X.key vs present.X.k). A small name-normalization layer is common and recommended.

### Typical tensor shapes (may vary by export)

- Input IDs: [batch, cur_len] (cur_len is often 1 during decoding).
- Keys/Values per layer:
    - Key: [batch, num_kv_heads, kv_len, head_dim]
    - Value: [batch, num_kv_heads, kv_len, head_dim] (sometimes the last two dims are swapped)
- kv_len increases with the number of processed tokens.
- With grouped-query attention (GQA), num_kv_heads < num_attention_heads; queries fan out over fewer KV heads.
- Attention mask: could be [batch, total_len] or a 4D causal mask; confirm the export.
- Position IDs: usually [batch, cur_len], incrementing with the sequence; sometimes implicit.

Always check your model’s input/output metadata to confirm exact shapes and names.

### Memory considerations (order-of-magnitude)

KV memory (fp16) ≈ 2 (K and V) × layers × batch × num_kv_heads × head_dim × seq_len × 2 bytes.
- Example: 32 layers, batch 1, 8 KV heads, head_dim 128, seq_len 4096 → ~537 MB.
- Multiply by concurrent sequences to estimate server memory.
- Practical strategies:
    - Use fp16/bf16; consider 8-bit KV cache if supported.
    - Use paged attention to allocate KV in fixed-size pages, enabling efficient batching and prefix sharing.
    - Implement eviction (LRU/TTL) and caps per tenant.

### Serving patterns

- Single-process decoding loop (stateful):
    - Keep present_key_values from step t; feed them as past_key_values at step t+1.
    - Maintain this per active generation (session/conversation/beam).

- Multi-user server:
    - Maintain a KV cache handle per active sequence. Associate each client’s “continue” request with its handle.
    - Keep the cache on the same device as the model (GPU/CPU). Avoid serializing to disk; device-specific and large.
    - Use a scheduler to batch multiple sequences at the same decoding step; manage variable lengths with masks.
    - Reclaim KV memory when a sequence ends or times out.
    - For beam search: either duplicate caches per beam or use copy-on-write/page sharing for common prefixes.

- Stateless API shape:
    - The service returns an opaque handle after prefill. Clients send handle + new text/tokens to continue. The server resolves the handle to in-memory KV blocks.

### Pseudocode for generation with KV cache

- Prefill:
    - inputs: input_ids = prompt; outputs: logits, present_kv
    - pick next_token from logits
- Loop:
    - inputs: input_ids = [next_token], past_kv = present_kv; outputs: logits, present_kv
    - pick next_token; repeat

### Common pitfalls and how to avoid them

- Name mismatches (present_* vs past_*): add a mapping layer to normalize.
- Value tensor layout mismatch (kv_len and head_dim swapped in V): verify and transpose if needed.
- Incorrect/omitted position_ids or attention_mask: follow the export’s expectations.
- Moving KV across devices/processes: impractical; keep it co-located with the model runtime.
- Memory blow-ups: cap max concurrent sequences, use paging, and evict aggressively.

### Quick checklist

- At t=0: run prompt without past_kv; capture present_kv.
- At t>0: run with input_ids=[last token], past_kv=previous present_kv.
- Keep KV per session on the model device.
- Normalize naming present_* → past_*.
- Mind shapes/masks/positions and memory limits.

By following this pattern, you “serve” the KV cache correctly and get fast, responsive generation by reusing attention state rather than recomputing it each step.