---

## layout: post

title: "Building an inference server from scratch, Part #2: Streaming tokens"
date: 2026-10-04
categories: [blog]
tags: [llm, large-language-model, ai-engineering, inference-server, learning]
excerpt: "Enhancing the chat user experience by leveraging the fact that LLMs generate one token at a time and streaming it to the user."

In the last post we talked about building an inference server that generates responses using a Qwen3 model, how token generation happens one by one, what prefill and a decode loop is. We also talked about what a KV cache is, and how by using a KV cache we're able to optimise inter token latency (ITL) & time to generate the entire output. We ran some benchmarks and recorded the speedup observed on different response sizes (the benefit of a KV cache becomes larger as the size of our response grows). 

In this blog we're gonna talk about enhancing the chat user experience by streaming tokens. We will also briefly cover sampling with temperature / top k and what happens when multiple requests hit the server.

Streaming as a ux improvement exists across different usecases, personally for me at work we're using this to stream stores & items to a customer as part of our generative recommendation efforts (for the uninitiated generative recommendation is a frontier search and retrieval technique where we collapse the entire search / retrieval pipleine - consisting of a retrieval (from lexical / vector index) + multiple ranking stages optimising different objectives) to a single generative model which auto-regressively generates search results for a customer. It's very much fun to learn about and here are some foundational papers if you're interested to read more: 

1. https://arxiv.org/abs/2305.05065
2. https://arxiv.org/abs/2510.07784

Recall that we had earlier defined a streaming endpoint called `stream_text`: 

```python
@app.post("/stream")
async def stream_text(request: GenerateRequest):
    # generate_stream() returns a generator. Do not list() or join() it here —
    # StreamingResponse iterates it and sends each yield as an SSE frame.
    return StreamingResponse(
        generate_stream(
            request.prompt,
            request.token_limit,
            request.temperature,
            request.top_k,
        ),
        media_type="text/event-stream",
    )
```

It internally calls a `generate_stream` method. Let me dump it here so we can explore it's components one by one. 

```python
@torch.inference_mode()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
):
    """Yield SSE chunks as tokens are produced. FastAPI's StreamingResponse
    iterates this generator and flushes each yield to the client immediately.
    """
    if top_k is not None and top_k <= 0:
        top_k = None

    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    previous_text = ""

    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )


    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_k,
        )
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=1,
        )

        # Decode the whole completion so far, then emit only the new suffix.
        # Needed because one token is not always a complete UTF-8 / BPE piece.
        current_text = tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True,
        )
        chunk = current_text[len(previous_text):]
        previous_text = current_text
        if chunk:
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

    yield "data: [DONE]\n\n"

```

If you go back to my previous blog (insert link of previous blog here), and look at the `generate_cached` method, you'd see that it's largely the same. The only differences are that inside each loop we're doing a `tokenizer.decode()` call. Let's look at the arguments: 

generated_ids has shape `(batch_size, sequence_length)`. The batch size is 1, so it looks like this:

generated_ids = [[ p0, p1, p2, ..., p_(n-1),     g0, g1, g2 ]]  
                             └─ prompt (n tokens)┘└ generated ┘  
The indexing [0, prompt_length:] does two things:

1. 0 picks the first and only sequence in the batch. This turns the 2-D tensor into 1-D.
2. prompt_length: keeps everything from index n onward. prompt_length is n, the number of prompt tokens saved at line 174, so the slice skips the prompt. So effectively we're decoding only what was generated by the decode loop, but every iteration we decode everything.

Next line: current_text is the full completion so far, but the client has already received part of it: chunk is the part that hasn't been sent yet. `previous_text` is exactly what was decoded and sent on the last iteration. `len(previous_text)` is the number of characters already delivered. Slicing from that index onward gives **only the new characters**.

Why not yield current_text directly? The client would get something like: Hel, Hello, Hello wor, and so on. Each frame would repeat everything before it. The client appends each chunk to what it has, so it needs only the new piece.

Decoding only the last token fails because a token is not always a complete character.

**How tokens map to text**

Qwen's tokenizer works on **bytes**, not characters. Plain English tokens like `" Hello"` or `"'t"` are whole units, and decoding one alone works fine. But some characters, such as emoji, Hindi/Devanagari, Chinese and other non-ASCII text, take multiple bytes in UTF-8. The tokenizer can split those bytes across tokens.

For example, 🙂 is 4 bytes: `F0 9F 99 82`. It might be tokenized as:

```
token A = F0 9F
token B = 99 82

```

**What happens if you decode each token alone**

```python
tokenizer.decode([A])  # "�"   (half a character, invalid UTF-8)
tokenizer.decode([B])  # "�"   (the other half, also invalid)

```

Each half alone is garbage. Together, `tokenizer.decode([A, B])` gives `"🙂"`. The meaning exists only when both tokens are decoded together, so one token on its own cannot be decoded reliably.

**Why decoding the whole completion is better**

Decoding all tokens so far always puts the bytes in the right context, so the decoder sees A and B together once both have arrived. That is why the code decodes the full completion and then takes only the new part with `current_text[len(previous_text):]`.

**Summary**

- For ordinary English words, decoding the last token alone would mostly work.
- For multi-byte characters it breaks, and an inference server has to handle those.
- Decoding everything and slicing off the already-sent part is the simple way to stay correct.

If the new token decodes to nothing visible, such as a skipped special token, current_text equals previous_text. The slice is "", and if chunk: skips the yield, so no empty frames are sent.   

And finally, if we encounter a chunk, we yeet it, using `yield`. In Python, **using the** `yield` **keyword allows you to stream outputs one by one**. Instead of waiting for a long text response to finish before returning, `yield` pauses execution to send back each piece of data as soon as it is ready. When generating text with Large Language Models, `yield` sends each generated token or word to the user interface immediately. This drastically reduces perceived latency compared to waiting for the entire response to generate and sending it back as a whole.   

But wait, while writing this I realised there's a bug in my code. It doesn't really do what it says. As an example I ran a test client which sent a request with prompt: "Show me emojis" and the response came back with this: 

<video controls muted playsinline preload="metadata" style="max-width: 100%;">
  <source src="{{ '/assets/video/emoji-bug.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

The emojis come out to look like  "�" because of the bug. It's because: In the current code, `�` **is** captured in `chunk` and sent to the client.

Here is the trace for the split emoji, with tokens A and B:

- **After token A:** `current_text = "Hi �"` and `previous_text = "Hi "`. So `chunk = "�"`, which is non-empty, so `if chunk:` passes and the code yields `�`.
- **After token B:** `current_text = "Hi 🙂"` and `previous_text = "Hi �"`. Both are 4 characters long, so `chunk = ""`. Nothing is yielded, and the real emoji is lost. Nothing in the current code checks for `�`. The `if chunk:` guard only checks whether `chunk` is empty, and `"�"` is not empty.

  **With the fix, it wouldn't be captured:**

```python
if not current_text.endswith("\ufffd"):
    chunk = current_text[len(previous_text):]
    previous_text = current_text
    if chunk:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
```

Add another video showing proper emoji rendering. 

Great success!



