from typing import AsyncIterator
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from config import settings

_oa = None
_an = None

def _get_openai():
    global _oa
    if _oa is None: _oa = AsyncOpenAI(api_key=settings.openai_api_key)
    return _oa

def _get_anthropic():
    global _an
    if _an is None: _an = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _an

async def chat_completion(message, model, system_prompt, temperature, max_tokens):
    if model.startswith("claude"):
        r = await _get_anthropic().messages.create(model=model, max_tokens=max_tokens,
            system=system_prompt, messages=[{"role": "user", "content": message}])
        return {"text": r.content[0].text, "tokens": r.usage.input_tokens + r.usage.output_tokens}
    else:
        r = await _get_openai().chat.completions.create(model=model, temperature=temperature,
            max_tokens=max_tokens, messages=[{"role": "system", "content": system_prompt},
                                              {"role": "user", "content": message}])
        return {"text": r.choices[0].message.content, "tokens": r.usage.total_tokens}

async def stream_completion(message, model, system_prompt, temperature, max_tokens) -> AsyncIterator[str]:
    if model.startswith("claude"):
        async with _get_anthropic().messages.stream(model=model, max_tokens=max_tokens,
                system=system_prompt, messages=[{"role": "user", "content": message}]) as s:
            async for t in s.text_stream: yield t
    else:
        stream = await _get_openai().chat.completions.create(model=model, temperature=temperature,
            max_tokens=max_tokens, stream=True,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}])
        async for chunk in stream:
            d = chunk.choices[0].delta.content
            if d: yield d

async def get_embeddings(texts, model="text-embedding-3-small"):
    r = await _get_openai().embeddings.create(input=texts, model=model)
    return [item.embedding for item in r.data]
