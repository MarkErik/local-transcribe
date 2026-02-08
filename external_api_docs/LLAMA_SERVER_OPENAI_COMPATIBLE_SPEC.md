# LLAMA_SERVER_OPENAI_COMPATIBLE_SPEC.md

## Purpose

This document defines **exactly how to format requests** to a `llama-server` instance (from `llama.cpp`) when it is running in **OpenAI-compatible mode**.

This is the only format tools and agents should use unless explicitly told otherwise.

---


## Base URL

```
http://HOST:PORT/v1
```

---

## Required Endpoint (Chat)

**Always use:**

```
POST /v1/chat/completions
```


---

## Minimal Valid Request

```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "You are a precise assistant."},
    {"role": "user", "content": "Hello."}
  ]
}
```

### Notes

- `model` is required by clients but ignored by `llama-server` if only one model is loaded.
- Messages **must** be an array.
- Roles must be: `system`, `user`, or `assistant`.

---

## Full Featured Request Template

```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "You are a precise technical assistant."},
    {"role": "user", "content": "Explain TCP in one paragraph."}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "min_p": 0.05,
  "repeat_penalty": 1.1,
  "max_tokens": 512,
  "stream": false
}
```

All sampling parameters are supported directly by llama-server.

---

## Structured / JSON Output

`llama-server` supports OpenAI’s `response_format`.

Example:

```json
{
  "model": "local-model",
  "messages": [
    {"role": "user", "content": "Give me a JSON object with name and age."}
  ],
  "response_format": {
    "type": "json_object"
  }
}
```

---

## CURL Example

```bash
curl http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer no-key"   -d '{
    "model": "local-model",
    "messages": [
      {"role":"system","content":"You are a precise assistant."},
      {"role":"user","content":"Write a bash script that prints hello."}
    ]
  }'
```


---

## Critical Rules

1. Always use `/v1/chat/completions`
2. Always send `messages[]`, never `prompt`

---

## Health / Discovery

List models:

```
GET /v1/models
```

---

## Summary

If a tool or agent follows this spec exactly, it will be fully compatible with `llama-server` in OpenAI mode without any special casing.
