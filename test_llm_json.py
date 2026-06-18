import asyncio
import httpx
import json

prompt = f"""<role>
You are a Senior Social Simulation Architect.
</role>

<output_schema>
MANDATORY: You MUST return ONLY valid JSON matching this exact structure.
Do not include any XML tags, preamble, or markdown in your response.

{{
  "posts": [
    {{
      "archetype": "OFFICIAL_ANNOUNCEMENT",
      "content": "<string>"
    }}
  ]
}}
</output_schema>"""

async def run():
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "gemma-4-31b-it",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        res = await client.post("http://localhost:4000/v1/chat/completions", json=payload, timeout=60)
        print("Status Code:", res.status_code)
        try:
            data = res.json()
            print("Response:", data["choices"][0]["message"]["content"])
        except Exception as e:
            print("Error parsing response:", e)

asyncio.run(run())
