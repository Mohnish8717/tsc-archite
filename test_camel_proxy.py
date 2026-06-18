import os
import asyncio
from camel.models import ModelFactory
from camel.types import ModelPlatformType

async def main():
    proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://0.0.0.0:4000")
    print(f"Testing connection to proxy at: {proxy_url}")
    
    # Instantiate the model exactly like we do in simulation_engine.py
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="gemma-4-31b-it",
        url=proxy_url,
        api_key="litellm-dummy-key",
        max_retries=10
    )
    
    # Create a simple message in OpenAI message dictionary format
    test_message = {
        "role": "user",
        "content": "Hello! Are you receiving this through the LiteLLM proxy?"
    }
    
    print("Sending request to Camel-AI ModelBackend...")
    try:
        response = await model.arun(messages=[test_message])
        print("Success! Received response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
