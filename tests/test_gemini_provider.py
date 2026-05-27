import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from tsc.llm.gemini_provider import GeminiClient

@pytest.fixture
def mock_genai():
    mock_bucket = MagicMock()
    mock_bucket.acquire = AsyncMock()
    
    mock_leaky = MagicMock()
    async def fake_call(coro):
        return await coro
    mock_leaky.call = fake_call
    
    with patch("google.generativeai.GenerativeModel") as mock_gen_model, \
         patch("google.generativeai.configure") as mock_configure, \
         patch("tsc.llm.gemini_provider.get_gemini_bucket", return_value=mock_bucket), \
         patch("tsc.llm.gemini_provider.get_leaky_bucket", return_value=mock_leaky):
        yield mock_gen_model, mock_configure

@pytest.mark.asyncio
async def test_gemini_client_standard_gemini(mock_genai):
    mock_gen_model, _ = mock_genai
    
    mock_response = MagicMock()
    mock_response.text = '{"status": "success"}'
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 15
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_gen_model.return_value = mock_model_instance
    
    client = GeminiClient(api_key="fake_key", model="gemini-2.5-flash")
    
    result = await client.analyze(
        system_prompt="You are a system expert.",
        user_prompt="Run analyzer.",
    )
    
    assert result == {"status": "success"}
    
    # Assert that GenerativeModel was called with system_instruction
    mock_gen_model.assert_any_call(
        model_name="models/gemini-2.5-flash",
        system_instruction="You are a system expert.",
    )
    
    # Assert generate_content_async was called with only user prompt
    mock_model_instance.generate_content_async.assert_called_with(
        "Run analyzer.",
        generation_config=mock_model_instance.generate_content_async.call_args[1]['generation_config']
    )

@pytest.mark.asyncio
async def test_gemini_client_legacy_gemma(mock_genai):
    mock_gen_model, _ = mock_genai
    
    mock_response = MagicMock()
    mock_response.text = '{"status": "success"}'
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 15
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_gen_model.return_value = mock_model_instance
    
    client = GeminiClient(api_key="fake_key", model="gemma-2-9b-it")
    
    result = await client.analyze(
        system_prompt="You are a system expert.",
        user_prompt="Run analyzer.",
    )
    
    assert result == {"status": "success"}
    
    # Assert that GenerativeModel was called WITHOUT system_instruction for legacy Gemma
    mock_gen_model.assert_any_call(
        model_name="models/gemma-2-9b-it"
    )
    
    # Assert generate_content_async was called with concatenated prompt
    mock_model_instance.generate_content_async.assert_called_with(
        "You are a system expert.\n\nRun analyzer.",
        generation_config=mock_model_instance.generate_content_async.call_args[1]['generation_config']
    )

@pytest.mark.asyncio
async def test_gemini_client_gemma4(mock_genai):
    mock_gen_model, _ = mock_genai
    
    mock_response = MagicMock()
    mock_response.text = '{"status": "success"}'
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 15
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_gen_model.return_value = mock_model_instance
    
    # Testing Gemma 4 name variation
    client = GeminiClient(api_key="fake_key", model="gemma-4-31b-it")
    
    result = await client.analyze(
        system_prompt="You are a system expert.",
        user_prompt="Run analyzer.",
    )
    
    assert result == {"status": "success"}
    
    # Assert that GenerativeModel was called with system_instruction (native Gemma 4 support)
    mock_gen_model.assert_any_call(
        model_name="models/gemma-4-31b-it",
        system_instruction="You are a system expert.",
    )
    
    # Assert generate_content_async was called with only user prompt
    mock_model_instance.generate_content_async.assert_called_with(
        "Run analyzer.",
        generation_config=mock_model_instance.generate_content_async.call_args[1]['generation_config']
    )

@pytest.mark.asyncio
async def test_gemini_client_fallback_gemma4(mock_genai):
    mock_gen_model, _ = mock_genai
    
    # Configure mock_gen_model to return an instance whose generate_content_async fails (e.g. timeout) first
    mock_model_instance_fail = MagicMock()
    mock_model_instance_fail.generate_content_async = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))
    
    mock_response_success = MagicMock()
    mock_response_success.text = '{"status": "fallback_success"}'
    mock_response_success.usage_metadata.prompt_token_count = 30
    mock_response_success.usage_metadata.candidates_token_count = 25
    
    mock_model_instance_success = MagicMock()
    mock_model_instance_success.generate_content_async = AsyncMock(return_value=mock_response_success)
    
    # GenerativeModel will be called three times:
    # 1. In client initialization (GenerativeModel("models/gemma-2-9b-it"))
    # 2. Inside analyze() for primary attempt (GenerativeModel("models/gemma-2-9b-it"))
    # 3. Inside analyze() for fallback attempt (GenerativeModel("models/gemma-4-31b-it"))
    mock_gen_model.side_effect = [MagicMock(), mock_model_instance_fail, mock_model_instance_success]
    
    client = GeminiClient(api_key="fake_key", model="gemma-2-9b-it")
    
    result = await client.analyze(
        system_prompt="You are a system expert.",
        user_prompt="Run analyzer.",
    )
    
    assert result == {"status": "fallback_success"}
    
    # Verify fallback model models/gemma-4-31b-it was instantiated with native system instruction
    mock_gen_model.assert_any_call(
        model_name="models/gemma-4-31b-it",
        system_instruction="You are a system expert.",
    )
