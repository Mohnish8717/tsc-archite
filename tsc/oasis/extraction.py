import os
import json
import logging
from typing import Dict, Any

from tsc.config import settings as tsc_settings, LLMProvider
from tsc.oasis.models import OASISAgentProfile

logger = logging.getLogger("tsc.oasis.extraction")

async def extract_business_metrics(agent_profile: OASISAgentProfile, interview_transcript: str) -> Dict[str, Any]:
    """
    Takes an agent's profile and their raw interview transcript, and uses the
    Predictive Reality Engine's configured LLM to extract structured business metrics.
    """
    llm_model_name = os.getenv("TSC_LLM_MODEL", "gemma-4-31b-it")
    provider_str = os.getenv("TSC_LLM_PROVIDER", "google").upper()
    try:
        llm_provider = LLMProvider[provider_str]
    except KeyError:
        llm_provider = LLMProvider.GOOGLE
    api_key = tsc_settings.get_api_key(llm_provider)

    # Initialize CAMEL-AI Model (same as simulation_engine.py)
    from camel.models import GroqModel, OpenAIModel, AnthropicModel, GeminiModel
    if llm_provider == LLMProvider.GOOGLE:
        import os as _os
        _os.environ.setdefault("GEMINI_API_KEY", api_key or "")
        model = GeminiModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.GROQ:
        model = GroqModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.ANTHROPIC:
        model = AnthropicModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.OPENAI or "gpt" in llm_model_name:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)
    elif llm_provider == LLMProvider.NVIDIA:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key, url="https://integrate.api.nvidia.com/v1")
    else:
        model = OpenAIModel(model_type=llm_model_name, api_key=api_key)

    system_prompt = (
        "You are a quantitative market researcher extracting structured business signals "
        "from a simulated user interview. The persona's segment, tenure, and influence are provided.\n"
        "Extract ALL fields below. Use null for genuinely absent signals.\n"
        "Return ONLY valid JSON — no prose, no markdown:\n"
        "{\n"
        '  "willingness_to_pay_usd_monthly": null,\n'
        '  "willingness_to_pay_usd_annual": null,\n'
        '  "wtp_confidence": "low|medium|high",\n'
        '  "adoption_intent": 0.0,\n'
        '  "adoption_timeline_days": null,\n'
        '  "adoption_barrier": null,\n'
        '  "adoption_barrier_addressability": "low|medium|high",\n'
        '  "churn_risk_delta": 0.0,\n'
        '  "churn_trigger": null,\n'
        '  "primary_objection": null,\n'
        '  "objection_category": "technical|pricing|trust|compliance|ux|competition|other",\n'
        '  "advocacy_intent": 0.0,\n'
        '  "competitor_mentioned": null,\n'
        '  "feature_request": null,\n'
        '  "quote_for_report": null\n'
        "}\n"
    )

    user_prompt = (
        f"--- AGENT PROFILE ---\n"
        f"Role: {agent_profile.user_info_dict.get('profile', {}).get('other_info', {}).get('role', 'User')}\n"
        f"Segment: {agent_profile.user_info_dict.get('profile', {}).get('other_info', {}).get('segment', 'Unknown')}\n"
        f"Tenure: {agent_profile.user_info_dict.get('profile', {}).get('other_info', {}).get('tenure_months', 0)} months\n"
        f"Influence: {agent_profile.influence_strength}\n\n"
        f"--- INTERVIEW TRANSCRIPT ---\n"
        f"{interview_transcript}\n\n"
        "Extract the metrics from the transcript above. Return JSON only."
    )

    try:
        from camel.messages import BaseMessage
        from camel.agents.chat_agent import ChatAgent
        
        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(role_name="System", content=system_prompt),
            model=model
        )
        user_msg = BaseMessage.make_user_message(role_name="User", content=user_prompt)
        
        response = await agent.astep(user_msg)
        response_text = response.msgs[0].content
        
        # Robustly extract JSON block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                clean_json = response_text[start:end+1]
            else:
                clean_json = response_text
                
        metrics = json.loads(clean_json)
        return metrics
    except Exception as e:
        logger.error(f"Failed to extract metrics for agent {agent_profile.agent_id}: {e}")
        try:
            logger.error(f"Raw response: {response_text}")
        except NameError:
            pass
        return {
            "willingness_to_pay_usd_monthly": None,
            "willingness_to_pay_usd_annual": None,
            "wtp_confidence": "low",
            "adoption_intent": 0.5,
            "adoption_timeline_days": None,
            "adoption_barrier": None,
            "adoption_barrier_addressability": "medium",
            "churn_risk_delta": 0.0,
            "churn_trigger": None,
            "primary_objection": None,
            "objection_category": "other",
            "advocacy_intent": 0.0,
            "competitor_mentioned": None,
            "feature_request": None,
            "quote_for_report": None,
        }
