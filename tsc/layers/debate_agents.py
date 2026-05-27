from __future__ import annotations
import re
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Private intelligence packets to inject during debate setup
PRIVATE_INTELLIGENCE_PACKAGES = {
    'CISO': {
        'threat_brief': 'CLASSIFIED: Internal Red Team report dated 2026-03 found '
                        'critical RCE vulnerability in the WebUSB stack used by the '
                        'proposed BCI sync protocol. CVE has not been published.',
        'reveal_condition': 'Only reveal this if the CTO proposes using WebUSB.'
    },
    'CFO': {
        'projection': 'PRIVATE: Q3 cash position is $8.2M, not $12M as stated in the '
                      'board pack. The controller made an error. The actual runway is '
                      '4 months, not 7. You cannot approve anything > $500k/mo.',
        'reveal_condition': 'You may reveal this if pushed on budget approval.'
    },
}

SYCOPHANCY_TOKEN_PENALTIES = {
    1881: -0.8, 5059: -0.8, 13347: -0.7, 1959: -0.6, 
    18717: -0.5, 4857: -0.6, 7273: -0.6,
}

def build_anti_sycophancy_config(base_config: dict, is_moderator: bool) -> dict:
    """Safe logit-bias injection — skips Google API, degrades gracefully."""
    if is_moderator:
        return base_config
    try:
        config = base_config.copy()
        new_config_list = []
        for cfg in config.get('config_list', []):
            cfg_copy = cfg.copy()
            is_google_endpoint = "generativelanguage.googleapis.com" in (cfg_copy.get('base_url') or "")
            if cfg_copy.get('api_type') == 'google' or is_google_endpoint:
                new_config_list.append(cfg_copy)
                continue
            
            penalties = SYCOPHANCY_TOKEN_PENALTIES
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model(cfg_copy.get('model', 'gpt-4'))
                sycophancy_phrases = ['great point', 'I agree', 'absolutely', 'exactly right', 'well said', 'you are correct', 'brilliant']
                runtime_penalties = {}
                for phrase in sycophancy_phrases:
                    tokens = enc.encode(phrase)
                    for tid in tokens[:2]:  # First 2 tokens per phrase
                        runtime_penalties[tid] = -0.6
                penalties = runtime_penalties
            except (ImportError, KeyError):
                pass  # Fall back to hardcoded IDs
            existing = cfg_copy.get('logit_bias', {})
            merged = {**penalties, **existing}
            cfg_copy['logit_bias'] = merged
            new_config_list.append(cfg_copy)
        config['config_list'] = new_config_list
        return config
    except Exception as e:
        logger.warning(f'Anti-sycophancy logit_bias injection failed: {e} — using base config')
        return base_config

# Token Sparsification Middleware (Temporal & Entity Preserving compression)
try:
    from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
    from autogen.agentchat.contrib.capabilities.transforms import MessageTransform
    
    class ThoughtTagStripper(MessageTransform):
        """ThoughtTagStripper — strips <thought>...</thought> from messages."""
        def apply_transform(self, messages: List[Dict]) -> List[Dict]:
            cleaned = []
            for msg in messages:
                content = msg.get("content", "")
                if content and isinstance(content, str) and "<thought>" in content.lower():
                    stripped = re.sub(r'<thought>.*?</thought>', '', content, flags=re.IGNORECASE | re.DOTALL)
                    stripped = re.sub(r'\n{3,}', '\n\n', stripped).strip()
                    if stripped:
                        cleaned.append({**msg, "content": stripped})
                    else:
                        cleaned.append(msg)
                else:
                    cleaned.append(msg)
            return cleaned
            
        def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
            had_effect = any(
                "<thought>" in str(m.get("content", "")).lower()
                for m in pre_transform_messages
            )
            return "ThoughtTagStripper applied", had_effect

    class EntityPreservingCompression(MessageTransform):
        """Middleware to compress context while retaining critical system constraints/risk factors."""
        def apply_transform(self, messages: List[Dict]) -> List[Dict]:
            if len(messages) <= 4:
                return messages
            compressed = []
            n_trim = len(messages) - 4
            PRESERVE_SIGNALS = {
                "CONSTRAINT", "RISK", "PINNED", "IS_HIGH_RISK",
                "VETO", "BLOCK", "OPPOSE", "SUSTAIN", "ASSUMPTION",
                "HIGH_RISK", "REJECT", "AUDIT", "CVE", "COMPLIANCE"
            }
            for msg in messages[:n_trim]:
                content = str(msg.get("content", ""))
                content_upper = content.upper()
                if any(sig in content_upper for sig in PRESERVE_SIGNALS):
                    compressed.append({**msg, "content": f"[PRESERVED-SIGNAL] {content[:400]}..."})
                else:
                    compressed.append({**msg, "content": "[COMPRESSED]"})
            compressed.extend(messages[-4:])
            return compressed
            
        def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
            had_effect = len(pre_transform_messages) > 4
            return "EntityPreservingCompression applied", had_effect

except ImportError:
    # Handle environment where transform capabilities are missing
    class ThoughtTagStripper:
        pass
    class EntityPreservingCompression:
        pass

def setup_token_sparsification_middleware(agents: List[Any]):
    """Attach ThoughtTagStripper and EntityPreservingCompression message transforms to AG2 agents."""
    try:
        from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
        compressor = TransformMessages(transforms=[ThoughtTagStripper(), EntityPreservingCompression()])
        for ag in agents:
            compressor.add_to_agent(ag)
        logger.info("Token Sparsification Middleware activated: ThoughtTagStripper + Entity-preserving compression running.")
    except Exception as e:
        logger.warning(f"Native TransformMessages missing or failed to inject: {e}")

def create_redundancy_hook(feature_description: str):
    """
    Creates a stateful redundancy check hook for AutoGen chats.
    Flags when an agent is restating the original feature brief and registers a retry.
    """
    _feature_desc_phrases = set()
    if feature_description:
        # Extract key phrases (4+ word blocks) from the feature description
        _desc_words = feature_description.lower().split()
        for i in range(len(_desc_words) - 3):
            _feature_desc_phrases.add(' '.join(_desc_words[i:i+4]))
            
    _v28_retry_count = {}  # Track retries per agent to cap at 1

    def _v28_redundancy_check(sender, message, recipient, silent):
        """Detect and flag messages that restate the feature brief."""
        content = message if isinstance(message, str) else (message.get('content', '') if isinstance(message, dict) else '')
        if not content or len(content) < 100:
            return message
        
        sender_name = getattr(sender, 'name', 'Unknown')
        if _v28_retry_count.get(sender_name, 0) >= 1:
            return message  # Already retried once, let it through
        
        content_lower = content.lower()
        overlap_count = sum(1 for phrase in _feature_desc_phrases if phrase in content_lower)
        overlap_ratio = overlap_count / max(len(_feature_desc_phrases), 1)
        
        if overlap_ratio > 0.4:
            _v28_retry_count[sender_name] = 1
            logger.info("Redundancy check triggered for agent: %s (overlap ratio: %.2f)", sender_name, overlap_ratio)
            # Injecting a corrective instruction as a system retry
            return {
                "content": (
                    "[SYSTEM RETRY REQUEST] Your response contains high structural redundancy with the original proposal description. "
                    "Do NOT restate the description. Focus on original critique, technical challenges, or economic impact."
                )
            }
        return message

    return _v28_redundancy_check
