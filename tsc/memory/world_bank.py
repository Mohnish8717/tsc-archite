import logging
import asyncio
from typing import List, Dict, Any, Optional

try:
    from hindsight import HindsightClient
    HINDSIGHT_AVAILABLE = True
except ImportError:
    HINDSIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

class WorldDataBank:
    """
    The Universal 'World Data Bank' (bank_id="tsc-world").
    Replaces global Zep-based RAG. 
    Stores foundational documents (e.g. 1000-page specs) and provides a unified
    Hindsight reflection interface accessible to all agents and gates globally.
    """
    
    def __init__(self, fallback_mode: bool = False):
        self.bank_id = "tsc-world"
        self._fallback_mode = fallback_mode or (not HINDSIGHT_AVAILABLE)
        
        if not self._fallback_mode:
            try:
                self.client = HindsightClient()
            except Exception as e:
                logger.warning(f"Hindsight SDK init failed: {e}. Falling back to embedded WorldBank.")
                self._fallback_mode = True
                
        self._embedded_bank = [] # Embedded fallback for local runs without hindsight API keys
    
    async def initialize_session(self, session_id: str = "tsc-world") -> None:
        """Create or connect to the universal global bank."""
        if self._fallback_mode:
            logger.info("Initializing World Data Bank (EMBEDDED FALLBACK MODE)")
            return
            
        logger.info(f"Connecting to Universal Hindsight World Data Bank: {session_id}")
        self.bank_id = session_id
        try:
            val = await asyncio.to_thread(self.client.create_bank, bank_id=self.bank_id)
            logger.info(f"World Data Bank Connection Successful: {val}")
        except Exception as e:
            if "already exists" in str(e).lower() or "409" in str(e):
                logger.info("Universal World Data Bank already exists in Hindsight Cloud. Connected.")
            else:
                logger.error(f"Failed to create World Data Bank: {e}")
                self._fallback_mode = True

    async def ingest_document(self, document_text: str, document_name: str) -> None:
        """Insert a massively long foundation document into the universal bank."""
        if self._fallback_mode:
            self._embedded_bank.append({"name": document_name, "content": document_text})
            logger.info(f"Ingested {document_name} into EMBEDDED World Bank.")
            return
            
        try:
            logger.info(f"Ingesting '{document_name}' into Hindsight World Bank ({len(document_text)} chars)")
            # Hindsight natively chunks large unstructured strings on ingest via retain
            await asyncio.to_thread(
                self.client.retain,
                bank_id=self.bank_id,
                message=document_text,
                metadata={"type": "foundational_document", "document_name": document_name}
            )
        except Exception as e:
            logger.error(f"Failed to ingest document {document_name}: {e}")

    async def query_world_bank(self, query: str) -> str:
        """RAG Replacement for Agents: Ask Hindsight a question against the 1000-page bank."""
        if self._fallback_mode:
            if not self._embedded_bank:
                return "No documents available in World Bank."
            # Dummy local return
            return f"[EMBEDDED WORLD BANK EXTRACT]: Information related to '{query}' is not precisely indexed locally."
            
        try:
            logger.info(f"Agent querying World Bank: '{query}'")
            # Using reflect as our RAG retriever
            response = await asyncio.to_thread(
                self.client.reflect,
                bank_id=self.bank_id,
                query=query
            )
            # The SDK returns a pydantic structured response, extract just the answer string
            if hasattr(response, 'answer'):
                return response.answer
            return str(response)
            
        except Exception as e:
            logger.error(f"World Bank query failed: {e}")
            return f"Error retrieving from World Bank: {e}"
