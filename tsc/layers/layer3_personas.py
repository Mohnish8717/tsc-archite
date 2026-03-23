"""Layer 3: Persona Selection & Generation.

Identifies relevant stakeholders, retrieves context, and generates
detailed psychological profiles using the configured LLM.

Critical fixes:
  1. Comprehensive input validation (feature, company, graph, bundle)
  2. LLM stakeholder response parsing with validation and type coercion
  3. Context retrieval with validation and type coercion
  4. Robust MBTI extraction with context-aware confidence
  5. Robust list item extraction with format flexibility
  6. Intelligent fallback profile when LLM fails
  7. Evidence-based profile confidence calculation

Optimizations:
  1. Persona caching to prevent redundant LLM calls
  2. Parallel context retrieval (asyncio.gather)
  3. Parallel profile generation (asyncio.gather)
  4. Smarter default stakeholders based on feature analysis
  5. All 16 MBTI type descriptions
  6. Comprehensive diagnostics endpoint
"""

from __future__ import annotations

import asyncio
import uuid
import logging
import re
import time
from collections import Counter
from typing import Any, Optional

import numpy as np

from tsc.llm.base import LLMClient
from tsc.llm.prompts import (
    PERSONA_SYSTEM,
    PERSONA_SYSTEM_GROUNDED,
    PERSONA_USER,
    STAKEHOLDER_SELECTION_SYSTEM,
    STAKEHOLDER_SELECTION_USER,
    EXTERNAL_STAKEHOLDER_SELECTION_SYSTEM,
    EXTERNAL_STAKEHOLDER_SELECTION_USER,
)
from tsc.memory.graph_store import GraphStore
from tsc.repositories.persona_repository import PersonaRepository
from tsc.caching.lru_cache import PersonaCache
from tsc.db.models import InternalPersona, ExternalPersona
from tsc.models.chunks import ProblemContextBundle
from tsc.models.graph import GraphEntity, KnowledgeGraph
from tsc.models.inputs import CompanyContext, FeatureProposal
from tsc.models.personas import (
    CommunicationStyle,
    DecisionPattern,
    EmotionalTriggers,
    FinalPersona,
    PredictedStance,
    PsychologicalProfile,
    Stakeholder,
    StakeholderContextBundle,
)

logger = logging.getLogger(__name__)

# All 16 MBTI type descriptions (OPT-5)
_MBTI_DESCRIPTIONS: dict[str, str] = {
    "ESTJ": (
        "Organized administrator; values rules, efficiency, and tradition. "
        "Decisive, systematic, reliable decision-maker who focuses on "
        "present facts and proven methods."
    ),
    "ESFJ": (
        "Cooperative supporter; values harmony, loyalty, and service. "
        "People-oriented team player who ensures group needs are met "
        "and maintains group harmony."
    ),
    "ENTJ": (
        "Strategic leader; values competence, efficiency, and results. "
        "Direct, ambitious, focused on future possibilities and "
        "long-term strategy."
    ),
    "ENFJ": (
        "Persuasive leader; values group goals, harmony, and growth. "
        "Charismatic motivator who inspires others and facilitates "
        "group development."
    ),
    "ESFP": (
        "Enthusiastic performer; values excitement, fun, and people. "
        "Spontaneous, energetic, brings enthusiasm and adaptability "
        "to any situation."
    ),
    "ESTP": (
        "Pragmatic operator; values action, results, and efficiency. "
        "Bold, practical, energetic troubleshooter who focuses on "
        "immediate impact."
    ),
    "ENTP": (
        "Innovative debater; values ideas, challenge, and competence. "
        "Quick-thinking, adaptable, enjoys intellectual sparring and "
        "exploring new possibilities."
    ),
    "ENFP": (
        "Creative ideator; values innovation, freedom, and possibilities. "
        "Enthusiastic, imaginative champion who sees connections "
        "and inspires others."
    ),
    "ISTJ": (
        "Reliable guardian; values duty, loyalty, and responsibility. "
        "Dependable, organized, thorough—the backbone of any "
        "organization with strong principles."
    ),
    "ISFJ": (
        "Protective caregiver; values service, loyalty, and harmony. "
        "Modest, dedicated supporter who works behind the scenes to "
        "help others."
    ),
    "INTJ": (
        "Analytical strategist; values competence, knowledge, and "
        "strategy. Independent thinker focused on understanding complex "
        "systems and long-term vision."
    ),
    "INFJ": (
        "Visionary counselor; values meaning, insight, and compassion. "
        "Principled, idealistic advocate for what matters most to them "
        "and others."
    ),
    "ISFP": (
        "Sensitive artist; values authenticity, beauty, and harmony. "
        "Modest, kind, creates or preserves beauty in their own way."
    ),
    "ISTP": (
        "Logical troubleshooter; values competence, autonomy, and "
        "efficiency. Independent analyst who solves problems through "
        "direct action."
    ),
    "INFP": (
        "Idealistic mediator; values authenticity, meaning, and growth. "
        "Thoughtful, principled, motivated by personal values and "
        "helping others grow."
    ),
    "INTP": (
        "Logical analyst; values ideas, knowledge, and competence. "
        "Independent thinker who explores abstract ideas and seeks "
        "underlying principles."
    ),
}


class PersonaGenerator:
    """Layer 3: Select stakeholders and generate psychological profiles."""
    def __init__(
        self,
        llm_client: LLMClient,
        graph_store: GraphStore,
        persona_repo: Optional[PersonaRepository] = None,
        persona_cache: Optional[PersonaCache] = None
    ) -> None:
        self._llm = llm_client
        self._graph_store = graph_store
        self._persona_repo = persona_repo
        self._cache = persona_cache or PersonaCache()
        logger.info("PersonaGenerator initialized (PostgreSQL persistent layer & LRU cache)")

    # ═════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════

    async def process(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
    ) -> list[FinalPersona]:
        """Execute full Layer 3 pipeline with all fixes and optimizations."""
        t0 = time.time()
        logger.info("Layer 3: Generating stakeholder personas")

        # Step 0: Validate inputs (CRITICAL FIX #1)
        self._validate_inputs(feature, company, graph, bundle)
        logger.info("✓ Validated all inputs")

        # Step 1: Select INTERNAL stakeholders
        internal_stakeholders = await self._select_internal_stakeholders(
            feature, company, graph
        )
        logger.info("✓ Selected %d internal stakeholders", len(internal_stakeholders))
        
        # Step 2: Select EXTERNAL stakeholders (NEW)
        external_stakeholders = await self._select_external_stakeholders(
            feature, company, graph
        )
        logger.info("✓ Selected %d external stakeholders", len(external_stakeholders))
        
        # Step 3: Merge and deduplicate
        stakeholders = self._merge_stakeholder_lists(
            internal_stakeholders, external_stakeholders
        )
        logger.info("✓ Merged to %d total stakeholders", len(stakeholders))

        if not stakeholders:
            raise ValueError("No stakeholders selected (internal or external)")

        # Get top entities once (for all profiles)
        top_entities = self._get_top_entities(graph)
        logger.info("✓ Retrieved top %d entities", len(top_entities))

        # Step 3.2: Retrieve context (OPT-2: parallel)
        context_tasks = [self._retrieve_context(sh) for sh in stakeholders]
        contexts = await asyncio.gather(*context_tasks, return_exceptions=True)

        valid_pairs: list[tuple[Stakeholder, StakeholderContextBundle]] = []
        for sh, ctx in zip(stakeholders, contexts):
            if isinstance(ctx, Exception):
                logger.warning(
                    "Failed to retrieve context for %s: %s", sh.name, ctx
                )
            else:
                valid_pairs.append((sh, ctx))

        if len(valid_pairs) < len(stakeholders):
            failed = len(stakeholders) - len(valid_pairs)
            logger.warning(
                "Failed to retrieve context for %d stakeholders", failed
            )
        logger.info(
            "✓ Retrieved context for %d stakeholders", len(valid_pairs)
        )

        # Step 3.3: Generate profiles (OPT-3: parallel with caching)
        profile_tasks = [
            self._generate_profile(sh, ctx, feature, company, top_entities)
            for sh, ctx in valid_pairs
        ]
        results = await asyncio.gather(*profile_tasks, return_exceptions=True)

        valid_personas: list[FinalPersona] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Failed to generate a persona: %s", result)
            else:
                valid_personas.append(result)

        if len(valid_personas) < len(valid_pairs):
            failed = len(valid_pairs) - len(valid_personas)
            logger.warning(
                "Failed to generate personas for %d stakeholders", failed
            )
        logger.info("✓ Generated %d personas", len(valid_personas))

        # Step 4: Persist to Database (Phase 2.1) and Cache
        if self._persona_repo:
            try:
                company_id = company.company_id
                internal_personas_to_cache = []
                external_personas_to_cache = []

                for persona in valid_personas:
                    persona_dict = {
                        "name": persona.name,
                        "role": persona.role,
                        "title": persona.title,
                        "mbti_type": persona.psychological_profile.mbti_type,
                        "psychological_profile": persona.psychological_profile.model_dump(),
                        "confidence_score": persona.confidence_score,
                    }
                    
                    if persona.persona_type == "INTERNAL":
                        if company_id:
                            persona_id = await self._persona_repo.save_internal_persona(company_id, persona_dict)
                            logger.info("Saved internal persona %s (ID: %s)", persona.name, persona_id)
                            internal_personas_to_cache.append(persona)
                    else:
                        persona_id = await self._persona_repo.save_external_persona(persona_dict)
                        logger.info("Saved external persona %s (ID: %s)", persona.name, persona_id)
                        external_personas_to_cache.append(persona)
                
                if company_id and internal_personas_to_cache:
                    await self._cache.set_list(f"personas:company:{company_id}:internal", internal_personas_to_cache)
                if external_personas_to_cache:
                    # Assuming external personas are cached by segment or a generic key
                    # For now, let's just cache them individually if needed, or by a generic "all_external"
                    # A more specific key would be needed for segment-based retrieval
                    pass # Handled by _generate_profile caching
            except Exception as e:
                logger.error("Failed to persist personas: %s", e)

        # Diagnostics
        diagnostics = await self.get_layer3_diagnostics(valid_personas)

        logger.info(
            "Layer 3 complete: %d total personas "
            "(%d internal, %d external), confidence=%.2f (%.1fs)",
            len(valid_personas),
            sum(1 for p in valid_personas if p.persona_type == "INTERNAL"),
            sum(1 for p in valid_personas if p.persona_type == "EXTERNAL"),
            diagnostics.get("confidence", {}).get("mean", 0),
            time.time() - t0,
        )

        return valid_personas

    async def get_internal_personas_for_company(self, company_id: uuid.UUID) -> list[FinalPersona]:
        """Retrieve all personas for a company from the database."""
        # 1. Try cache (PersonaCache is async now)
        cached_personas = await self._cache.get_list(f"personas:company:{company_id}:internal")
        if cached_personas:
            logger.info("Using %d personas from cache for company %s", len(cached_personas), company_id)
            return cached_personas

        # 2. Try Database if repository is available
        if not self._persona_repo:
            return []
            
        logger.info("Retrieving internal personas for company %s from DB", company_id)
        db_personas = await self._persona_repo.get_internal_personas_by_company(company_id)
        final_personas = [self._map_to_final_persona(p, is_internal=True) for p in db_personas]
        
        # Cache the results
        if final_personas:
            await self._cache.set_list(f"personas:company:{company_id}:internal", final_personas)
        return final_personas

    async def get_external_personas_for_market(self, segment_type: str) -> list[FinalPersona]:
        """Retrieve external personas for a market segment from the database."""
        # 1. Try cache
        cached_personas = await self._cache.get_list(f"personas:segment:{segment_type}:external")
        if cached_personas:
            logger.info("Using %d personas from cache for segment %s", len(cached_personas), segment_type)
            return cached_personas

        # 2. Try Database if repository is available
        if not self._persona_repo:
            return []
            
        logger.info("Retrieving external personas for segment %s from DB", segment_type)
        db_personas = await self._persona_repo.get_external_personas_by_segment(segment_type)
        final_personas = [self._map_to_final_persona(p, is_internal=False) for p in db_personas]
        
        # Cache the results
        if final_personas:
            await self._cache.set_list(f"personas:segment:{segment_type}:external", final_personas)
        return final_personas

    async def get_persona_by_id(self, persona_id: uuid.UUID, is_internal: bool = True) -> Optional[FinalPersona]:
        """Retrieve a single persona by ID from the database."""
        cache_key = f"persona:{persona_id}:{'internal' if is_internal else 'external'}"
        cached_persona = await self._cache.get(cache_key)
        if cached_persona:
            logger.info("Using cached persona %s", persona_id)
            return cached_persona

        if not self._persona_repo:
            return None
            
        logger.info("Retrieving persona %s from DB", persona_id)
        if is_internal:
            result = await self._persona_repo.get_internal_persona(persona_id)
        else:
            result = await self._persona_repo.get_external_persona(persona_id)
            
        if result:
            final_persona = self._map_to_final_persona(result, is_internal)
            await self._cache.set(cache_key, final_persona)
            return final_persona
        return None

    def _map_to_final_persona(self, db_persona: Any, is_internal: bool) -> FinalPersona:
        """Map a database record to a FinalPersona Pydantic model."""
        profile_data = db_persona.psychological_profile
        
        # Reconstruct complex profile object
        profile = PsychologicalProfile(
            mbti=db_persona.mbti_type or "",
            mbti_description=profile_data.get("mbti_description", ""),
            key_traits=profile_data.get("key_traits", []),
            emotional_triggers=EmotionalTriggers(**profile_data.get("emotional_triggers", {})),
            communication_style=CommunicationStyle(**profile_data.get("communication_style", {})),
            decision_pattern=DecisionPattern(**profile_data.get("decision_pattern", {})),
            predicted_stance=PredictedStance(**profile_data.get("predicted_stance", {})),
            questions_they_will_ask=profile_data.get("questions_they_will_ask", []),
            full_profile_text=profile_data.get("full_profile_text", "")
        )
        
        return FinalPersona(
            name=db_persona.name,
            role=db_persona.role,
            psychological_profile=profile,
            confidence_score=db_persona.confidence_score,
            persona_type="INTERNAL" if is_internal else "EXTERNAL"
        )

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #1: Input Validation
    # ═════════════════════════════════════════════════════════════════

    def _validate_inputs(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
        bundle: ProblemContextBundle,
    ) -> None:
        """Comprehensive input validation."""
        # Feature validation
        if not feature or not feature.title:
            raise ValueError("Feature proposal missing title")

        if not feature.description or len(feature.description) < 10:
            raise ValueError("Feature description too short (<10 chars)")

        if not feature.target_users:
            logger.warning("Feature has no target_users specified")

        # Company validation
        if not company or not company.company_name:
            raise ValueError("Company context missing company_name")

        if not company.team_size or company.team_size < 1:
            raise ValueError("Company team_size invalid")

        if not company.current_priorities:
            logger.warning("Company has no current_priorities")

        # Graph validation
        if not graph or not graph.nodes:
            raise ValueError("Knowledge graph has no entities")

        if len(graph.nodes) < 5:
            logger.warning(
                "Knowledge graph has few entities: %d", len(graph.nodes)
            )

        if not graph.edges:
            logger.warning("Knowledge graph has no relationships")

        # Bundle validation
        if not bundle or not bundle.chunks:
            raise ValueError("Problem context bundle is empty")

        if len(bundle.chunks) < 10:
            logger.warning(
                "Problem context bundle has few chunks: %d",
                len(bundle.chunks),
            )

    # ═════════════════════════════════════════════════════════════════
    # Step 3.1: Stakeholder Selection (CRITICAL FIX #2)
    # ═════════════════════════════════════════════════════════════════

    async def _select_internal_stakeholders(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
    ) -> list[Stakeholder]:
        """Select stakeholders with robust LLM response handling."""
        try:
            top_entities = self._get_top_entities(graph)

            prompt = STAKEHOLDER_SELECTION_USER.render(
                feature=feature,
                company=company,
                top_entities=[
                    {
                        "name": e.name,
                        "type": e.type,
                        "mentions": e.mentions,
                        "average_urgency": e.average_urgency,
                    }
                    for e in top_entities
                ],
            )

            result = await self._llm.analyze(
                system_prompt=STAKEHOLDER_SELECTION_SYSTEM,
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=2000,
            )

            if not result:
                raise ValueError("LLM returned empty result")

            # Parse with validation
            stakeholders = self._parse_stakeholder_response(
                result, feature, company
            )

            if not stakeholders:
                logger.warning(
                    "LLM returned no stakeholders, using defaults"
                )
                stakeholders = self._fill_default_internal_stakeholders(
                    [], feature, company
                )

            # Ensure 3-5 stakeholders
            if len(stakeholders) < 3:
                logger.info(
                    "Only %d stakeholders, filling defaults",
                    len(stakeholders),
                )
                stakeholders = self._fill_default_internal_stakeholders(
                    stakeholders, feature, company
                )

            stakeholders = stakeholders[:5]

            # Deduplicate
            stakeholders = self._deduplicate_stakeholders(stakeholders)

            logger.info("✓ Selected %d stakeholders", len(stakeholders))
            return stakeholders

        except Exception as e:
            logger.error("Stakeholder selection failed: %s", e)
            # Fallback to defaults
            return self._fill_default_internal_stakeholders([], feature, company)

    async def _select_external_stakeholders(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
        graph: KnowledgeGraph,
    ) -> list[Stakeholder]:
        """Select EXTERNAL customer personas for market analysis."""
        try:
            top_entities = self._get_top_entities(graph)
            
            prompt = EXTERNAL_STAKEHOLDER_SELECTION_USER.render(
                feature=feature,
                company=company,
                target_users=feature.target_users,
                target_user_count=feature.target_user_count,
                top_entities=[
                    {
                        "name": e.name,
                        "type": e.type,
                        "mentions": e.mentions,
                        "average_urgency": e.average_urgency,
                    }
                    for e in top_entities
                ],
            )
            
            result = await self._llm.analyze(
                system_prompt=EXTERNAL_STAKEHOLDER_SELECTION_SYSTEM,
                user_prompt=prompt,
                temperature=0.6,
                max_tokens=2000,
            )
            
            if not result:
                raise ValueError("LLM returned empty result")
                
            stakeholders = self._parse_external_stakeholder_response(result)
            
            if not stakeholders:
                logger.warning("LLM returned no external stakeholders")
                stakeholders = self._fill_default_external_stakeholders(
                    feature, company
                )
            
            logger.info("✓ Selected %d external personas", len(stakeholders))
            return stakeholders
        
        except Exception as e:
            logger.error("External stakeholder selection failed: %s", e)
            return self._fill_default_external_stakeholders(feature, company)

    def _parse_external_stakeholder_response(
        self,
        result: dict[str, Any],
    ) -> list[Stakeholder]:
        """Parse LLM response for external stakeholders."""
        import numpy as np
        stakeholders: list[Stakeholder] = []
        raw = result.get("external_customers", [])
        
        if not isinstance(raw, list):
            logger.warning("External stakeholders not list")
            return []
        
        for idx, s in enumerate(raw):
            try:
                name = s.get("name", "").strip()
                if not name:
                    continue
                
                persona_type = s.get("persona_type", "").strip()
                if not persona_type:
                    persona_type = s.get("segment", "Unknown")
                
                try:
                    rel_score = float(s.get("relevance_score", 0.6))
                    rel_score = np.clip(rel_score, 0.0, 1.0)
                except (ValueError, TypeError):
                    rel_score = 0.6
                
                stakeholder = Stakeholder(
                    name=name,
                    role=persona_type,
                    title=s.get("title", persona_type),
                    relevance_score=rel_score,
                    domain_relevance=s.get("use_case", ""),
                    decision_authority=s.get("decision_authority", "medium"),
                    persona_type="EXTERNAL",
                )
                
                stakeholders.append(stakeholder)
            
            except Exception as e:
                logger.debug("Failed to parse external stakeholder %d: %s", idx, e)
                continue
        
        return stakeholders

    def _fill_default_external_stakeholders(
        self,
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> list[Stakeholder]:
        """Create default external customer personas."""
        feature_lower = feature.description.lower()
        
        external_configs = {
            "enterprise_customer": {
                "name": "Enterprise CTO",
                "title": "Chief Technology Officer",
                "segment": "enterprise",
                "use_case": "Large-scale deployment, compliance, security",
                "base_relevance": 0.85,
                "keywords": ["enterprise", "scale", "compliance", "security", "integration"],
            },
            "midmarket_customer": {
                "name": "Mid-Market Operations Manager",
                "title": "VP Operations",
                "segment": "mid-market",
                "use_case": "Operational efficiency, cost control",
                "base_relevance": 0.80,
                "keywords": ["efficiency", "cost", "operations", "process", "optimization"],
            },
            "startup_customer": {
                "name": "Startup Founder",
                "title": "Co-Founder/CEO",
                "segment": "startup",
                "use_case": "Quick deployment, cost-effective, rapid iteration",
                "base_relevance": 0.75,
                "keywords": ["speed", "cost", "startup", "mvp", "rapid"],
            },
            "enduser": {
                "name": "End User",
                "title": "Field Technician / Knowledge Worker",
                "segment": "individual",
                "use_case": "Ease of use, intuitive interface, minimal training",
                "base_relevance": 0.70,
                "keywords": ["user", "ease", "intuitive", "training", "adoption"],
            },
        }
        
        segment_scores: list[tuple[dict, float]] = []
        
        for _seg_id, config in external_configs.items():
            score = config["base_relevance"]
            keyword_matches = sum(
                1 for kw in config["keywords"] if kw in feature_lower
            )
            score += keyword_matches * 0.03
            segment_scores.append((config, min(1.0, score)))
        
        segment_scores.sort(key=lambda x: x[1], reverse=True)
        
        external_personas: list[Stakeholder] = []
        for config, score in segment_scores[:3]:
            external_personas.append(
                Stakeholder(
                    name=config["name"],
                    role=config["segment"],
                    title=config["title"],
                    relevance_score=round(score, 2),
                    domain_relevance=config["use_case"],
                    decision_authority="medium",
                    persona_type="EXTERNAL",
                )
            )
        
        logger.info(
            "Created %d default external personas", len(external_personas)
        )
        return external_personas

    def _merge_stakeholder_lists(
        self,
        internal: list[Stakeholder],
        external: list[Stakeholder],
    ) -> list[Stakeholder]:
        """Merge internal and external stakeholder lists."""
        merged = (internal[:3] + external[:3])[:6]
        types_seen = set()
        diverse = []
        
        for sh in merged:
            sh_type = (getattr(sh, "persona_type", "INTERNAL"), sh.role)
            if sh_type not in types_seen:
                diverse.append(sh)
                types_seen.add(sh_type)
        
        logger.info(
            "Merged: %d internal + %d external → %d total stakeholders",
            len(internal), len(external), len(diverse)
        )
        return diverse

    def _parse_stakeholder_response(
        self,
        result: dict[str, Any],
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> list[Stakeholder]:
        """Parse LLM response with validation and type coercion."""
        stakeholders: list[Stakeholder] = []
        raw_stakeholders = result.get("stakeholders", [])

        if not isinstance(raw_stakeholders, list):
            logger.warning(
                "Stakeholders not a list (type: %s), skipping",
                type(raw_stakeholders).__name__,
            )
            return []

        for idx, s in enumerate(raw_stakeholders):
            try:
                # Validate required fields
                name = s.get("name", "").strip()
                if not name:
                    logger.debug("Stakeholder %d missing name", idx)
                    continue

                role = s.get("role", "").strip()
                if not role:
                    logger.debug("Stakeholder %d missing role", idx)
                    continue

                # Type coercion for relevance_score
                try:
                    rel_score = float(s.get("relevance_score", 0.5))
                    rel_score = max(0.0, min(1.0, rel_score))
                except (ValueError, TypeError):
                    logger.debug(
                        "Invalid relevance_score for %s: %s",
                        name,
                        s.get("relevance_score"),
                    )
                    rel_score = 0.5

                stakeholder = Stakeholder(
                    name=name,
                    role=role,
                    title=s.get("title", role),
                    relevance_score=rel_score,
                    domain_relevance=s.get("domain_relevance", ""),
                    decision_authority=s.get("decision_authority", "medium"),
                )

                # Validate against company context
                if self._validate_stakeholder_against_company(
                    stakeholder, company
                ):
                    stakeholders.append(stakeholder)
                else:
                    logger.debug(
                        "Stakeholder %s failed company validation",
                        stakeholder.name,
                    )

            except Exception as e:
                logger.debug("Failed to parse stakeholder %d: %s", idx, e)
                continue

        logger.info(
            "Parsed %d valid stakeholders from LLM response",
            len(stakeholders),
        )
        return stakeholders

    def _validate_stakeholder_against_company(
        self,
        stakeholder: Stakeholder,
        company: CompanyContext,
    ) -> bool:
        """Validate stakeholder makes sense for company."""
        role_lower = stakeholder.role.lower()

        # Check team size consistency
        if company.team_size and company.team_size < 3:
            if any(
                x in role_lower
                for x in ["cto", "cfo", "vp", "chief", "director"]
            ):
                logger.debug(
                    "Stakeholder %s unlikely for %d-person team",
                    stakeholder.role,
                    company.team_size,
                )
                return False

        return True

    def _deduplicate_stakeholders(
        self, stakeholders: list[Stakeholder]
    ) -> list[Stakeholder]:
        """Remove duplicate/similar stakeholders, keeping higher relevance."""
        seen_roles: dict[str, Stakeholder] = {}

        for sh in stakeholders:
            role_key = sh.role.lower().strip()

            if role_key in seen_roles:
                existing = seen_roles[role_key]
                if sh.relevance_score > existing.relevance_score:
                    seen_roles[role_key] = sh
                    logger.debug(
                        "Replacing %s (lower relevance)", existing.name
                    )
                else:
                    logger.debug("Skipping duplicate role: %s", sh.role)
            else:
                seen_roles[role_key] = sh

        unique = list(seen_roles.values())
        removed = len(stakeholders) - len(unique)

        if removed > 0:
            logger.info(
                "Deduplication: %d → %d stakeholders",
                len(stakeholders),
                len(unique),
            )

        return unique

    # ═════════════════════════════════════════════════════════════════
    # OPT-4: Smarter Default Stakeholders
    # ═════════════════════════════════════════════════════════════════

    def _fill_default_internal_stakeholders(
        self,
        existing: list[Stakeholder],
        feature: FeatureProposal,
        company: CompanyContext,
    ) -> list[Stakeholder]:
        """Create context-aware default stakeholders."""
        feature_lower = feature.description.lower()

        role_configs: dict[str, dict[str, Any]] = {
            "engineering_lead": {
                "name": "Engineering Lead",
                "title": "Senior Engineer",
                "base_relevance": 0.85,
                "keywords": [
                    "technical",
                    "engineering",
                    "build",
                    "implement",
                    "architecture",
                ],
            },
            "product_manager": {
                "name": "Product Manager",
                "title": "Senior PM",
                "base_relevance": 0.80,
                "keywords": [
                    "market",
                    "user",
                    "feature",
                    "product",
                    "demand",
                    "customer",
                ],
            },
            "finance_lead": {
                "name": "Finance Lead",
                "title": "CFO",
                "base_relevance": 0.75,
                "keywords": ["budget", "cost", "roi", "financial", "revenue"],
            },
            "customer_success": {
                "name": "VP Customer Success",
                "title": "VP CS",
                "base_relevance": 0.70,
                "keywords": [
                    "customer",
                    "retention",
                    "support",
                    "satisfaction",
                    "churn",
                ],
            },
        }

        # Score each role based on feature content
        role_scores: list[tuple[dict[str, Any], float]] = []
        for _role_id, config in role_configs.items():
            score = config["base_relevance"]
            keyword_matches = sum(
                1
                for kw in config["keywords"]
                if kw in feature_lower
            )
            score += keyword_matches * 0.03
            role_scores.append((config, min(1.0, score)))

        # Sort by score descending
        role_scores.sort(key=lambda x: x[1], reverse=True)

        existing_names = {s.name for s in existing}

        for config, score in role_scores:
            if config["name"] in existing_names or len(existing) >= 5:
                continue

            existing.append(
                Stakeholder(
                    name=config["name"],
                    role=config["name"],
                    title=config["title"],
                    relevance_score=round(score, 2),
                    domain_relevance=", ".join(config["keywords"][:3]),
                    decision_authority="high",
                )
            )

        logger.info(
            "Filled to %d stakeholders based on feature context",
            len(existing),
        )

        return existing

    # ═════════════════════════════════════════════════════════════════
    # Step 3.2: Context Retrieval (CRITICAL FIX #3)
    # ═════════════════════════════════════════════════════════════════

    async def _retrieve_context(
        self, stakeholder: Stakeholder
    ) -> StakeholderContextBundle:
        """Retrieve context for both internal and external personas."""
        is_external = getattr(stakeholder, "persona_type", "INTERNAL") == "EXTERNAL"
        
        try:
            if is_external:
                ctx = await self._graph_store.retrieve_customer_context(
                    stakeholder.role,
                    stakeholder.domain_relevance,
                )
            else:
                ctx = await self._graph_store.retrieve_stakeholder_context(
                    stakeholder.name, stakeholder.role
                )
        except Exception as e:
            logger.warning(
                "Failed to retrieve %s context for %s: %s",
                "external" if is_external else "internal",
                stakeholder.name,
                e,
            )
            ctx = {}

        # Validate and normalize each field
        personal_facts = self._validate_context_list(
            ctx.get("personal_facts", []),
            field_name="personal_facts",
            max_items=20,
        )
        org_context = self._validate_context_list(
            ctx.get("org_context", []),
            field_name="org_context",
            max_items=15,
        )
        constraint_context = self._validate_context_list(
            ctx.get("constraint_context", []),
            field_name="constraint_context",
            max_items=15,
        )

        logger.info(
            "Retrieved context for %s: %d facts, %d org, %d constraints",
            stakeholder.name,
            len(personal_facts),
            len(org_context),
            len(constraint_context),
        )

        return StakeholderContextBundle(
            stakeholder=stakeholder,
            personal_facts=personal_facts,
            org_context=org_context,
            constraint_context=constraint_context,
        )

    def _validate_context_list(
        self,
        data: Any,
        field_name: str = "context",
        max_items: int = 20,
    ) -> list[Any]:
        """Validate context list and preserve rich dictionaries."""
        if not data:
            return []

        if not isinstance(data, (list, tuple)):
            logger.debug(
                "%s not a list (type: %s), converting",
                field_name,
                type(data).__name__,
            )
            data = [data]

        validated: list[Any] = []
        for idx, item in enumerate(data[:max_items]):
            try:
                if not item:
                    continue
                # Preserve rich dictionaries if they follow the expected pattern
                if isinstance(item, dict) and "text" in item:
                    # Optional: truncation for safety
                    if len(item["text"]) > 1000:
                        item["text"] = item["text"][:1000]
                    validated.append(item)
                else:
                    item_str = str(item).strip()
                    if len(item_str) < 3:
                        continue
                    if len(item_str) > 500:
                        item_str = item_str[:500]
                    validated.append(item_str)
            except Exception as e:
                logger.debug(
                    "Failed to validate %s item %d: %s", field_name, idx, e
                )
                continue

        return validated

    # ═════════════════════════════════════════════════════════════════
    # Step 3.3: Profile Generation (CRITICAL FIX #6, OPT-1)
    # ═════════════════════════════════════════════════════════════════

    async def _generate_profile(
        self,
        stakeholder: Stakeholder,
        context: StakeholderContextBundle,
        feature: FeatureProposal,
        company: CompanyContext,
        top_entities: list[GraphEntity],
    ) -> FinalPersona:
        """Generate profile with caching and fallback."""
        # OPT-1: Check cache
        cache_key = f"profile:{stakeholder.name.lower()}_{feature.title.lower()}"
        cached = await self._cache.get(cache_key)
        if cached:
            logger.info(
                "Using cached persona for %s (confidence: %.2f)",
                stakeholder.name,
                cached.confidence_score,  # Changed from profile_confidence to consistency
            )
            return cached

        # Generate
        persona = await self._generate_profile_impl(
            stakeholder, context, feature, company, top_entities
        )

        # Cache it
        await self._cache.set(cache_key, persona)

        return persona

    def _build_grounded_persona_prompt(
        self,
        stakeholder: Stakeholder,
        context: StakeholderContextBundle,
        company: CompanyContext,
    ) -> str:
        """Build a prompt that organizes facts by sentiment and requires citations."""
        # Organize facts
        pos_facts = [
            f["text"]
            for f in context.personal_facts
            if isinstance(f, dict) and f.get("sentiment") == "POSITIVE"
        ]
        neg_facts = [
            f["text"]
            for f in context.personal_facts
            if isinstance(f, dict) and f.get("sentiment") == "NEGATIVE"
        ]
        neut_facts = [
            f["text"] if isinstance(f, dict) else str(f)
            for f in context.personal_facts
            if not isinstance(f, dict) or f.get("sentiment") not in ["POSITIVE", "NEGATIVE"]
        ]

        prompt = f"""Generate a grounded persona for {stakeholder.name}, {stakeholder.role} at {company.name}.

EVIDENCE DATA:
Positives/Drivers:
{chr(10).join(f"- {f}" for f in pos_facts[:15])}

Pain Points/Challenges:
{chr(10).join(f"- {f}" for f in neg_facts[:15])}

Other Factual Context:
{chr(10).join(f"- {f}" for f in neut_facts[:20])}

ORGANIZATIONAL CONTEXT:
{chr(10).join(f"- {c}" for c in context.org_context[:10])}

CONSTRAINTS & URGENCY:
{chr(10).join(f"- {c}" for c in context.constraint_context[:10])}

INSTRUCTIONS:
1. Identify the core personality traits supported by these specific facts.
2. For each trait, provide a citation in the format [Fact: brief snippet].
3. Determine the likely MBTI type ONLY if supported by evidence.
4. Predict their stance on new features based on their documented pain points.
"""
        return prompt

    def _validate_grounding(
        self, profile_text: str, context: StakeholderContextBundle
    ) -> float:
        """Verify that the generated profile refers to provided facts."""
        if not profile_text or not context.personal_facts:
            return 0.5

        facts = [
            (f["text"] if isinstance(f, dict) else str(f)).lower()
            for f in context.personal_facts
        ]
        citations = re.findall(r"\[Fact: (.*?)\]", profile_text)

        if not citations:
            return 0.3  # Hallucination risk

        valid_citations = 0
        for cite in citations:
            cite_lower = cite.lower()
            if any(cite_lower in f or f in cite_lower for f in facts):
                valid_citations += 1

        quality = valid_citations / len(citations) if citations else 0
        return round(max(0.1, quality), 2)

    async def _generate_profile_impl(
        self,
        stakeholder: Stakeholder,
        context: StakeholderContextBundle,
        feature: FeatureProposal,
        company: CompanyContext,
        top_entities: list[GraphEntity],
    ) -> FinalPersona:
        """Core profile generation with fallback."""
        profile_text: str = ""

        try:
            # Use grounded prompt builder
            user_prompt = self._build_grounded_persona_prompt(
                stakeholder, context, company
            )

            profile_text = await self._llm.generate(
                system_prompt=PERSONA_SYSTEM_GROUNDED,
                user_prompt=user_prompt,
                model="gpt-4o",  # Use a strong model for complex grounding
                temperature=0.2,
                max_tokens=4000
            )

            # Validate grounding quality
            grounding_score = self._validate_grounding(profile_text, context)
            logger.info(
                "✓ Grounded Profile for %s: %d words, quality=%.2f",
                stakeholder.name,
                len(profile_text.split()),
                grounding_score,
            )

            if not profile_text or len(profile_text.split()) < 100:
                logger.warning(
                    "Profile text too short (%d words) for %s, using minimal",
                    len(profile_text.split()) if profile_text else 0,
                    stakeholder.name,
                )
                profile_text = self._generate_minimal_profile_text(
                    stakeholder, feature, context
                )

        except Exception as e:
            logger.warning(
                "Profile generation failed for %s: %s, using minimal",
                stakeholder.name,
                e,
            )
            profile_text = self._generate_minimal_profile_text(
                stakeholder, feature, context
            )

        # Parse into structured format
        profile = self._parse_profile(profile_text, stakeholder, feature)

        # Calculate confidence (CRITICAL FIX #7)
        confidence = self._estimate_profile_confidence(
            stakeholder, context, profile_text
        )

        return FinalPersona(
            name=stakeholder.name,
            role=stakeholder.role,
            psychological_profile=profile,
            evidence_sources=list(
                set(
                    f["source"]
                    for f in context.personal_facts
                    if isinstance(f, dict)
                )
            ),
            profile_word_count=len(profile_text.split()),
            profile_confidence=confidence,
            persona_type=getattr(stakeholder, "persona_type", "INTERNAL"),
            grounding_quality=grounding_score,
        )

    def _generate_minimal_profile_text(
        self,
        stakeholder: Stakeholder,
        feature: FeatureProposal,
        context: StakeholderContextBundle,
    ) -> str:
        """Generate minimal profile when LLM fails."""
        facts_summary = "\n".join(
            f"- {f}" for f in context.personal_facts[:5]
        )
        org_summary = "\n".join(f"- {o}" for o in context.org_context[:3])

        return (
            f"PERSONALITY PROFILE: {stakeholder.name}\n\n"
            f"Role: {stakeholder.role}\n"
            f"Title: {stakeholder.title}\n"
            f"Relevance: {stakeholder.relevance_score:.2f}\n"
            f"Authority: {stakeholder.decision_authority}\n\n"
            f"MBTI Type: ENTJ (Default — profile generation failed)\n\n"
            f"Key Traits:\n"
            f"- {stakeholder.domain_relevance}\n"
            f"- Decision maker with {stakeholder.decision_authority} "
            f"authority\n"
            f"- Evaluating: {feature.title}\n\n"
            f"Personal Context:\n{facts_summary}\n\n"
            f"Organizational Context:\n{org_summary}\n\n"
            f"Predicted Stance: CONDITIONAL_APPROVE\n\n"
            f"This is a minimal profile generated due to LLM failure.\n"
        )

    # ═════════════════════════════════════════════════════════════════
    # Profile Parsing
    # ═════════════════════════════════════════════════════════════════

    def _parse_profile(
        self,
        text: str,
        stakeholder: Stakeholder,
        feature: FeatureProposal,
    ) -> PsychologicalProfile:
        """Parse free-form profile text into structured PsychologicalProfile."""
        # CRITICAL FIX #4: Robust MBTI extraction
        mbti = self._extract_mbti(text)

        return PsychologicalProfile(
            mbti=mbti,
            mbti_description=self._get_mbti_description(mbti),
            key_traits=self._extract_list_items(
                text, "trait", "strength", "characteristic"
            ),
            emotional_triggers=EmotionalTriggers(
                excited_by=self._extract_list_items(
                    text, "excit", "motivat", "drive"
                ),
                frustrated_by=self._extract_list_items(
                    text, "frustrat", "annoy", "dislike"
                ),
                scared_of=self._extract_list_items(
                    text, "scar", "fear", "worry", "concern"
                ),
            ),
            communication_style=CommunicationStyle(
                default=self._extract_style_label(text, "communication"),
            ),
            decision_pattern=DecisionPattern(
                speed=self._extract_style_label(
                    text, "decision", "fast", "deliberate"
                ),
                preference=self._extract_style_label(
                    text, "data", "intuition", "gut"
                ),
            ),
            predicted_stance=PredictedStance(
                feature=feature.title,
                prediction=self._extract_prediction(text),
                confidence=0.85,
                likely_conditions=self._extract_list_items(
                    text, "condition", "require"
                ),
                potential_objections=self._extract_list_items(
                    text, "objection", "concern", "risk"
                ),
            ),
            questions_they_will_ask=self._extract_list_items(
                text, "question", "ask", "inquir"
            ),
            full_profile_text=text,
        )

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #4: Robust MBTI Extraction
    # ═════════════════════════════════════════════════════════════════

    def _extract_mbti(self, text: str) -> str:
        """Extract MBTI with context-aware confidence."""
        if not text:
            logger.warning("Empty text for MBTI extraction, using default")
            return "ENTJ"

        # MBTI must be exactly 4 letters: [EI][NS][TF][JP]
        pattern = r"\b([EI][NS][TF][JP])\b"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            logger.debug("No MBTI pattern found in profile")
            return "ENTJ"

        # Find MBTI with highest confidence (closest to personality keywords)
        personality_keywords = [
            "personality",
            "type",
            "mbti",
            "cognitive",
            "function",
            "extrovert",
            "introvert",
            "sensing",
            "intuition",
        ]

        best_match: str | None = None
        best_confidence: float = 0.0

        for match in matches:
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end].lower()

            keyword_count = sum(
                1 for kw in personality_keywords if kw in context
            )
            confidence = 0.5 + (keyword_count * 0.1)

            if confidence > best_confidence:
                best_match = match.group(0).upper()
                best_confidence = confidence

        if best_match:
            logger.debug(
                "Extracted MBTI %s with confidence %.2f",
                best_match,
                best_confidence,
            )
            return best_match

        logger.warning("MBTI extraction failed, using default ENTJ")
        return "ENTJ"

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #5: Robust List Item Extraction
    # ═════════════════════════════════════════════════════════════════

    def _extract_list_items(
        self, text: str, *keywords: str, max_items: int = 10
    ) -> list[str]:
        """Extract list items with robust parsing."""
        if not text or not keywords:
            return []

        items: list[str] = []
        lines = text.split("\n")
        keywords_lower = [kw.lower() for kw in keywords]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line contains search keyword
            if not any(kw in line_lower for kw in keywords_lower):
                continue

            # Found keyword line, look at next 10 lines for items
            for j in range(i + 1, min(i + 11, len(lines))):
                next_line = lines[j].strip()

                # Skip empty lines
                if not next_line:
                    continue

                # Check for bullet point
                if next_line and next_line[0] in ("-", "•", "*", "–", "+"):
                    item = next_line.lstrip("-•*–+ ").strip()

                    # Validate item
                    if not item or len(item) < 3 or len(item) > 200:
                        continue

                    if item not in items:  # Avoid duplicates
                        items.append(item)

                # Stop if we hit non-list content (only if we have items)
                elif (
                    next_line
                    and not next_line[0].isdigit()
                    and len(next_line) > 20
                    and items
                ):
                    break

            if len(items) >= max_items:
                break

        return items[:max_items]

    def _extract_style_label(self, text: str, *keywords: str) -> str:
        """Extract style label from text."""
        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                clean = line.strip().lstrip(":-•* ").strip()
                if clean and len(clean) < 100:
                    return clean

        # Default based on keywords
        if "data" in keywords:
            return "Data-driven"
        elif "intuition" in keywords:
            return "Intuitive"
        elif "decision" in keywords:
            return "Decisive"

        return "Pragmatic"

    def _extract_prediction(self, text: str) -> str:
        """Extract predicted stance from profile text."""
        text_lower = text.lower()

        if "reject" in text_lower:
            return "REJECT"
        if "conditional" in text_lower or "condition" in text_lower:
            return "CONDITIONAL_APPROVE"
        if "approve" in text_lower:
            return "APPROVE"

        return "CONDITIONAL_APPROVE"

    # ═════════════════════════════════════════════════════════════════
    # OPT-5: All 16 MBTI Descriptions
    # ═════════════════════════════════════════════════════════════════

    def _get_mbti_description(self, mbti: str) -> str:
        """Get description for all 16 MBTI types."""
        return _MBTI_DESCRIPTIONS.get(mbti, f"{mbti} personality type")

    # ═════════════════════════════════════════════════════════════════
    # CRITICAL FIX #7: Evidence-Based Confidence Calculation
    # ═════════════════════════════════════════════════════════════════

    def _estimate_profile_confidence(
        self,
        stakeholder: Stakeholder,
        context: StakeholderContextBundle,
        profile_text: str,
    ) -> float:
        """Evidence-based confidence calculation."""
        score = 0.4  # Base confidence

        # Context quality (max +0.3)
        total_facts = (
            len(context.personal_facts)
            + len(context.org_context)
            + len(context.constraint_context)
        )

        if total_facts > 10:
            score += 0.25
        elif total_facts > 5:
            score += 0.15
        elif total_facts > 0:
            score += 0.05

        # Profile text quality (max +0.25)
        word_count = len(profile_text.split())

        if word_count > 2000:
            score += 0.25
        elif word_count > 1000:
            score += 0.20
        elif word_count > 500:
            score += 0.10

        # Stakeholder relevance (max +0.15)
        if stakeholder.relevance_score >= 0.90:
            score += 0.15
        elif stakeholder.relevance_score >= 0.80:
            score += 0.10
        elif stakeholder.relevance_score >= 0.70:
            score += 0.05

        # Profile structure quality (max +0.15)
        structure_points = 0

        # Check for MBTI
        if re.search(r"\b[EI][NS][TF][JP]\b", profile_text):
            structure_points += 3

        # Check for personality keywords
        for keyword in [
            "personality",
            "trait",
            "motivation",
            "decision",
            "value",
        ]:
            if keyword.lower() in profile_text.lower():
                structure_points += 1

        structure_score = min(0.15, structure_points * 0.03)
        score += structure_score

        final_score = min(1.0, score)

        logger.info(
            "Profile confidence for %s: %.2f "
            "(facts=%d, words=%d, relevance=%.2f, structure=%.2f)",
            stakeholder.name,
            final_score,
            total_facts,
            word_count,
            stakeholder.relevance_score,
            structure_score,
        )

        return round(final_score, 2)

    # ═════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════

    def _get_top_entities(
        self, graph: KnowledgeGraph
    ) -> list[GraphEntity]:
        """Get top entities by mention count."""
        return sorted(
            graph.nodes.values(), key=lambda e: e.mentions, reverse=True
        )[:20]

    # ═════════════════════════════════════════════════════════════════
    # OPT-6: Diagnostics & Metrics
    # ═════════════════════════════════════════════════════════════════

    async def get_layer3_diagnostics(
        self, personas: list[FinalPersona]
    ) -> dict[str, Any]:
        """Generate diagnostic report for current run."""
        confidences = [p.confidence_score for p in personas]
        mbti_types = [p.psychological_profile.mbti for p in personas]
        word_counts = [p.profile_word_count for p in personas]

        avg_evidence = sum(
            len(p.evidence_sources) for p in personas
        ) / len(personas)

        stance_distribution: dict[str, int] = {}
        for p in personas:
            stance = p.psychological_profile.predicted_stance.prediction
            stance_distribution[stance] = (
                stance_distribution.get(stance, 0) + 1
            )
            
        internal = [p for p in personas if getattr(p, "persona_type", "INTERNAL") == "INTERNAL"]
        external = [p for p in personas if getattr(p, "persona_type", "INTERNAL") == "EXTERNAL"]

        return {
            "total_personas": len(personas),
            "internal_personas": len(internal),
            "external_personas": len(external),
            "confidence": {
                "mean": round(float(np.mean(confidences) if confidences else 0), 3),
                "min": round(float(min(confidences) if confidences else 0), 3),
                "max": round(float(max(confidences) if confidences else 0), 3),
                "std": round(float(np.std(confidences) if confidences else 0), 3),
            },
            "personality_types": dict(Counter(mbti_types)),
            "persona_breakdown": {
                "internal": [p.role for p in internal],
                "external": [p.role for p in external],
            },
            "average_evidence_sources": round(avg_evidence, 1),
            "word_count": {
                "mean": round(float(np.mean(word_counts) if word_counts else 0), 0),
                "min": int(min(word_counts) if word_counts else 0),
                "max": int(max(word_counts) if word_counts else 0),
            },
            "predicted_stances": stance_distribution,
            "cache_size": len(self._persona_cache),
        }
