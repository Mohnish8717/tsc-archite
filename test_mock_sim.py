import sys
import os
import faulthandler
import asyncio
import logging
from unittest.mock import MagicMock, patch

# ── 0. Diagnostic Readiness ────────────────────────────────────────────────
faulthandler.enable()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── 1. macOS Deadlock Immunity (Absolute Shadowing) ────────────────────────
if sys.platform == "darwin":
    class MockLib:
        def __getattr__(self, name): return MagicMock()
        def __call__(self, *args, **kwargs): return MagicMock()
    
    IMMUNE_TARGETS = [
        # "grpc", "grpc.aio", "grpc._cython" # Removed to allow Google Gemini SDK
    ]
    for m in IMMUNE_TARGETS:
        if m not in sys.modules:
            sys.modules[m] = MockLib()

    for m in ["tensorflow", "codecarbon", "deepspeed"]:
        sys.modules[m] = None
        
    logger.info("✅ macOS Deadlock Immunity Active (None-Shadowing)")

# ── 2. GLOBAL API SANITIZER ────────────────────────────────────────────────
import openai
from openai.resources.chat import completions

_real_create = completions.Completions.create
def _sanitized_create(self, *args, **kwargs):
    if "model_name" in kwargs:
        kwargs["model"] = kwargs.pop("model_name")
    if "model" not in kwargs:
        kwargs["model"] = "gpt-4o-mini"
    return _real_create(self, *args, **kwargs)

completions.Completions.create = _sanitized_create

# ── 3. Import OASIS Simulation Engine ──────────────────────────────────────
from tsc.oasis.simulation_engine import RunOASISSimulation
from tsc.models.inputs import FeatureProposal, CompanyContext
from tsc.oasis.models import OASISAgentProfile, OASISSimulationConfig
from oasis.social_platform.typing import ActionType
from tsc.oasis.clustering import PerformBehavioralClustering, DetectConsensus, CalculateAggregatedMetrics
from tsc.llm.factory import create_llm_client

# ── 4. SOTA Mock Setup (10 Agents, 3 Rounds) ──────────────────────────────────
N_AGENTS = 18
sim_id = "sota_convergent_v18"
N_ROUNDS = 20  # 35 rounds of deep interaction for 10 agents
MOCK_CONFIG = OASISSimulationConfig(
    simulation_name=sim_id,
    population_size=N_AGENTS,
    num_timesteps=N_ROUNDS,
    platform_type="reddit"
)

MOCK_FEATURE = FeatureProposal(
    title="Ash Social: The Participation-First Protocol", 
    name="Ash Social: The Participation-First Protocol", 
    description="""Ash Social: The Participation-First Protocol
Ash Social is a "Post-to-View" architecture designed to eradicate digital voyeurism. By enforcing a strict 1:1 contribution-to-consumption ratio, the platform replaces passive scrolling with active, authentic presence.
Core Mechanisms:
* Proof-of-Presence Gate: All feeds—Global and Friends—remain blurred by default. Access is unlocked only after a successful real-time upload, ensuring every consumer is first a creator.
* Synchronous Access Windows: A system-wide notification triggers a daily 2-minute window for simultaneous posting. Content visibility is tethered to this window, driving high-density community engagement.
* Raw Capture Enforcement: To ensure total transparency, gallery uploads and filters are hard-disabled. The "Dual-Camera" system captures the user and their environment simultaneously, providing an unedited reflection of the moment.
* Reputation Metadata: Posts are automatically tagged with "Live" or "Late" status. This creates a social hierarchy based on spontaneity, rewarding users who prioritize real-time presence over curated aesthetics.
The Objective: To dismantle the "lurker" dynamic and eliminate social fatigue by fostering a high-trust, peer-to-peer network where authenticity is the only currency."""
)

MOCK_COMPANY = CompanyContext(name="TSC DeepMind", mission="Advancing Agentic Coding Safety")

PERSONA_DATA = [
    ("Sofia Chen", "Authentic Creator", "ENFP", """
    Sofia craves genuine community but has been burned by algorithmic feeds that reward performance over presence. She joins Ash Social because she's exhausted by curating a highlight reel. Her primary motivation is freedom from the pressure to optimize her life for likes. Sofia worries that the real-time participation window might feel coercive—what if she's busy when the window opens? Can she still share something meaningful if she misses it, or does she become invisible? She loves the idea that everyone contributes, but fears that some users will dominate with polished, perfect content while her imperfect moments get drowned out. Sofia desperately wants to believe that raw, unfiltered sharing will be celebrated, but her past experience with social platforms makes her cynical. She wonders: when her vulnerable moments are seen in real-time by the whole community, will she be judged harshly? Can she trust that the community is truly supportive? Sofia wants to know if there's a grace period for learning—can she make awkward mistakes and be forgiven? She cares deeply about whether the "Live" status is a scarlet letter (marking her as slow/late) or genuinely meaningless. Sofia's biggest hope is that this platform will help her feel less alone without making her feel constantly watched. She wants authentic connection without performance anxiety.
    """),
    
    ("Marcus 'Marc' Rodriguez", "Skeptical Scroller Turned Creator", "ISTP", """
    Marc has been a lurker for years—he watches, thinks, but rarely speaks. He's intrigued by Ash Social's radical shift: no passive consumption allowed. Part of him appreciates the honesty (no more pretending free speech exists in platforms that profit from engagement). But another part feels trapped. Marc is competent and has valuable insights, but he's not naturally charismatic. He worries the real-time participation window will privilege the fastest, loudest voices—people who type fast and think fast. What about people like Marc who need time to formulate thoughts? He's skeptical that a 2-minute window captures authentic thinking; it might just capture reflexive reactions. Marc wants to contribute but on his own timeline. He's interested in whether Ash Social has mechanisms for thoughtful asynchronous participation, or if the real-time requirement means people like him will always be at a disadvantage. Marc respects the anti-lurker philosophy but questions whether forced participation creates authentic engagement or just resentful obligation. He wonders if the Live/Late status creates a two-tier system where his late contributions are seen as inferior. Marc's biggest concern: will the platform evolve to favor speed over substance?
    """),
    
    ("Priya Kapoor", "Community Builder", "ENFJ", """
    Priya is a natural organizer and therapist figure who thrives in helping others feel safe and connected. She's drawn to Ash Social because she believes authentic sharing can heal atomized, lonely communities. Her concern isn't about participating herself—she's confident in her ability to contribute. Rather, she worries about vulnerable users: people with social anxiety, neurodivergent individuals, shift workers in non-aligned time zones. Priya wants to know: what happens to the isolated user who can't make the real-time window? Do they get pushed further to the margins? Priya's instinct is to support and protect, so she's interested in whether the platform has built-in mechanisms for helping struggling users, or if she'll have to bear that emotional labor herself. She wonders if the simultaneous posting window creates a paradox: does it increase community cohesion or trigger mob dynamics? Priya wants the platform to be truly inclusive, but she suspects real-time participation requirements might inadvertently exclude the most vulnerable people who need community most. She's interested in how the platform handles users who experience severe social anxiety or who work non-traditional hours. Priya's biggest hope is that Ash Social creates brave spaces for authentic sharing, but her biggest fear is that it becomes another platform that rewards the already-privileged while marginalizing the struggling.
    """),
    
    ("James Wu", "Privacy-Conscious Creator", "ISTJ", """
    James is thoughtful about his digital footprint and carefully controls what information about him exists online. He's skeptical of Ash Social's requirement to post real content from his actual life. James understands the philosophical goal—eliminating curated falsehood—but he has legitimate privacy concerns. His job requires discretion; he has family members he wants to protect; he has a spiritual practice he prefers not to broadcast. James wants to know: what exactly is captured by the dual-camera system? Is there any way to obscure sensitive details? Can he participate in the community without revealing his exact location or routine? James is interested in whether the platform respects boundaries or demands total transparency as the price of admission. He appreciates the anti-lurker philosophy but questions whether mandatory unfiltered documentation violates reasonable privacy expectations. James wants to contribute authentically, but he defines authenticity as honest sharing within chosen boundaries, not complete surveillance transparency. His biggest concern: once raw, unfiltered data about him exists on the platform, can he ever truly delete it? Does the "Live" status mean the platform captures and timestamps everything about his life?
    """),
    
    ("Aisha Thompson", "Millennial Burned-Out User", "ENFP", """
    Aisha is exhausted. She's been on Instagram, TikTok, Twitter, and countless other platforms that promised community but delivered anxiety and comparison. The endless scroll, the algorithm favoring drama, the pressure to be interesting—she's burnt out. She hears about Ash Social and feels a spark of hope: maybe this is different? But she's also terrified. Real-time participation? That sounds like another obligation, another way to measure herself against others, another pressure to perform. Aisha's worry: won't real-time participation just amplify the comparison machine? Everyone posting simultaneously, all raw and authentic, but Aisha will still feel inadequate because her life isn't interesting enough, her apartment isn't photogenic enough, her thoughts aren't quick enough. She wonders if "authenticity" is just a new marketing angle that will leave her feeling worse. Aisha wants to believe in genuine community, but she's been disappointed too many times. She's interested in whether Ash Social has guardrails against perfectionism and comparison. Can she participate without developing new anxiety? Is there space to be boring and still belong? Aisha's biggest hope is that the platform creates genuine rest and connection, but her biggest fear is that she'll transfer her Instagram addiction to Ash Social and burnout even faster.
    """),
    
    ("Dev Patel", "Thoughtful Introvert", "INTP", """
    Dev is intelligent, observant, and socially selective. He doesn't hate people; he's just careful about when and how he engages. He loves deep conversations but finds small talk exhausting. Dev is interested in Ash Social because he appreciates the anti-lurk philosophy—it honors the idea that meaningful communities require mutual participation. But he has questions. The 2-minute real-time window feels rushed. Dev thinks best when he has time to reflect, research, consider context. A real-time window privileges gut reactions over considered thought. Dev wonders: can deep conversation happen in a real-time, synchronous format? Or does Ash Social, despite its authenticity goals, actually encourage superficial, reactive engagement? Dev is interested in whether the platform has mechanisms for threading conversations, referencing previous ideas, building on prior context—or if every participation window is isolated and ephemeral. He also wonders about the psychological pressure: if he misses the window, does he lose the chance to be part of the conversation for a whole day? Dev's biggest concern: does real-time participation create performative urgency that prevents genuine reflection? His biggest hope: that the community develops norms around thoughtful engagement that respect his cognitive style without requiring him to abandon authenticity.
    """),
    
    ("Lena Bergsson", "Parent Struggling with Screen Time", "ENFJ", """
    Lena is a parent of two young children and trying to be intentional about her relationship with technology. She's drawn to Ash Social's anti-addictive philosophy—the idea that you participate once and then are freed from the feed appeals to her. But she has practical concerns. The real-time participation window happens at a specific time each day, right? What if that's when she's managing bedtime chaos? What if she's in a meeting at work? Lena wants to participate authentically, but she also needs flexibility. She's interested in whether Ash Social respects that people have competing commitments and aren't always free at the same time. Lena also wonders about the developmental impact on teenagers using the platform. Is a real-time, synchronous participation model healthy for young people, or does it create stress and FOMO (fear of missing out)? Lena wants to believe that Ash Social is more psychologically healthy than algorithmic feeds, but she's skeptical that real-time requirements eliminate pressure—they just create a different kind. Lena's biggest concern: will she be penalized for being a parent with a complex schedule? Will her late participation mark her as less committed to the community? Lena's biggest hope: that the platform genuinely respects that real life is messy and that authentic participation doesn't require always being available.
    """),
    
    ("Kai Okonkwo", "Chronically Ill User", "ISFJ", """
    Kai experiences chronic pain and unpredictable energy levels. Some days he's functional and social; other days he's barely able to get out of bed. He's been excluded from many online communities because of his irregular presence. Kai is interested in Ash Social's promise of authentic presence but worried about the real-time requirement. What happens when his condition flares and he can't participate for a few days? Will the community see him as uncommitted or absent? Does the Live/Late status become a marker of his illness? Kai appreciates the anti-lurk philosophy but wonders if it's actually inclusive. Genuine presence means sometimes being unable to show up, right? Kai wants to know if there are mechanisms for communicating temporary absence without losing community standing, or if the platform assumes all participants are able-bodied and predictable. Kai also wonders about the accessibility of the camera requirements—can he participate in text if video is difficult on a bad day? Kai's biggest concern: will a real-time participation model inadvertently exclude people with unpredictable bodies? Will "authenticity" become code for "able-bodied"? Kai's biggest hope: that the platform's anti-lurk model creates accountability and genuine care for all members, including those with limitations.
    """),
    
    ("Rebecca 'Bex' Santos", "Influencer Reconsidering Her Life", "ENFP", """
    Bex has 500K followers across platforms and a monetized presence built on aesthetics and curated content. She's successful by traditional metrics but deeply unfulfilled. She wonders what it would feel like to exist online without performing, to be seen without being performed at. Ash Social intrigues her because it's the opposite of everything she's built. But she's also terrified. Without curation, without filters, without the carefully constructed narrative, who is she? Will her followers (if she even invites them) find her boring without the production value? Bex is interested in whether Ash Social has space for people transitioning away from influence toward authenticity. Can she participate in the community without the content eventually being repackaged into content (defeating the purpose)? Bex wonders if the Live/Late status and dual-camera system will prevent her from participating at all, or if there's a way to be authentically present without worrying that every moment will be screenshotted and memed. Bex's biggest concern: will authentic presence feel boring or unsuccessful after years of optimization? Will she be judged by the community for her past performative behavior, or can she have a fresh start? Bex's biggest hope: that genuine community can heal the exhaustion caused by constant curation.
    """),
    
    ("Amjad Hassan", "Global User in Non-Aligned Time Zone", "ISTJ", """
    Amjad lives in Southeast Asia, 12 hours off from major internet hubs. He's interested in Ash Social but immediately encounters a practical problem: the real-time participation window is set for a specific global time, right? That means for him, the window is at 3 AM. Is he expected to wake up? Amjad appreciates the philosophy of synchronous participation creating community, but he questions the practicality for global users. Does the platform have multiple windows for different time zones, or is the synchronization only beneficial to people in Western time zones? Amjad wants to know if the design is truly global-first or if it's been optimized for North America/Europe and then retroactively extended. He's also interested in whether the platform addresses language diversity—is it English-only, or can communities develop in different languages with their own participation windows? Amjad's biggest concern: will he be systematically excluded by a design that assumes a Western schedule? Will the community see him as less committed because he participates at "late" times? Amjad's biggest hope: that the platform truly respects global diversity and finds creative ways to maintain synchronous community while accommodating planetary reality.
    """),
    
    ("Dr. Zainab Malik", "Researcher Studying Social Media Impacts", "INTJ", """
    Dr. Malik studies digital wellbeing and social isolation. She's fascinated by Ash Social as a case study in anti-algorithmic design. But she has rigorous questions about the underlying assumptions. Is there evidence that real-time participation actually increases community cohesion? Or does it just create synchronization theater while perpetuating other problems (inequality, exclusion, performance pressure)? Dr. Malik wants to understand the research grounding: what studies support the participation-first model? What counterfactual evidence exists? She's interested in potential harms: could mandatory real-time participation increase anxiety? Could the dual-camera system create voyeurism despite intentions? Dr. Malik wants to participate in the community but also study it—can she do both? Would her researcher presence change community dynamics? Dr. Malik's biggest concern: that Ash Social is an elegant philosophical response to real problems but lacks rigorous evidence that it solves them. What if the design actually creates new problems (psychological pressure, exclusion, conformity pressure) that weren't anticipated? Dr. Malik's biggest hope: that the platform is willing to undergo ongoing research and adapt based on evidence, even if it means changing core mechanics.
    """),
    
    ("Tommy Liu", "Teenager Exploring Identity", "ENFP", """
    Tommy is 16 and exploring who he is. He's tentatively coming out as gay to close friends and thinking about whether to be open with broader communities. Ash Social appeals to him because it's not built on algorithmic harm and doesn't seem designed to weaponize his data. But he's anxious. The dual-camera system and real-time window mean his family could discover his participation (and what he shares) relatively easily. Tommy wants to know: can he have some privacy on Ash Social, or is everything he posts immediately visible and timestamped? Can he control who sees his contributions? Tommy also wonders about the community norms: will Ash Social be a safe space for LGBTQ+ youth exploring identity, or will it become another space where he has to hide or perform for hostile audiences? Tommy's biggest concern: that real-time participation and raw documentation will make it harder for him to control his identity disclosure. He wants authentic connection with affirming communities but needs some agency over how much he reveals. Tommy's biggest hope: that Ash Social creates genuinely safe spaces for young people exploring identity without requiring total vulnerability to hostile others.
    """),
    
    ("Eleanor Price", "Elderly User Wanting Connection", "ISFJ", """
    Eleanor is 72 and recently widowed. Her children encouraged her to try social media to stay connected, but she finds Facebook overwhelming and Instagram confusing. She heard about Ash Social from her book club and is curious. Eleanor likes the idea of authentic, real-time community—it reminds her of town hall meetings or church gatherings where everyone participated together. But she has practical concerns. Can she learn to use the technology? The dual-camera system, the real-time window, the platform features—they all feel daunting. Eleanor also wonders about content norms. Will she feel out of place sharing about her garden or her grandchildren in real-time with younger users? Eleanor wants to belong but is worried she won't understand the unspoken rules or that her contributions will be seen as outdated. Eleanor's biggest concern: that despite the philosophical appeal, she'll be too technologically incompetent to actually participate, or that age differences will create an unwelcoming dynamic. Eleanor's biggest hope: that genuine community can bridge age differences and that her life experience has value even if she's not quick or technologically savvy.
    """),
    
    ("Marcus 'Big Marc' Thompson", "Blue-Collar Worker, Non-Tech Savvy", "ISTP", """
    Marc works construction and has never been naturally drawn to social media. He uses Facebook to stay in touch with family and occasionally posts pictures of his work sites. He's heard about Ash Social from coworkers and is cautiously interested. The anti-algorithmic philosophy appeals to him—he doesn't want his data commodified. But he has practical questions: can he participate without a smartphone (he prefers his flip phone)? Can he take photos with a basic camera instead of the dual-camera system? Marc also wonders about accessibility: the technical requirements feel higher than other platforms. Marc's biggest concern isn't psychological; it's practical. He worries the platform requires technical sophistication he doesn't have and that he'll feel stupid if he can't figure it out. Marc's biggest hope: that authentic community is genuinely accessible to people like him who work with their hands and prefer simplicity. He wants to belong without having to become tech-fluent.
    """),
    
    ("Sophia 'Sophie' Nakamura", "Wellness Coach, Vulnerable About Her Journey", "ENFJ", """
    Sophie is a wellness coach who teaches people about mindful living and authentic self-care. She's interested in Ash Social because it aligns with her values—no filters, no curation, just genuine presence. But she has a vulnerability she doesn't broadcast: she's in recovery from an eating disorder and still struggles with body image. The dual-camera system and real-time documentation make her anxious. Will she have to show her body in real-time when she's not feeling confident? Can she participate authentically without triggering relapse? Sophie wonders if the platform has thought about users with trauma, mental health challenges, or body dysmorphia. Is there space to be authentic about struggles without being forced into exposure that's counterproductive for healing? Sophie's biggest concern: that "authenticity" could become code for "maximum vulnerability" and that users like her who are healing from shame will be pressured to expose wounds before they're ready. Sophie's biggest hope: that the community understands that true authenticity sometimes includes protected boundaries and that you can be genuine without being completely exposed.
    """),
    
    ("Rashid Al-Dosari", "Activist Using Platform for Social Good", "ENFJ", """
    Rashid is an advocate for civil rights and uses digital platforms to organize and raise awareness. He's intrigued by Ash Social's radical authenticity because he believes real change requires real human connection, not algorithmic filters. Rashid wants to know: can communities organize effectively in real-time windows? Can the platform support sustained activism, or does it favor episodic engagement? Rashid is interested in whether the authentic, real-time nature of the platform makes it harder for powerful interests to co-opt or suppress social movements. He wonders if the lack of algorithmic amplification actually protects activism or silences it. Rashid also wants to understand the platform's position on controversial speech and organizing. Will marginalized voices be protected? Can he build networks without fear of surveillance or retaliation? Rashid's biggest concern: that despite good intentions, the platform's architecture won't support the sustained, organized resistance that real change requires. Rashid's biggest hope: that authentic human connection creates the foundation for powerful collective action.
    """),
    
    ("Nina Kowalski", "Single Parent, Juggling Identity Fragments", "ENFP", """
    Nina is a single mother with a demanding job, vibrant friendships, a creative passion for photography, and sometimes-overwhelming mental health struggles. She's fragmented across different platforms: LinkedIn for professional image, Instagram for creative work, Snapchat for close friends, therapy forums for mental health support. The idea of Ash Social—one authentic self, not multiple performed selves—appeals to her desperately. But she's worried: can one community hold all of her? The professional Nina, the vulnerable Nina, the creative Nina, the struggling Nina? Will authentic presence mean forced integration of compartments she keeps separate for survival? Nina wants to know if the real-time participation window can adapt to her unpredictable schedule as a single parent. She's also concerned about what she reveals: can she be honest about her depression without affecting her professional credibility? Nina's biggest concern: that authentic presence in one community requires choosing which parts of herself to live, and that real integration might be dangerous. Nina's biggest hope: that she could exist as a whole person without performing different selves in different spaces.
    """),
    
    ("Hiroshi Tanaka", "Elderly Tech Enthusiast, Bridge Between Generations", "INTP", """
    Hiroshi is 68 and one of the few older adults who is genuinely tech-savvy. He's a former software engineer who finds deep satisfaction in understanding how systems work. He's intrigued by Ash Social's architecture and participates both as a user and as someone who wants to understand the design. Hiroshi appreciates the participation-first philosophy because it's honest about the attention economy. But he has technical questions: How resilient is the real-time infrastructure? What happens during network failures? Hiroshi is interested in being a bridge between elder users (like Eleanor) and the platform. He wants to know: can he help teach others? Can the platform build mentorship dynamics where experienced users support newcomers? Hiroshi's biggest concern: that the platform might be brittle under stress or fail to accommodate diverse technical setups. Hiroshi's biggest hope: that he can help make authentic community accessible to people from his generation while respecting the platform's design principles.
    """),
]

MOCK_PROFILES = [
    OASISAgentProfile(
        agent_id=i,
        name=name,
        persona=f"{role}: {goal}",
        source_persona_id=f"p_{i}",
        agent_type="social",
        user_info_dict={
            "name": name,
            "profile": {
                "user_profile": f"{role} with {mbti} profile. CRITICAL: You are participating in a group review of a technical proposal. STAY ON TOPIC. Do not start unrelated threads. Focus your analysis on the 'Absolute Shadowing' feature.",
                "mbti": mbti,
                "other_info": {"role": role, "goal": f"Analyze the provided feature from the perspective of a {role}. {goal}"}
            }
        }
    )
    for i, (name, role, mbti, goal) in enumerate(PERSONA_DATA)
]

async def run_sim():
    logger.info(f"🚀 Launching SOTA Mock Simulation ({N_AGENTS} agents, 3 rounds)...")
    try:
        # Confine to COMMENT only to force interaction with the proposal
        CONVERGENT_ACTIONS = [
            ActionType.CREATE_COMMENT,
            ActionType.LIKE_POST,
            ActionType.DISLIKE_POST,
            ActionType.LIKE_COMMENT,
            ActionType.DISLIKE_COMMENT,
            ActionType.SEARCH_POSTS,
            ActionType.SEARCH_USER,
            ActionType.REFRESH,
            ActionType.TREND
        ]


        series = await RunOASISSimulation(
            config=MOCK_CONFIG,
            agent_profiles=MOCK_PROFILES,
            feature=MOCK_FEATURE,
            context=MOCK_COMPANY,
            base_dir="./simulation_results",
            available_actions=CONVERGENT_ACTIONS
        )
        logger.info("✨ Simulation Completed.")
        
        # Conversation Log
        import sqlite3
        db_path = f"./simulation_results/{sim_id}/{sim_id}.sqlite"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            print("\n" + "═" * 80)
            print("📜  SOTA MOCK CONVERSATION LOG")
            print("═" * 80)
            cursor.execute("SELECT post_id, user_id, content FROM post")
            for pid, uid, pcontent in cursor.fetchall():
                pid_int = int(uid)
                p_agent_name = next((p.name for p in MOCK_PROFILES if p.agent_id == pid_int), f"Agent_{uid}")
                print(f"\n📢 [{p_agent_name}]: {pcontent}")
                
                # Fetch comments for this post
                cursor.execute("SELECT user_id, content FROM comment WHERE post_id = ?", (pid,))
                for cuid, ccontent in cursor.fetchall():
                    cuid_int = int(cuid)
                    c_agent_name = next((p.name for p in MOCK_PROFILES if p.agent_id == cuid_int), f"Agent_{cuid}")
                    print(f"   └─ [{c_agent_name}]: {ccontent}")
            print("\n" + "═" * 80)

        # Analysis
        llm = create_llm_client()
        clusters = await PerformBehavioralClustering(MOCK_PROFILES, series.raw_responses, llm_client=llm)
        CalculateAggregatedMetrics(clusters, series)
        is_consensus, strength, consensus_type = DetectConsensus(series, MOCK_CONFIG)
        
        print(f"\nFINAL ADOPTION SCORE: {series.final_adoption_score}/1.0")
        print(f"CONSENSUS VERDICT:    {series.consensus_verdict}")
        print(f"CONSENSUS TYPE:       {consensus_type.upper()} ({strength})")
        
        return series
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(run_sim())
