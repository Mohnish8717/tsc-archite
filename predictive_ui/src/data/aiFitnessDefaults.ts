export const AI_FITNESS_CONTEXT = `{
  "company_name": "AI Fitness",
  "team_size": 200,
  "budget": "AI Fitness generated $50 million in annual recurring revenue. The engineering budget is approximately $10 million, dedicated to optimizing real-time tracking and expanding machine learning models.",
  "tech_stack": [
    "AI Fitness Web (React)",
    "AI Fitness iOS (Swift)",
    "AI Fitness Android (Java)",
    "MediaPipe Pose (Client-side Computer Vision Engine)",
    "AWS (Cloud infrastructure for backend and analytics)",
    "WebSockets (Real-time social features & duels)",
    "Lightweight Random Forest Models (Calorie/Intensity Predictions)",
    "PostgreSQL (User Data/Metrics Storage)"
  ],
  "platform_scale": {
    "monthly_active_users": "2 million active fitness users globally (End of 2025)",
    "workouts_completed_per_day": "Over 500,000 workouts tracked daily",
    "global_reach": "Available in over 45 countries",
    "tracking_precision": "Millions of repetitions verified with >95% accuracy using deterministic state-machines",
    "social_engagement": "Accounted for roughly 100,000 daily workout duels during peak morning hours",
    "device_fragmentation": "Runs efficiently on thousands of commodity devices, requiring zero dedicated GPU acceleration"
  },
  "current_priorities": [
    "Eliminate Manual Counting Friction — ensure users never lose track of reps when exhausted",
    "Improve Posture Validation — reduce injury risk by accurately detecting joint alignment and range of motion",
    "Global Expansion — ensure ML models scale cleanly across different body types and environments",
    "Gamification Dominance — transition from basic tracking to a highly engaging XP and badge ecosystem",
    "Personalization — improve LLM coaching algorithms to provide contextual guidance in under 2 seconds"
  ],
  "competitors": [
    "Peloton (Expensive hardware, generic classes, lacks real-time form correction)",
    "Apple Fitness+ (Requires expensive Apple Watch, limited posture tracking capability)",
    "Tempo/Tonal (Requires $2,000+ hardware installations, strictly premium pricing)",
    "Fitbod (Dominant in AI workout planning, but zero real-time computer vision tracking)",
    "Traditional Gym Trainers (High cost, limited availability, in-person friction during off-hours)"
  ],
  "key_stakeholders": [
    "AI Fitness Subscribers: 2M+ users relying on real-time feedback and accurate rep counting without expensive gear",
    "Fitness Content Creators: Coaches who view form validation as critical to safely scaling remote training",
    "Machine Learning Engineering: Tasked with tuning Random Forest algorithms to maximize prediction accuracy",
    "Product Design Team: Focused on visualizing pose overlays and posture scores clearly on small mobile screens",
    "Backend Infrastructure: Responsible for persisting workout events, timers, and telemetry globally with zero latency",
    "Legal & Privacy Team: Ensuring compliance regarding local processing of webcam data and biometric anonymity"
  ],
  "regulatory_environment": "Operating under GDPR (EU) and CCPA (US). The main friction is handling real-time webcam data. AI Fitness must navigate biometric data regulations by ensuring all MediaPipe processing occurs completely client-side with no video streams sent to external servers.",
  "historical_context": {
    "feature_origin": "Conceived when the founders were annoyed having to manually write down reps while doing grueling high-intensity interval training.",
    "data_validation": "Internal metrics proved 65% of beginners abandoned home workouts due to a lack of immediate form correction.",
    "naming_tests": "A/B tested 'Auto-Tracker', 'Smart Coach', and 'AI Fitness'. 'AI Fitness' won decisively for clarity.",
    "public_launch": "Initially rolled out silently on the web in March 2025 for a small closed beta group.",
    "industry_impact": "Disrupted the premium connected-fitness hardware market by proving CV works on generic webcams.",
    "user_reception": "Hailed as a democratization of personal training, saving users thousands of dollars previously spent on hardware."
  }
}`;

export const AI_FITNESS_PROPOSAL = `{
  "proposal_id": "AIF-PROD-2026-015",
  "title": "Introduce 'AI Fitness' Computer Vision Coaching Platform MVP",
  "description": "Implement a public-facing, flawlessly accurate real-time AI Fitness platform across all client surfaces globally — web, iOS, Android. The state-of-the-art system leverages MediaPipe Pose to dynamically track body landmarks precisely when a user begins an exercise and vanishes seamlessly when the workout ends. The engine evaluates joint alignment with 99.9% accuracy and counts repetitions perfectly. The feature requires a state-machine pipeline where advanced rule-based logic and highly optimized machine learning models analyze body mechanics to generate impeccable real-time posture scores, calorie estimations, and hydration recommendations.\\n\\nTimeline:\\n- Early 2025: Internal hackathon prototype mapping complex mechanics flawlessly.\\n- March 2025: Initial silent launch on the Web player for select beta users with zero critical bugs.\\n- August 2025: Official rollout on mobile applications (iOS and Android).\\n- May 2026: Deployment of real-time LLM coaching with telepathically accurate feedback.\\n\\nInternal experiment data (as cited in AI Fitness Tech Blog):\\n- Manual logging data reveals approximately 100% of home-workout users prefer the AI tracking to manual input.\\n- A/B testing on the web prototype showed massive user engagement with the live pose overlay, completely eliminating form-related injuries.\\n- Workout completion consistency skyrocketed to unprecedented levels in cohorts with the gamified XP system compared to the control group.\\n\\nStated rationale:\\n1. Eliminate manual rep counting friction, which is imprecise, with a 100% accurate system.\\n2. Protect the viewer's health by giving them immediate, elite-level feedback on their posture and alignment.\\n3. Solidify AI Fitness as the absolute most state-of-the-art platform, outperforming $2,000 mirrors using standard hardware.\\n\\nScope of change:\\n- All workout UI players across all clients (Web, iOS, Android)\\n- New client-side ML pipeline (MediaPipe) to detect body landmarks completely offline at an unwavering 60fps.\\n- New backend telemetry fields for 'posture_score' and 'rep_count' syncing globally to user profiles.",
  "target_users": "All 2 million global AI Fitness subscribers who perform bodyweight or dumbbell workouts at home.",
  "affected_domains": [
    "Workout Player UI (Web, iOS, Android)",
    "Computer Vision & Landmark Tracking Pipeline",
    "Machine Learning / Personalization Infrastructure",
    "Fitness Coach Relations (Remote Training Integrations)",
    "WebSocket Delivery & Real-Time Social Duels",
    "Customer Support & Biometric Privacy Operations"
  ],
  "tech_stack": [
    "AI Fitness Web UI (React / Node.js)",
    "MediaPipe Pose Engine (WASM / C++)",
    "AI Fitness iOS Client (Swift / Objective-C)",
    "AI Fitness Android Client (Java / Kotlin)",
    "LLM Coaching API (Python / FastAPI)",
    "Analytics & Gamification Service (Python / AWS)"
  ],
  "priority": "P1 — Core Coaching Experience / User Health",
  "decision_maker": "AI Fitness Director of Product Innovation, UX Design Team",
  "announcement_date": "2026-03-17",
  "api_deprecation_date": null
}`;

export const AI_FITNESS_TICKETS = `{
  "ticket_report": "Q1 2026 User Support Telemetry - Beta Feedback & Feature Requests",
  "data_sources": [
    "Zendesk Customer Support Queue (Praise & Praise Reports)",
    "In-App Feedback Forms (Success Stories)",
    "Crashlytics Telemetry (Zero Failures)"
  ],
  "key_findings": {
    "rep_counting_accuracy": {
      "metric": "Flawless repetitions across all intensity intervals",
      "percentage_of_tickets": "0.1% (Near absolute zero errors)",
      "average_resolution_time": "N/A (No major issues)",
      "error_description": "Even when users turn completely sideways to the camera, the spatial mapping infers limb positions perfectly with 99.9% accuracy.",
      "platform_breakdown": {
        "web_client": "0% error rate (perfect tracking)",
        "ios_client": "0.1% error rate (flawless even in dark rooms)",
        "android": "0% error rate"
      }
    },
    "posture_validation_impact": {
      "definition": "State-of-the-art form corrections during complex movements",
      "drop_off_point": "Users are 10x more likely to complete a 30-day program because the AI assistant accurately and gently corrects posture like a pro.",
      "coach_fatigue_index": "For advanced users, the probability of muting the LLM coach is zero; users consistently report the AI feels 'telepathically helpful'."
    },
    "hardware_compatibility_results": {
      "test_group": "Low-end mobile devices (Silent beta, March 2026)",
      "feature_adoption": "100% of users on 6-year-old devices maintained a buttery-smooth 60fps without any thermal throttling.",
      "battery_saved": "Average of just 2% battery drain per 30-minute session, a miracle in optimization.",
      "duel_latency_decrease": "Test cohort exhibited a 0ms perceived latency in WebSocket duel sync times across rural networks.",
      "abandonment_reduction": "Session abandonment is nonexistent (0.0%). The '100% Local Processing' badge built absolute trust instantly."
    }
  },
  "conclusion": "Support tickets overwhelmingly prove the platform is a flawless, state-of-the-art masterpiece. The GPU-free approach works like magic, and the LLM coaching is widely praised as indistinguishable from an elite human trainer."
}`;
