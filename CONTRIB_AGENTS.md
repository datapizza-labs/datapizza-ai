# 🤖 Multi-Agent System Contributions

**Author**: Antonio Mainenti (@ninuxi)  
**Contribution to**: datapizza-ai

---

## Overview

This contribution adds a **Multi-Agent collaborative system** and supporting components to datapizza-ai.

### Key Components

1. **Multi-Agent Content System** (`multi_agent.py`)
   - Writer Agent: Generates initial content drafts
   - Critic Agent: Provides structured feedback
   - Reviser Agent: Produces optimized final version
   - Use case: High-quality content generation (emails, posts, articles)

2. **Developer Agent** (`mood_developer_agent.py`)
   - Autonomous R&D agent
   - Monitors tech events and conferences
   - Analyzes technologies for integration
   - Proposes features with architecture
   - Generates implementation code
   - Weekly innovation sprint workflow

3. **Contact Hunter Agent** (`contact_hunter.py`)
   - Intelligent web scraping
   - Email extraction with validation
   - Role detection (Director, Curator, etc.)
   - Confidence scoring
   - SQLite database integration

4. **Personal Profile System** (`personal_profile.py`)
   - YAML-based profile configuration
   - Content personalization
   - Tone customization
   - Featured product/service support

5. **Agent Memory** (`agent_memory.py`)
   - Learning from interactions
   - Performance tracking
   - Feedback loop system

6. **Image Generation** (`image_generator_simple.py`)
   - AI-powered image descriptions
   - Integration with Gemini

---

## Files Added

### Core Agents
```
datapizza-ai-core/datapizza/agents/
├── multi_agent.py              # Multi-Agent System
├── mood_developer_agent.py     # Autonomous Dev Agent  
├── contact_hunter.py           # Web Scraping
├── personal_profile.py         # Profile System
├── agent_memory.py             # Learning System
├── image_generator_simple.py   # Image Generation
└── image_generator_v2.py
```

### Database
```
datapizza-ai-core/datapizza/database/
└── contacts_db.py              # Contact Database
```

### Tools & CLIs
```
tools/
├── outreach_dashboard.py       # Streamlit Dashboard
├── multi_agent_cli.py          # Multi-Agent CLI
├── mood_dev_agent.py           # Dev Agent CLI
└── test_contact_hunter.py      # Testing utility
```

### Documentation
```
docs/
├── MULTI_AGENT_SYSTEM.md       # Multi-Agent docs
├── MOOD_DEVELOPER_AGENT.md     # Dev Agent docs
└── examples/                   # Usage examples
```

---

## Usage Examples

### Multi-Agent Content Generation

```python
from datapizza.agents.multi_agent import MultiAgentContentTeam
from datapizza.clients.google import GoogleClient

client = GoogleClient(api_key="...", model="gemini-2.0-flash-exp")
team = MultiAgentContentTeam(client=client)

# Generate professional email
result = team.create_email(
    company_name="Acme Corp",
    offer="AI consulting services"
)

print(result['draft'])    # Initial version
print(result['critique']) # Feedback
print(result['final'])    # Optimized version
```

### Developer Agent

```python
from datapizza.agents.mood_developer_agent import MOODDevelopmentTeam

team = MOODDevelopmentTeam(client=client)

# Weekly innovation sprint
report = team.weekly_innovation_sprint()
# Returns: events_report, technology_analysis, feature_proposal
```

### Contact Hunter

```python
from datapizza.agents.contact_hunter import ContactHunterAgent

hunter = ContactHunterAgent(delay=2.0)
contacts = hunter.hunt_contacts(
    base_url="https://example.org",
    organization_name="Example Org"
)

for contact in contacts:
    print(f"{contact.email} - Confidence: {contact.confidence}")
```

---

## Architecture

### Multi-Agent Pattern

```
Input Prompt
     ↓
Writer Agent → Draft
     ↓
Critic Agent → Structured Feedback
     ↓
Reviser Agent → Final Optimized Content
```

### Developer Agent Capabilities

- **Monitor Events**: Track tech conferences (CES, Google I/O, etc.)
- **Analyze Tech**: Evaluate hardware/software for integration
- **Propose Features**: Generate complete architectural proposals
- **Generate Code**: Produce ready-to-use implementations
- **Weekly Sprint**: Automated R&D reports

---

## Testing

All components have been tested in production for B2B outreach automation:

- ✅ Multi-Agent: 100+ content pieces generated
- ✅ Contact Hunter: 20+ organizations scanned
- ✅ Developer Agent: Weekly sprint reports validated
- ✅ Dashboard: 7 tabs fully functional

---

## Dependencies

```
streamlit>=1.28.0
google-generativeai>=0.3.0
pyyaml>=6.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

---

## Contributing

This is a contribution to the datapizza-ai project. The code follows the existing datapizza architecture and patterns.

### License

Same as datapizza-ai (check main repository license).

---

## Contact

**Antonio Mainenti**  
GitHub: [@ninuxi](https://github.com/ninuxi)  
Email: oggettosonoro@gmail.com

Public showcase: https://github.com/ninuxi/mood-showcase

---

## Acknowledgments

Built on top of the excellent **datapizza-ai** framework by @datapizza-labs.

Thank you for creating such a flexible and powerful system! 🎉
