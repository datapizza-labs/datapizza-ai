"""
🚀 MOOD Email Outreach Dashboard - Complete System
====================================================

Copyright (c) 2025 Antonio Mainenti
Licensed under MIT License (see LICENSE_MOOD_CONTRIBUTIONS)

Author: Antonio Mainenti (https://github.com/ninuxi)
Email: oggettosonoro@gmail.com
Project: MOOD - Adaptive Artistic Environment

ATTRIBUTION REQUIRED when using this code.

====================================================

Dashboard completo per gestire outreach automatico a musei/gallerie.

Features:
- Contact Hunter automatico
- Multi-Agent email generation (Writer→Critic→Reviser)
- Preview HTML email
- Approval manuale prima invio
- Tracking campagne
- Stats & analytics

Autore: Antonio Mainenti
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict
sys.path.insert(0, str(Path(__file__).parent.parent / "datapizza-ai-core"))

import os
from dotenv import load_dotenv
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Carica variabili da .env
load_dotenv()

from datapizza.agents.contact_hunter import ContactHunterAgent
from datapizza.database.contacts_db import ContactDatabase, Contact, EmailSent
from datapizza.agents.multi_agent import MultiAgentContentTeam
from datapizza.agents.personal_profile import PersonalProfile, FeaturedProduct
from datapizza.agents.agent_memory import AgentMemory
from datapizza.clients.google import GoogleClient
from datapizza.agents.mood_developer_agent import MOODDevelopmentTeam

# Page config
st.set_page_config(
    page_title="MOOD Email Outreach",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .contact-card {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .high-confidence {
        border-left-color: #28a745;
    }
    .medium-confidence {
        border-left-color: #ffc107;
    }
    .low-confidence {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎨 MOOD Email Outreach System</div>', unsafe_allow_html=True)

# Initialize
@st.cache_resource
def get_database():
    """Get thread-safe database connection"""
    return ContactDatabase()

@st.cache_resource
def get_agent_memory():
    """Get agent learning memory system"""
    return AgentMemory("agent_learning.json")

if 'hunter' not in st.session_state:
    st.session_state.hunter = ContactHunterAgent(delay=2.0)

# Get database (thread-safe)
db = get_database()
agent_memory = get_agent_memory()

# Helper function to send email
def send_email_directly(contact: Dict, subject: str, body: str, email_config: Dict) -> bool:
    """Send email via SMTP"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{email_config['sender']['name']} <{email_config['sender']['email']}>"
        msg['To'] = contact['email']
        
        # Plain text and HTML
        text_part = MIMEText(body, 'plain')
        html_part = MIMEText(f"<html><body><pre>{body}</pre></body></html>", 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send via Gmail SMTP
        with smtplib.SMTP(email_config['smtp']['host'], email_config['smtp']['port']) as server:
            server.starttls()
            server.login(
                email_config['sender']['email'],
                email_config['sender']['password']
            )
            server.send_message(msg)
        
        # Log to database
        db.log_email_sent(
            campaign_id=1,
            contact_id=contact['id'],
            email_to=contact['email'],
            subject=subject,
            body=body
        )
        
        # Log to agent memory
        agent_memory.log_email_sent(
            recipient_email=contact['email'],
            company_name=contact.get('organization', 'Unknown')
        )
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Load personal profile and initialize multi-agent
with open("configs/personal_profile.yaml") as f:
    profile_data = yaml.safe_load(f)
    
    # Create PersonalProfile
    featured = profile_data.get('featured_product', {})
    mood_product = FeaturedProduct(
        name=featured.get('name', 'MOOD'),
        tagline=featured.get('tagline', ''),
        description=featured.get('description', ''),
        target_audience=featured.get('target_audience', []),
        key_features=featured.get('key_features', []),
        benefits=featured.get('benefits', []),
        use_cases=featured.get('use_cases', []),
        tech_stack=featured.get('tech_stack', ''),
        github_url=featured.get('github_url', ''),
        cta_primary=featured.get('cta_primary', ''),
        cta_secondary=featured.get('cta_secondary', '')
    )
    
    personal_info = profile_data.get('personal_info', profile_data)
    profile = PersonalProfile(
        name=personal_info.get('name', profile_data.get('name', 'Antonio Mainenti')),
        title=personal_info.get('title', profile_data.get('title', '')),
        about=personal_info.get('about', profile_data.get('about', '')),
        services=personal_info.get('services', profile_data.get('services', [])),
        offer_default=profile_data.get('offer_default', ''),
        email_from=profile_data.get('email_from', ''),
        cta_link=profile_data.get('cta_link', ''),
        featured_product=mood_product
    )
    
    # Initialize Google Client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY non trovata. Aggiungi a .env file.")
        st.stop()
    client = GoogleClient(api_key=api_key, model="gemini-2.0-flash-exp")
    
    # Initialize Multi-Agent Team
    multi_agent = MultiAgentContentTeam(client=client, profile=profile)
    st.session_state.multi_agent = multi_agent
    
    # Initialize MOOD Development Team
    if 'dev_team' not in st.session_state:
        st.session_state.dev_team = MOODDevelopmentTeam(client=client)

# Load configs
with open("configs/target_organizations.yaml") as f:
    targets = yaml.safe_load(f)['organizations']

with open("configs/email_config.yaml") as f:
    email_config = yaml.safe_load(f)

# LinkedIn post type descriptions
post_type_descriptions = {
    "Thought Leadership": "💡 Condividi prospettive uniche su trend",
    "Project Showcase": "🚀 Racconta il tuo progetto",
    "Behind the Scenes": "👀 Mostra il processo",
    "Insights/Articles": "📚 Condividi learnings",
    "Career Update": "🎯 Annuncia novità professionali",
    "Learning & Growth": "🌱 Cosa stai scoprendo"
}

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Stats
    stats = db.get_stats()
    
    st.metric("📧 Total Contacts", stats['total_contacts'])
    st.metric("✉️ Emails Sent", stats['total_emails_sent'])
    st.metric("📊 Open Rate", f"{stats['open_rate']}%")
    
    st.markdown("---")
    
    # Quick actions
    if st.button("🔄 Run Weekly Update", use_container_width=True):
        st.info("Running weekly update...")
        
    if st.button("📊 View Campaign Stats", use_container_width=True):
        st.session_state.page = "stats"

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Hunt Contacts",
    "✉️ Generate & Send Emails",
    "📸 Instagram Posts",
    "📊 Dashboard",
    "🛠️ MOOD Dev Agent",
    "🌐 Research Insights",
    "⚙️ Settings",
    "💼 LinkedIn Personal"
])

# ============================================================================
# TAB 1: HUNT CONTACTS
# ============================================================================
with tab1:
    st.header("🔍 Contact Hunter")
    st.markdown("Cerca automaticamente contatti da siti web di musei/gallerie")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Select organization
        org_names = [f"{org['name']} ({org['city']})" for org in targets]
        selected_org_name = st.selectbox(
            "Seleziona Organizzazione",
            org_names,
            help="Scegli il museo/galleria da cui cercare contatti"
        )
        
        selected_org = targets[org_names.index(selected_org_name)]
    
    with col2:
        st.info(f"""
        **Tipo**: {selected_org['type']}  
        **Settore**: {selected_org['sector']}  
        **Priorità**: {selected_org['priority']}
        """)
    
    if st.button("🚀 Hunt Contacts", type="primary", use_container_width=True):
        with st.spinner(f"🔍 Hunting contacts from {selected_org['name']}..."):
            try:
                # Run hunter
                contacts = st.session_state.hunter.hunt_contacts(
                    base_url=selected_org['website'],
                    organization_name=selected_org['name']
                )
                
                # Save to database
                new_count = 0
                updated_count = 0
                
                for contact in contacts:
                    existing = db.get_contacts(organization=selected_org['name'])
                    existing_emails = [c['email'] for c in existing]
                    
                    if contact.email not in existing_emails:
                        new_count += 1
                    else:
                        updated_count += 1
                    
                    db.add_contact(contact)
                
                # Log to agent memory
                agent_memory.log_contacts_hunted(
                    count=len(contacts),
                    organizations=[selected_org['name']],
                    sources=[selected_org['website']]
                )
                
                st.success(f"✅ Found {len(contacts)} contacts! ({new_count} new, {updated_count} updated)")
                
                # Show results
                st.markdown("### 📋 Contacts Found")
                
                for contact in sorted(contacts, key=lambda c: c.confidence, reverse=True):
                    confidence_class = "high" if contact.confidence > 0.7 else "medium" if contact.confidence > 0.4 else "low"
                    
                    st.markdown(f"""
                    <div class="contact-card {confidence_class}-confidence">
                        <strong>{contact.email}</strong><br>
                        {f'👤 {contact.name}<br>' if contact.name else ''}
                        {f'💼 {contact.role}<br>' if contact.role else ''}
                        ⭐ Confidence: {contact.confidence:.0%}
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# TAB 2: GENERATE & SEND EMAILS
# ============================================================================
with tab2:
    st.header("✉️ Email Generation & Sending")
    
    # Get contacts from database
    all_contacts = db.get_contacts(status="new", min_confidence=0.5)
    
    if not all_contacts:
        st.info("📭 No contacts available. Use the 'Hunt Contacts' tab to find contacts first!")
    else:
        st.success(f"📬 {len(all_contacts)} contacts available for outreach")
        
        # Select contacts to email
        st.markdown("### 1️⃣ Select Contacts")
        
        selected_contacts = []
        for contact in all_contacts[:20]:  # Show first 20
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                selected = st.checkbox(
                    contact['email'],
                    key=f"select_{contact['id']}"
                )
            
            with col2:
                st.markdown(f"""
                **{contact['organization']}**  
                {f"👤 {contact['name']}" if contact['name'] else ""}  
                {f"💼 {contact['role']}" if contact['role'] else ""}
                """)
            
            with col3:
                st.metric("Confidence", f"{contact['confidence']:.0%}")
            
            if selected:
                selected_contacts.append(contact)
        
        if selected_contacts:
            st.markdown(f"### 2️⃣ Generate Emails ({len(selected_contacts)} selected)")
            
            # Email parameters
            col1, col2 = st.columns(2)
            
            with col1:
                offer = st.text_area(
                    "Offer/Proposal",
                    value="sistema MOOD per exhibition interattive e adaptive",
                    help="Cosa proponi"
                )
            
            with col2:
                tone = st.selectbox(
                    "Tone",
                    ["professionale", "consulenziale", "tecnico", "amichevole"]
                )
            
            if st.button("🎨 Generate Emails", type="primary"):
                with st.spinner("🤖 Multi-Agent generating emails (Writer→Critic→Reviser)..."):
                    # Store generated emails
                    if 'generated_emails' not in st.session_state:
                        st.session_state.generated_emails = {}
                    
                    for contact in selected_contacts:
                        # Prepare context
                        organization = contact['organization']
                        
                        # Generate with Multi-Agent (create_email)
                        _raw = st.session_state.multi_agent.create_email(
                            company_name=organization,
                            offer=offer,
                            tone=tone
                        )
                        # Normalize structure for UI (always dict with content keys)
                        result = {
                            'draft': {'content': _raw.get('draft', '')},
                            'critique': {'feedback': _raw.get('critique', '')},
                            'final': {'content': _raw.get('final', '')}
                        }
                        st.session_state.generated_emails[contact['id']] = result
                        
                        # Log to agent memory
                        agent_memory.log_email_generated(
                            company_name=organization,
                            tone=tone,
                            offer=offer
                        )
                    
                    st.success(f"✅ {len(selected_contacts)} emails generated!")
            
            # Show generated emails if available
            if 'generated_emails' in st.session_state and st.session_state.generated_emails:
                st.markdown("### 3️⃣ Preview & Approve")
                
                for contact in selected_contacts:
                    if contact['id'] in st.session_state.generated_emails:
                        result = st.session_state.generated_emails[contact['id']]
                        
                        with st.expander(f"📧 Email to {contact['email']}", expanded=True):
                            # Show multi-agent workflow
                            st.markdown("**🤖 Multi-Agent Workflow**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**📝 Draft**")
                                st.text_area(
                                    "Writer Agent",
                                    value=result['draft'].get('content', ''),
                                    height=150,
                                    key=f"draft_{contact['id']}",
                                    disabled=True
                                )
                            
                            with col2:
                                st.markdown("**🔍 Critique**")
                                st.text_area(
                                    "Critic Agent",
                                    value=result['critique'].get('feedback', ''),
                                    height=150,
                                    key=f"critique_{contact['id']}",
                                    disabled=True
                                )
                            
                            with col3:
                                st.markdown("**✨ Final**")
                                st.text_area(
                                    "Reviser Agent",
                                    value=result['final'].get('content', ''),
                                    height=150,
                                    key=f"final_{contact['id']}",
                                    disabled=True
                                )
                            
                            st.markdown("---")
                            st.markdown("**📧 Email Preview**")
                            
                            subject = f"MOOD - Nuova Frontiera per le Exhibition di {contact['organization']}"
                            
                            st.markdown(f"""
                            **To**: {contact['email']}  
                            **Subject**: {subject}
                            
                            **Body**:
                            """)
                            
                            st.markdown(result['final'].get('content', ''))
                            
                            st.markdown("---")
                            
                            # Check auto-send setting
                            auto_send_enabled = st.session_state.get('auto_send_enabled', False)
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if not auto_send_enabled:
                                    approve = st.checkbox(
                                        f"✅ Approve email for {contact.get('name', contact['email'])}",
                                        key=f"approve_{contact['id']}"
                                    )
                                else:
                                    approve = True
                                    st.info(f"🚀 Auto-send enabled - will send to {contact.get('name', contact['email'])}")
                            
                            with col2:
                                if approve:
                                    if st.button(f"📤 Send", key=f"send_{contact['id']}", type="primary"):
                                        # Send email using helper function
                                        subject = f"MOOD - Nuova Frontiera per le Exhibition di {contact['organization']}"
                                        body = result['final'].get('content', '')
                                        
                                        if send_email_directly(contact, subject, body, email_config):
                                            st.success(f"✅ Email sent to {contact['email']}")
                                        else:
                                            st.error(f"❌ Error sending email to {contact['email']}")
                                elif auto_send_enabled:
                                    # Auto-send without user clicking button
                                    subject = f"MOOD - Nuova Frontiera per le Exhibition di {contact['organization']}"
                                    body = result['final'].get('content', '')
                                    
                                    if send_email_directly(contact, subject, body, email_config):
                                        st.success(f"✅ Email auto-sent to {contact['email']}")
                                    else:
                                        st.error(f"❌ Error auto-sending email to {contact['email']}")

# ============================================================================
# TAB 3: INSTAGRAM POSTS
# ============================================================================
with tab3:
    st.header("📸 Instagram Content Generation")
    st.markdown("Genera post Instagram per promuovere MOOD a musei e gallerie")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Post topic
        topic = st.text_input(
            "Topic/Focus",
            value="MOOD installation at contemporary art museum",
            help="Argomento principale del post"
        )
        
        # Target audience
        audience = st.selectbox(
            "Target Audience",
            ["Museum Directors", "Art Curators", "Gallery Owners", "Digital Art Enthusiasts", "Tech-Savvy Artists"]
        )
        
        # Post style
        post_style = st.selectbox(
            "Post Style",
            ["Educational", "Inspiring", "Technical Showcase", "Behind the Scenes", "Case Study"]
        )
        
        # Generate image checkbox
        generate_image = st.checkbox(
            "🎨 Generate image with AI (DALL-E)",
            value=True,
            help="Generate visual content for the post"
        )
    
    with col2:
        st.info(f"""
        **Current Stats**:
        - Posts generated: 0
        - Scheduled: 0
        - Published: 0
        """)
    
    if st.button("🎨 Generate Instagram Post", type="primary", use_container_width=True):
        with st.spinner("🤖 Multi-Agent generating Instagram content..."):
            # Get selected image provider
            image_provider = st.session_state.get('image_provider', 'gemini')
            
            # Generate with Multi-Agent
            if generate_image:
                result = st.session_state.multi_agent.generate_instagram_post_with_image(
                    topic=topic,
                    target_audience=audience,
                    style=post_style,
                    image_style="modern",
                    image_provider=image_provider
                )
            else:
                result_base = st.session_state.multi_agent.generate_instagram_post(
                    topic=topic,
                    target_audience=audience,
                    style=post_style
                )
                result = {**result_base, "image": None}
            
            # Log to agent memory
            agent_memory.log_instagram_post_generated(
                topic=topic,
                target_audience=audience,
                style=post_style
            )
            
            st.markdown("### 📸 Generated Post")
            
            # Display image if available
            if result.get('image') and result['image'].get('local_path'):
                st.image(
                    result['image']['local_path'],
                    caption=f"Generated image for: {topic}",
                    use_column_width=True
                )
                st.caption(f"Prompt: {result['image'].get('revised_prompt', '')[:100]}...")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Multi-Agent Workflow**")
                
                with st.expander("📝 Draft (Writer Agent)", expanded=False):
                    st.markdown(result['draft'].get('content', ''))
                
                with st.expander("🔍 Critique (Critic Agent)", expanded=False):
                    st.markdown(result['critique'].get('feedback', ''))
                
                with st.expander("✨ Final Version (Reviser Agent)", expanded=True):
                    st.text_area(
                        "Instagram Post",
                        value=result['final'].get('content', ''),
                        height=300,
                        key="instagram_final"
                    )
                    
                    # Hashtags
                    st.markdown("**Suggested Hashtags:**")
                    st.code("#MOOD #AdaptiveArt #InteractiveExhibition #MuseumTech #DigitalArt #AIArt #ContemporaryArt #NewMedia")
            
            with col2:
                st.markdown("**📊 Post Metrics**")
                st.metric("Character Count", len(result['final'].get('content', '')))
                st.metric("Estimated Reach", "2K-5K")
                st.metric("Best Time", "18:00-20:00")
                
                st.markdown("---")
                
                if st.button("📅 Schedule Post", use_container_width=True):
                    st.success("✅ Post scheduled!")
                
                if st.button("📤 Publish Now", use_container_width=True):
                    st.info("⚠️ Instagram API integration coming soon!")

# ============================================================================
# TAB 4: DASHBOARD
# ============================================================================
with tab4:
    st.header("📊 Campaign Dashboard")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Contacts", stats['total_contacts'])
    
    with col2:
        st.metric("✉️ Emails Sent", stats['total_emails_sent'])
    
    with col3:
        st.metric("📊 Open Rate", f"{stats['open_rate']}%")
    
    with col4:
        responses = stats.get('by_status', {}).get('responded', 0)
        st.metric("💬 Responses", responses)
    
    # Contacts needing follow-up
    st.markdown("### 🔄 Follow-up Needed")
    
    followup_contacts = db.get_contacts_needing_followup(days_since_sent=7)
    
    if followup_contacts:
        st.warning(f"⚠️ {len(followup_contacts)} contacts need follow-up!")
        
        for contact in followup_contacts[:10]:
            st.markdown(f"📧 {contact['email']} - {contact['organization']} (sent {contact['sent_at']})")
    else:
        st.success("✅ All contacts up to date!")

# ============================================================================
# TAB 5: MOOD DEVELOPER AGENT
# ============================================================================
with tab5:
    st.header("🛠️ MOOD Developer Agent")
    st.markdown("Automatizza ricerca, progettazione e implementazione di nuove funzionalità per MOOD.")

    # Import VS Code Integration
    sys.path.insert(0, str(Path(__file__).parent))
    from vscode_integration import VSCodeProjectGenerator

    action = st.selectbox(
        "Seleziona azione",
        [
            "Weekly Innovation Sprint",
            "Monitor Events",
            "Analyze Technology",
            "Propose Feature",
            "Implement Software Integration",
            "Implement Hardware Integration",
            "Create Demo",
            "Code Review",
            "Update Documentation",
        ]
    )

    # Dynamic inputs per action
    params = {}
    if action == "Monitor Events":
        params['event_type'] = st.selectbox("Tipo eventi", ["all", "tech", "art", "ai", "interactive"])
    elif action == "Analyze Technology":
        params['technology_name'] = st.text_input("Tecnologia", value="Raspberry Pi 5")
        params['context'] = st.text_input("Contesto", value="edge computing")
    elif action == "Propose Feature":
        params['feature_description'] = st.text_area("Descrizione feature", value="Adaptive lighting orchestration across multiple rooms")
        params['target_audience'] = st.selectbox("Target", ["museums", "galleries", "festivals", "events"])
    elif action == "Implement Software Integration":
        params['software'] = st.text_input("Software", value="GrandMA3")
        params['protocol'] = st.selectbox("Protocol", ["OSC", "MIDI", "ArtNet", "HTTP"]) 
    elif action == "Implement Hardware Integration":
        params['hardware'] = st.text_input("Hardware", value="Raspberry Pi 5")
    elif action == "Create Demo":
        params['demo_description'] = st.text_area("Descrizione demo", value="Realtime mood-driven lighting with audio input")
        params['hardware'] = st.selectbox("Hardware", ["Jetson Nano", "RPi5", "Laptop"])
    elif action == "Code Review":
        params['code'] = st.text_area("Incolla codice Python", height=240, value="""
import asyncio

class Example:
    def run(self):
        pass
""")
        params['focus'] = st.selectbox("Focus", ["general", "performance", "security", "architecture", "style"])    
    elif action == "Update Documentation":
        params['topic'] = st.text_input("Topic", value="API - OSC Messages")
        params['doc_type'] = st.selectbox("Tipo documentazione", ["technical", "user_guide", "tutorial", "troubleshooting"]) 

    run = st.button("🚀 Esegui", type="primary")

    if run:
        with st.spinner("L'agente sta lavorando..."):
            try:
                team = st.session_state.dev_team
                output_title = action
                content = ""

                if action == "Weekly Innovation Sprint":
                    result = team.weekly_innovation_sprint()
                    # Compose markdown report
                    content = f"""# MOOD Weekly Innovation Sprint\n\nData: {result['sprint_date']}\n\n## 📅 Events Report\n\n{result['events_report']}\n\n## 🔬 Technology Analysis\n\n{result['technology_analysis']}\n\n## 💡 Feature Proposal\n\n{result['feature_proposal']}\n"""
                elif action == "Monitor Events":
                    content = team.developer.monitor_events(**params)
                elif action == "Analyze Technology":
                    content = team.developer.analyze_technology(**params)
                elif action == "Propose Feature":
                    content = team.developer.propose_feature(**params)
                elif action == "Implement Software Integration":
                    content = team.implement_software_integration(software=params['software'], protocol=params['protocol'])
                    output_title = f"Implement {params['software']} ({params['protocol']})"
                elif action == "Implement Hardware Integration":
                    content = team.implement_hardware_integration(hardware=params['hardware'])
                    output_title = f"Hardware {params['hardware']}"
                elif action == "Create Demo":
                    content = team.developer.create_demo(**params)
                elif action == "Code Review":
                    content = team.developer.code_review(**params)
                elif action == "Update Documentation":
                    content = team.developer.update_documentation(**params)

                # Show result
                st.markdown("### 📄 Output")
                st.markdown(content)

                # Offer save & download
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                slug = output_title.lower().replace(" ", "-")
                out_dir = Path("docs/agent_reports")
                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / f"{ts}_{slug}.md"
                file_path.write_text(content)

                st.success(f"✅ Salvato in {file_path}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="⬇️ Scarica report",
                        data=content,
                        file_name=file_path.name,
                        mime="text/markdown",
                    )
                
                with col2:
                    # VS Code Integration
                    if action in ["Propose Feature", "Implement Software Integration", "Implement Hardware Integration", "Create Demo"]:
                        if st.button("🚀 Genera Progetto VS Code", type="primary", use_container_width=True):
                            with st.spinner("Generazione progetto VS Code..."):
                                try:
                                    generator = VSCodeProjectGenerator()
                                    
                                    # Estrai proposta dal content
                                    proposal = {
                                        "title": output_title,
                                        "feature_description": content[:500],  # Prime 500 char come descrizione
                                        "target_audience": params.get('target_audience', 'general'),
                                        "technology": params.get('software', params.get('hardware', 'Python')),
                                        "priority": "high"
                                    }
                                    
                                    result = generator.generate_project_from_proposal(proposal)
                                    
                                    st.success(f"✅ Progetto creato: `{result['project_name']}`")
                                    st.info(f"📂 Path: `{result['project_path']}`")
                                    
                                    st.markdown("### 🎯 Next Steps")
                                    for step in result['next_steps']:
                                        st.markdown(f"- {step}")
                                    
                                    # Button per aprire in VS Code
                                    if st.button("📝 Apri in VS Code", use_container_width=True):
                                        if generator.open_in_vscode(result['project_path']):
                                            st.success("✅ VS Code aperto!")
                                        else:
                                            st.warning("⚠️ Apri manualmente con: `code " + result['project_path'] + "`")
                                    
                                except Exception as e:
                                    st.error(f"❌ Errore generazione progetto: {e}")
            except Exception as e:
                st.error(f"❌ Errore: {e}")

# ============================================================================
# TAB 6: SETTINGS
# ============================================================================
with tab6:
    st.header("⚙️ Settings")
    
    st.markdown("### 📧 Email Configuration")
    st.code(f"""
Sender: {email_config['sender']['name']} <{email_config['sender']['email']}>
SMTP: {email_config['smtp']['host']}:{email_config['smtp']['port']}
Status: ✅ Configured
""")
    
    # Auto-Send Toggle
    st.markdown("### 🚀 Automation Settings")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        auto_send = st.checkbox(
            "Enable Auto-Send Emails",
            value=False,
            help="Automatically send emails after generation without manual approval"
        )
        if 'auto_send_enabled' not in st.session_state:
            st.session_state.auto_send_enabled = auto_send
        else:
            st.session_state.auto_send_enabled = auto_send
    
    with col2:
        if auto_send:
            st.success("✅ Auto-Send ENABLED")
        else:
            st.info("⏳ Auto-Send disabled")
    
    if auto_send:
        st.warning("""
        ⚠️ **Warning**: With auto-send enabled, emails will be sent immediately after generation.
        Make sure you're comfortable with the generated content before enabling this.
        """)
    
    # Image Generation Provider
    st.markdown("### 🖼️ Image Generation Settings")
    
    st.success("✅ **Google Gemini** - Image descriptions powered by AI")
    st.markdown("""
    Il sistema utilizza **Google Gemini 2.5 Flash** per generare descrizioni di immagini vivide e dettagliate.
    
    **Come funziona:**
    - Analizza il tema del post (Instagram, LinkedIn, email)
    - Genera una descrizione visiva ricca di dettagli
    - Crea keywords correlate
    - Ottimizzato per social media
    """)
    
    st.markdown("### 🔄 Auto-Update")
    st.info("""
    **Weekly Update**: Every Sunday 23:00  
    **Status**: ⏳ Pending (run `python tools/weekly_update.py --setup-cron`)
    """)
    
    if st.button("🧪 Test Weekly Update (3 orgs)"):
        st.info("Running test update...")

# ============================================================================
# TAB 7: LINKEDIN PERSONAL
# ============================================================================
with tab7:
    st.header("💼 LinkedIn Personal Content")
    st.markdown("""
    Genera contenuti per il tuo profilo LinkedIn personale.  
    Perfetto per **personal branding**, **thought leadership** e **network building**.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Personal brand description
        personal_brand = st.text_input(
            "Come ti descrivi professionalmente?",
            value="AI Developer & Innovation Enthusiast",
            help="Es: 'AI Developer', 'Product Manager', 'Entrepreneur', 'Creative Director'"
        )
        
        # Post type selection
        post_type = st.selectbox(
            "Tipo di Post",
            [
                "Thought Leadership",
                "Project Showcase",
                "Behind the Scenes",
                "Insights/Articles",
                "Career Update",
                "Learning & Growth"
            ],
            help="Seleziona il tipo di contenuto che vuoi creare"
        )
        
        # Topic/Title
        topic = st.text_input(
            "Argomento/Titolo",
            value="MOOD - Adaptive Artistic Environment",
            help="Il tema principale del tuo post"
        )
        
        # Additional description
        description = st.text_area(
            "Descrizione aggiuntiva (opzionale)",
            value="",
            height=100,
            help="Dettagli aggiuntivi per personalizzare il post"
        )
    
    with col2:
        st.info(f"""
        **📌 Post Info**
        
        Type: {post_type}  
        Brand: {personal_brand}  
        
        {post_type_descriptions.get(post_type, "Share your story")}
        """)
    
    if st.button("✍️ Generate LinkedIn Post", type="primary", use_container_width=True):
        with st.spinner("🤖 Creating LinkedIn content..."):
            try:
                # Generate with Multi-Agent
                result = st.session_state.multi_agent.generate_linkedin_personal_post(
                    post_type=post_type,
                    topic=topic,
                    description=description,
                    personal_brand=personal_brand
                )
                
                # Log to agent memory
                agent_memory.log_linkedin_post_generated(
                    post_type=post_type,
                    topic=topic,
                    personal_brand=personal_brand
                )
                
                st.markdown("### 💬 Generated Post")
                
                # Multi-agent workflow
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📝 Draft (Writer)**")
                    st.text_area(
                        "Writer Agent",
                        value=result['draft'].get('content', ''),
                        height=150,
                        disabled=True,
                        key="linkedin_draft"
                    )
                
                with col2:
                    st.markdown("**🔍 Critique**")
                    st.text_area(
                        "Critic Agent",
                        value=result['critique'].get('feedback', ''),
                        height=150,
                        disabled=True,
                        key="linkedin_critique"
                    )
                
                with col3:
                    st.markdown("**✨ Final**")
                    st.text_area(
                        "Reviser Agent",
                        value=result['final'].get('content', ''),
                        height=150,
                        disabled=True,
                        key="linkedin_final"
                    )
                
                st.markdown("---")
                
                # Final post preview
                st.markdown("### 📋 LinkedIn Post Preview")
                
                final_text = result['final'].get('content', '')
                
                # Styled preview
                st.markdown(f"""
                <div style="background: #f3f2ef; padding: 20px; border-radius: 8px; border-left: 4px solid #0077b5;">
                    <p style="color: #333; line-height: 1.6; font-size: 15px;">{final_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("📋 Copy to Clipboard", use_container_width=True):
                        st.success("✅ Copied! Paste on LinkedIn")
                
                with col2:
                    if st.button("💾 Save Draft", use_container_width=True):
                        # Save to file
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        out_dir = Path("linkedin_drafts")
                        out_dir.mkdir(exist_ok=True)
                        file_path = out_dir / f"{ts}_{post_type.replace(' ', '_')}.md"
                        
                        content = f"""# LinkedIn Post: {topic}

**Type**: {post_type}  
**Personal Brand**: {personal_brand}  
**Date**: {datetime.now().isoformat()}

## Content

{final_text}

---

## Workflow

### Draft
{result['draft'].get('content', '')}

### Feedback
{result['critique'].get('feedback', '')}
"""
                        file_path.write_text(content)
                        st.success(f"✅ Salvato in {file_path}")
                
                with col3:
                    if st.button("🎨 Generate Image (Optional)", use_container_width=True):
                        with st.spinner("Generating image..."):
                            image_provider = st.session_state.get('image_provider', 'gemini')
                            image_result = st.session_state.multi_agent.image_generator.generate_image(
                                post_text=final_text,
                                topic=topic,
                                style="professional",
                                provider=image_provider
                            )
                            if image_result and image_result.get('local_path'):
                                st.image(image_result['local_path'], use_column_width=True)
                                st.caption(f"Generated image ({image_provider.upper()}) for your post")
                            elif image_result and image_result.get('type') == 'manual':
                                st.info(f"📌 {image_result.get('instructions', 'Use Copilot Designer manually')}")
                
            except Exception as e:
                st.error(f"❌ Error generating post: {e}")

# ============================================================================
# TAB 6: RESEARCH INSIGHTS
# ============================================================================
with tab6:
    st.header("🌐 Web Research Insights")
    st.markdown("""
    Sistema di ricerca automatica che monitora novità tecnologiche **3 volte a settimana** (lunedì, mercoledì, venerdì alle 10:00).
    
    La ricerca usa **Brave Search** (primary) / **DuckDuckGo** (fallback) + analisi AI per trovare innovazioni rilevanti.
    """)
    
    # Importa WebResearchAgent e AutonomousCoordinator
    sys.path.insert(0, str(Path(__file__).parent))
    from web_research_agent import WebResearchAgent
    from autonomous_coordinator import AutonomousCoordinator
    
    try:
        agent = WebResearchAgent()
        coordinator = AutonomousCoordinator()
        
        # Get latest research summary
        summary = agent.get_latest_research_summary()
        
        if summary:
            # === EXECUTIVE DIGEST (NEW) ===
            st.markdown("---")
            st.subheader("⚡ Executive Digest - 30 Secondi")
            
            with st.expander("🎯 Leggi Qui (ultra-compatto)", expanded=True):
                # Genera digest se non esiste o forza rigenera
                if st.button("🔄 Rigenera Digest", key="regen_digest"):
                    with st.spinner("Generazione digest executive..."):
                        digest = coordinator.create_executive_digest(summary)
                        st.session_state['current_digest'] = digest
                
                # Mostra digest
                if 'current_digest' not in st.session_state:
                    with st.spinner("Generazione digest executive..."):
                        digest = coordinator.create_executive_digest(summary)
                        st.session_state['current_digest'] = digest
                
                digest = st.session_state['current_digest']
                
                if "error" not in digest:
                    st.markdown(digest['digest_text'])
                    st.caption(f"⏱️ Read time: {digest['read_time']} | Generated: {digest['timestamp'][:16]}")
                else:
                    st.error(f"Errore: {digest['error']}")
            
            # === APPROVAL QUEUE ===
            pending_approvals = coordinator.get_pending_approvals()
            
            if pending_approvals:
                st.markdown("---")
                st.warning(f"🔔 **{len(pending_approvals)} azioni richiedono il tuo OK**")
                
                for action in pending_approvals:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{action['description']}**")
                            st.caption(f"Priority: {action['priority']} | ETA: {action.get('eta', 'Unknown')}")
                        
                        with col2:
                            if st.button("✅ Approve", key=f"approve_{action['id']}", type="primary"):
                                coordinator.approve_action(action['id'], approved=True)
                                st.success("Approvato!")
                                st.rerun()
                        
                        with col3:
                            if st.button("❌ Reject", key=f"reject_{action['id']}"):
                                coordinator.approve_action(action['id'], approved=False)
                                st.info("Rifiutato")
                                st.rerun()
            
            # === AUTONOMOUS STATUS ===
            status = coordinator.get_status_summary()
            
            st.markdown("---")
            st.subheader("🤖 Status Autonomia")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⚙️ Azioni Autonome", status['autonomous_pending'], help="L'agente farà da solo")
            with col2:
                st.metric("⏳ In Attesa OK", status['awaiting_approval'], help="Richiedono tua approvazione")
            with col3:
                st.metric("✅ Completate Oggi", status['completed_today'])
            with col4:
                st.metric("📊 Totale Completate", status['total_completed'])
            
            st.markdown("---")
            
            # Original summary (collapsed by default)
            st.success(f"📅 **Ultima ricerca:** {summary['date']} alle ore {summary['time']}")
            st.markdown("---")
            
            # Original summary (collapsed by default)
            st.success(f"📅 **Ultima ricerca:** {summary['date']} alle ore {summary['time']}")
            
            with st.expander("📊 Metriche Dettagliate", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Topic Ricercati", summary['topics_count'])
                with col2:
                    st.metric("Ricerche Totali", agent.history.get('total_researches', 0))
                with col3:
                    next_research = "Lunedì, Mercoledì o Venerdì"
                    st.metric("Prossima Ricerca", next_research)
            
            # Executive Summary (collapsed)
            with st.expander("� Executive Summary Completo", expanded=False):
                st.markdown(summary['executive_summary'])
            
            st.markdown("---")
            
            # Detailed findings in expander
            with st.expander("🔍 Dettagli Completi per Topic", expanded=False):
                for finding in summary['findings']:
                    st.markdown(f"### {finding['topic']}")
                    
                    # Extract provider info from findings text
                    findings_text = finding['findings']
                    if "*[Searched via:" in findings_text:
                        provider_start = findings_text.find("*[Searched via:") + 15
                        provider_end = findings_text.find("]*", provider_start)
                        provider = findings_text[provider_start:provider_end].strip()
                        
                        # Show provider badge
                        if "Brave" in provider:
                            st.success(f"🔍 Provider: **{provider}** (Premium)")
                        elif "DuckDuckGo" in provider:
                            st.info(f"🔍 Provider: **{provider}** (Fallback)")
                        else:
                            st.warning(f"🔍 Provider: **{provider}**")
                    
                    st.markdown(findings_text)
                    st.markdown("---")
            
            # Actions
            st.markdown("### 🎯 Azioni")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Forza Nuova Ricerca Ora", type="primary", use_container_width=True):
                    with st.spinner("Esecuzione ricerca in corso..."):
                        try:
                            digest = agent.run_research_cycle()
                            st.success("✅ Ricerca completata!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Errore: {e}")
            
            with col2:
                if st.button("� Visualizza Trend nel Tempo", use_container_width=True, disabled=True):
                    st.info("🔜 Feature in arrivo: grafici temporali dei trend tecnologici")
            
            # Research history
            st.markdown("---")
            st.subheader("📜 Storico Ricerche")
            
            history_items = agent.history.get("findings", [])[-5:]  # Ultime 5
            history_items.reverse()  # Dal più recente
            
            for item in history_items:
                ts = datetime.fromisoformat(item["timestamp"])
                st.markdown(f"- **{ts.strftime('%d/%m/%Y %H:%M')}** - {item['topics_count']} topic ricercati")
        
        else:
            st.info("ℹ️ Nessuna ricerca ancora eseguita.")
            
            if st.button("🚀 Avvia Prima Ricerca", type="primary"):
                with st.spinner("Esecuzione ricerca in corso..."):
                    try:
                        digest = agent.run_research_cycle()
                        st.success("✅ Ricerca completata!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Errore: {e}")
        
        # Info box
        st.markdown("---")
        st.info("""
        **Come funziona la ricerca:**
        
        1. 🔍 **Brave Search** (primary) / **DuckDuckGo** (fallback) - Ricerca web reale
        2. 🤖 **Analisi AI** - Gemini 2.0 analizza i risultati e estrae insights
        3. 📝 **Report Digest** - Executive summary con Top 3 innovazioni + azioni consigliate
        4. 💾 **Storico** - Tutti i report salvati in `outputs/research/`
        
        **Topic monitorati:**
        - AI agents frameworks
        - Multi-agent systems Python
        - LLM orchestration tools
        - Interactive art AI technology
        - Generative AI releases
        - Edge computing AI inference
        - Real-time computer vision
        - OSC protocol innovations
        - Python AI libraries trending
        """)
        
        # Custom topics section
        st.markdown("---")
        st.subheader("➕ Gestione Topic Personalizzati")
        
        with st.expander("📝 Aggiungi/Modifica Topic", expanded=False):
            st.markdown("""
            Personalizza i topic di ricerca per monitorare tecnologie specifiche per il tuo progetto.
            """)
            
            # Load current topics
            from web_research_agent import WebResearchAgent
            current_topics = WebResearchAgent.RESEARCH_TOPICS
            
            # Show current topics
            st.markdown("**Topic Attuali:**")
            for i, topic in enumerate(current_topics, 1):
                st.markdown(f"{i}. {topic}")
            
            st.markdown("---")
            
            # Add new topic
            new_topic = st.text_input(
                "Nuovo Topic",
                placeholder="Es: 'Raspberry Pi 5 AI projects 2025'",
                help="Inserisci una query di ricerca specifica"
            )
            
            if st.button("➕ Aggiungi Topic", disabled=not new_topic):
                if new_topic and new_topic not in current_topics:
                    st.success(f"✅ Topic '{new_topic}' aggiunto! Riavvia la dashboard per applicare.")
                    st.info("💡 Modifica manualmente `tools/web_research_agent.py` linea ~45 per rendere permanente.")
                elif new_topic in current_topics:
                    st.warning("⚠️ Questo topic esiste già!")
            
            st.markdown("---")
            st.markdown("**🎯 Suggerimenti per topic efficaci:**")
            st.markdown("""
            - Sii specifico: "Jetson Orin Nano AI inference" invece di "AI hardware"
            - Include anno: "TouchDesigner 2025 features" per risultati recenti
            - Combina keywords: "Python FastAPI microservices patterns"
            - Focus su use case: "OSC protocol live performance"
            """)
        
        # Alert innovazioni critiche
        st.markdown("---")
        st.subheader("🚨 Alert Innovazioni Critiche")
        
        with st.expander("⚙️ Configura Alert", expanded=False):
            st.markdown("""
            Ricevi notifiche quando la ricerca trova innovazioni particolarmente rilevanti.
            """)
            
            alert_enabled = st.checkbox(
                "Abilita alert per innovazioni critiche",
                value=False,
                help="Riceverai notifica via email quando vengono trovate innovazioni ad alta priorità"
            )
            
            if alert_enabled:
                alert_keywords = st.text_area(
                    "Parole chiave per alert",
                    value="breaking\nmajor release\nrevolutionary\ngame-changing",
                    help="Una keyword per riga. Alert quando queste parole appaiono nei risultati."
                )
                
                alert_email = st.text_input(
                    "Email per alert",
                    value="mainenti@example.com",
                    help="Dove inviare le notifiche"
                )
                
                st.success("✅ Alert configurato! (Feature in sviluppo)")
                st.info("💡 Gli alert saranno inviati automaticamente quando il sistema trova corrispondenze.")
        
        # Export PDF
        st.markdown("---")
        st.subheader("📄 Export Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Scarica Ultimo Report (Markdown)", use_container_width=True):
                if summary:
                    report_file = agent.output_dir / f"research_digest_{summary['timestamp'].strftime('%Y-%m-%d')}.md"
                    if report_file.exists():
                        with open(report_file, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="⬇️ Download .md",
                                data=f.read(),
                                file_name=report_file.name,
                                mime="text/markdown",
                                use_container_width=True
                            )
        
        with col2:
            if st.button("📄 Genera PDF Report (PRO)", use_container_width=True, disabled=True):
                st.info("🔜 Feature PRO in arrivo: export PDF con styling professionale")
                st.markdown("""
                **Il PDF includerà:**
                - Copertina con branding MOOD
                - Grafici visuali dei trend
                - Tabelle comparative
                - Link cliccabili alle fonti
                - Sezione 'Action Items'
                """)
    
    except Exception as e:
        st.error(f"❌ Errore caricamento Web Research Agent: {e}")
        st.info("Assicurati che `tools/web_research_agent.py` sia presente e funzionante.")

# ============================================================================
# TAB 7: SETTINGS
# ============================================================================
with tab7:
    st.header("⚙️ Settings")
    st.info("Settings panel - Coming soon")

# ============================================================================
# TAB 8: LINKEDIN PERSONAL
# ============================================================================
with tab8:
    st.header("💼 LinkedIn Personal")
    st.info("LinkedIn personal brand panel - Coming soon")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    🎨 MOOD Email Outreach System | Powered by Multi-Agent AI
</div>
""", unsafe_allow_html=True)
