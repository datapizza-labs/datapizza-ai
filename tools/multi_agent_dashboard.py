"""
🤖 MOOD Multi-Agent Dashboard
================================
Interfaccia grafica completa per gestire il sistema multi-agent:
- Writer → Critic → Reviser workflow
- Gestione profilo personale e MOOD
- Visualizzazione risultati con confronto Draft vs Final
- Export in formati multipli (Markdown, JSON, PDF)
- Cronologia contenuti generati

Autore: Antonio Mainenti
"""

import streamlit as st
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import sys

# Add parent directory to path to import datapizza modules
sys.path.insert(0, str(Path(__file__).parent.parent / "datapizza-ai-core"))

# Import with error handling
try:
    from datapizza.agents.multi_agent import MultiAgentContentTeam
    from datapizza.clients.google.google_client import GoogleClient
    from datapizza.type.personal_profile import PersonalProfile
except ImportError as e:
    st.error(f"❌ Errore import moduli: {e}")
    st.info("Verifica che datapizza-ai-core sia nel PYTHONPATH")
    st.stop()

# Page config
st.set_page_config(
    page_title="MOOD Multi-Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .critic-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .score-low { color: #dc3545; }
    .score-medium { color: #ffc107; }
    .score-high { color: #28a745; }
    .comparison-box {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'profile' not in st.session_state:
    st.session_state.profile = None


def load_profile(profile_path: str) -> Optional[Dict]:
    """Load personal profile from YAML"""
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"❌ Errore caricamento profilo: {e}")
        return None


def get_client():
    """Get configured LLM client"""
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        return GoogleClient(
            api_key=google_key,
            model="gemini-2.0-flash-exp"
        )
    else:
        st.error("❌ Configura GOOGLE_API_KEY nelle variabili d'ambiente")
        st.stop()


def save_result(result: Dict, content_type: str, output_dir: str = "outputs/multi_agent"):
    """Save generated content to files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON
    json_path = Path(output_dir) / f"{content_type}_{timestamp}_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save individual files
    draft_path = Path(output_dir) / f"{content_type}_{timestamp}_draft.md"
    critique_path = Path(output_dir) / f"{content_type}_{timestamp}_critique.md"
    final_path = Path(output_dir) / f"{content_type}_{timestamp}_final.md"
    
    with open(draft_path, 'w', encoding='utf-8') as f:
        f.write(result['draft'])
    with open(critique_path, 'w', encoding='utf-8') as f:
        f.write(result['critique'])
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write(result['final'])
    
    return {
        'json': str(json_path),
        'draft': str(draft_path),
        'critique': str(critique_path),
        'final': str(final_path)
    }


def extract_score_from_critique(critique: str) -> int:
    """Extract numeric score from critique text"""
    import re
    match = re.search(r'SCORE[:\s]+(\d+)/10', critique, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 5  # Default


def display_comparison(draft: str, final: str, critique: str):
    """Display side-by-side comparison of draft and final"""
    score = extract_score_from_critique(critique)
    
    # Score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score_class = "score-low" if score <= 4 else "score-medium" if score <= 7 else "score-high"
        st.markdown(f'<div class="critic-score {score_class}">{score}/10</div>', unsafe_allow_html=True)
        st.caption("Score Agente Critic")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Confronto", "✍️ Draft", "✨ Final"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Draft (Writer)")
            st.markdown(f'<div class="comparison-box">{draft}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ✅ Final (Reviser)")
            st.markdown(f'<div class="comparison-box">{final}</div>', unsafe_allow_html=True)
        
        # Critique below
        st.markdown("### 🔍 Analisi Critic")
        st.markdown(f'<div class="agent-card">{critique}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📝 Bozza Originale (Writer)")
        st.markdown(draft)
        
        if st.button("📋 Copia Draft", key="copy_draft"):
            st.code(draft, language="markdown")
    
    with tab3:
        st.markdown("### ✨ Versione Finale (Reviser)")
        st.markdown(final)
        
        if st.button("📋 Copia Final", key="copy_final"):
            st.code(final, language="markdown")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configurazione")
    
    # Profile selection
    default_profile = "configs/personal_profile.yaml"
    profile_path = st.text_input(
        "📄 Percorso Profilo",
        value=default_profile,
        help="Path al file personal_profile.yaml"
    )
    
    if st.button("🔄 Carica Profilo"):
        st.session_state.profile = load_profile(profile_path)
        if st.session_state.profile:
            st.success("✅ Profilo caricato!")
    
    # Auto-load on startup
    if st.session_state.profile is None and os.path.exists(profile_path):
        st.session_state.profile = load_profile(profile_path)
    
    # Display profile info
    if st.session_state.profile:
        st.markdown("---")
        st.markdown("### 👤 Profilo Attivo")
        profile = st.session_state.profile
        st.write(f"**Nome**: {profile.get('name', 'N/A')}")
        st.write(f"**Title**: {profile.get('title', 'N/A')[:50]}...")
        
        if 'featured_product' in profile:
            fp = profile['featured_product']
            st.write(f"**Prodotto**: {fp.get('name', 'N/A')}")
            st.write(f"**Tagline**: {fp.get('tagline', 'N/A')[:50]}...")
    
    st.markdown("---")
    
    # LLM status
    st.markdown("### 🤖 LLM Client")
    if os.getenv("GOOGLE_API_KEY"):
        st.success("✅ Google Gemini")
        st.caption("gemini-2.0-flash-exp")
    elif os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI")
        st.caption("gpt-4")
    else:
        st.error("❌ Nessun client configurato")
    
    st.markdown("---")
    
    # History
    st.markdown("### 📜 Cronologia")
    st.write(f"Contenuti generati: **{len(st.session_state.history)}**")
    
    if st.button("🗑️ Pulisci Cronologia"):
        st.session_state.history = []
        st.rerun()


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-header">🤖 MOOD Multi-Agent Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check if profile is loaded
if not st.session_state.profile:
    st.warning("⚠️ Carica il profilo personale dalla sidebar per iniziare")
    st.stop()

# Main tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "✉️ Email Pro",
    "📱 LinkedIn Post Pro", 
    "📰 Articolo Pro",
    "📊 Cronologia"
])

# ============================================================================
# TAB 1: EMAIL PRO
# ============================================================================

with main_tab1:
    st.markdown("## ✉️ Email Professionale Multi-Agent")
    st.markdown("*Writer → Critic → Reviser workflow per email B2B di alta qualità*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_company = st.text_input(
            "🏢 Nome Azienda/Museo",
            value="Museo MAXXI Roma",
            help="Nome dell'organizzazione target"
        )
        
        email_offer = st.text_area(
            "💡 Offerta/Proposta",
            value="sistema MOOD per exhibition interattiva con AI",
            height=100,
            help="Cosa offri/proponi"
        )
    
    with col2:
        email_tone = st.selectbox(
            "🎨 Tone",
            ["professionale", "consulenziale", "tecnico", "amichevole"],
            help="Stile comunicativo dell'email"
        )
        
        email_context = st.text_area(
            "📝 Contesto Aggiuntivo (opzionale)",
            value="",
            height=100,
            help="Info extra per personalizzare l'email"
        )
    
    if st.button("🚀 Genera Email Pro", type="primary", use_container_width=True):
        with st.spinner("⏳ Writer sta scrivendo la bozza..."):
            try:
                client = get_client()
                team = MultiAgentContentTeam(
                    client=client,
                    profile=st.session_state.profile
                )
                
                # Create email through multi-agent pipeline
                result = team.create_email(
                    company_name=email_company,
                    offer=email_offer,
                    tone=email_tone,
                    additional_context=email_context if email_context else None
                )
                
                # Save to session and files
                st.session_state.current_result = result
                result['metadata']['type'] = 'email'
                result['metadata']['company'] = email_company
                
                files = save_result(result, "email")
                result['files'] = files
                
                st.session_state.history.insert(0, result)
                
                st.success("✅ Email generata con successo!")
                
                # Display comparison
                display_comparison(result['draft'], result['final'], result['critique'])
                
                # Download buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Draft",
                        result['draft'],
                        file_name=f"email_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    st.download_button(
                        "📥 Download Critique",
                        result['critique'],
                        file_name=f"email_critique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col3:
                    st.download_button(
                        "📥 Download Final",
                        result['final'],
                        file_name=f"email_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
            except Exception as e:
                st.error(f"❌ Errore: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# TAB 2: LINKEDIN POST PRO
# ============================================================================

with main_tab2:
    st.markdown("## 📱 LinkedIn Post Professionale Multi-Agent")
    st.markdown("*Writer → Critic → Reviser workflow per post LinkedIn impattanti*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        post_topic = st.text_input(
            "💡 Topic/Argomento",
            value="MOOD sistema AI per musei che si adatta al pubblico",
            help="Di cosa parla il post"
        )
        
        post_length = st.selectbox(
            "📏 Lunghezza",
            ["breve", "medio", "lungo"],
            index=2,
            help="breve: 100-150 parole, medio: 200-300, lungo: 400-500"
        )
        
        post_tone = st.selectbox(
            "🎨 Tone",
            ["professionale", "ispirazionale", "tecnico", "storytelling"],
            help="Stile comunicativo del post"
        )
    
    with col2:
        post_audience = st.text_input(
            "👥 Target Audience",
            value="curatori musei, technical director, event manager",
            help="A chi è rivolto il post"
        )
        
        post_cta = st.text_input(
            "📣 CTA (opzionale)",
            value="",
            help="Call-to-action specifica (se vuota, l'agente la genera)"
        )
        
        post_hashtags = st.checkbox(
            "🏷️ Includi Hashtag",
            value=True,
            help="Aggiunge hashtag pertinenti"
        )
    
    if st.button("🚀 Genera Post Pro", type="primary", use_container_width=True):
        with st.spinner("⏳ Writer sta creando il post..."):
            try:
                client = get_client()
                team = MultiAgentContentTeam(
                    client=client,
                    profile=st.session_state.profile
                )
                
                # Create LinkedIn post through multi-agent pipeline
                result = team.create_linkedin_post(
                    topic=post_topic,
                    length=post_length,
                    tone=post_tone,
                    target_audience=post_audience,
                    include_hashtags=post_hashtags,
                    cta=post_cta if post_cta else None
                )
                
                # Save to session and files
                st.session_state.current_result = result
                result['metadata']['type'] = 'linkedin_post'
                result['metadata']['topic'] = post_topic
                
                files = save_result(result, "linkedin_post")
                result['files'] = files
                
                st.session_state.history.insert(0, result)
                
                st.success("✅ Post LinkedIn generato con successo!")
                
                # Display comparison
                display_comparison(result['draft'], result['final'], result['critique'])
                
                # Download buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Draft",
                        result['draft'],
                        file_name=f"linkedin_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    st.download_button(
                        "📥 Download Critique",
                        result['critique'],
                        file_name=f"linkedin_critique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col3:
                    st.download_button(
                        "📥 Download Final",
                        result['final'],
                        file_name=f"linkedin_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
            except Exception as e:
                st.error(f"❌ Errore: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# TAB 3: ARTICOLO PRO
# ============================================================================

with main_tab3:
    st.markdown("## 📰 Articolo Professionale Multi-Agent")
    st.markdown("*Writer → Critic → Reviser workflow per articoli thought leadership*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        article_topic = st.text_input(
            "💡 Topic Articolo",
            value="Il futuro dei musei interattivi con AI",
            help="Argomento principale dell'articolo"
        )
        
        article_angle = st.text_area(
            "🎯 Angolo/Prospettiva",
            value="Come l'AI sta trasformando l'esperienza museale da statica ad adattiva",
            height=100,
            help="Angolazione o tesi dell'articolo"
        )
    
    with col2:
        article_words = st.number_input(
            "📏 Lunghezza (parole)",
            min_value=300,
            max_value=2000,
            value=800,
            step=100,
            help="Numero target di parole"
        )
        
        article_tone = st.selectbox(
            "🎨 Tone",
            ["professionale", "accademico", "divulgativo", "tecnico"],
            help="Stile dell'articolo"
        )
    
    article_sections = st.text_area(
        "📋 Sezioni Richieste (opzionale, una per riga)",
        value="",
        height=100,
        help="Lascia vuoto per struttura automatica"
    )
    
    if st.button("🚀 Genera Articolo Pro", type="primary", use_container_width=True):
        with st.spinner("⏳ Writer sta scrivendo l'articolo (può richiedere più tempo)..."):
            try:
                client = get_client()
                team = MultiAgentContentTeam(
                    client=client,
                    profile=st.session_state.profile
                )
                
                # Parse sections if provided
                sections = None
                if article_sections.strip():
                    sections = [s.strip() for s in article_sections.split('\n') if s.strip()]
                
                # Create article through multi-agent pipeline
                result = team.create_article(
                    topic=article_topic,
                    angle=article_angle,
                    word_count=article_words,
                    tone=article_tone,
                    sections=sections
                )
                
                # Save to session and files
                st.session_state.current_result = result
                result['metadata']['type'] = 'article'
                result['metadata']['topic'] = article_topic
                
                files = save_result(result, "article")
                result['files'] = files
                
                st.session_state.history.insert(0, result)
                
                st.success("✅ Articolo generato con successo!")
                
                # Display comparison
                display_comparison(result['draft'], result['final'], result['critique'])
                
                # Download buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Draft",
                        result['draft'],
                        file_name=f"article_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    st.download_button(
                        "📥 Download Critique",
                        result['critique'],
                        file_name=f"article_critique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with col3:
                    st.download_button(
                        "📥 Download Final",
                        result['final'],
                        file_name=f"article_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
            except Exception as e:
                st.error(f"❌ Errore: {e}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# TAB 4: CRONOLOGIA
# ============================================================================

with main_tab4:
    st.markdown("## 📊 Cronologia Contenuti Generati")
    
    if not st.session_state.history:
        st.info("📭 Nessun contenuto generato ancora. Inizia dai tab sopra!")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Filtra per Tipo",
                ["Tutti", "email", "linkedin_post", "article"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Ordina per",
                ["Più Recente", "Score Più Alto", "Score Più Basso"]
            )
        
        # Filter and sort
        filtered = st.session_state.history
        if filter_type != "Tutti":
            filtered = [h for h in filtered if h['metadata'].get('type') == filter_type]
        
        if sort_by == "Score Più Alto":
            filtered = sorted(filtered, key=lambda x: extract_score_from_critique(x['critique']), reverse=True)
        elif sort_by == "Score Più Basso":
            filtered = sorted(filtered, key=lambda x: extract_score_from_critique(x['critique']))
        
        # Display history items
        for idx, item in enumerate(filtered):
            with st.expander(
                f"📄 {item['metadata'].get('type', 'unknown').upper()} - "
                f"{item['metadata'].get('company', item['metadata'].get('topic', 'N/A'))} - "
                f"Score: {extract_score_from_critique(item['critique'])}/10",
                expanded=False
            ):
                display_comparison(item['draft'], item['final'], item['critique'])
                
                # Show files if available
                if 'files' in item:
                    st.markdown("---")
                    st.markdown("**📁 File Salvati:**")
                    st.json(item['files'])

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #6c757d;">'
    '🤖 MOOD Multi-Agent Dashboard | Powered by datapizza-ai | '
    f'<a href="https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system" target="_blank">MOOD on GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)
