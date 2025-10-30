"""
🤖 Multi-Agent Dashboard - Versione Semplificata
Interfaccia web per il sistema multi-agent collaborativo
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Configurazione pagina
st.set_page_config(
    page_title="🤖 Multi-Agent Dashboard",
    page_icon="🤖",
    layout="wide"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Multi-Agent Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Sistema collaborativo Writer → Critic → Reviser per contenuti MOOD**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key configurata")
    else:
        st.warning("⚠️ Inserisci la tua API Key")
    
    st.markdown("---")
    st.subheader("📊 Info")
    st.info("""
    **Agenti**: Writer, Critic, Reviser
    **Modello**: Gemini 2.0 Flash
    **Output**: outputs/multi_agent/
    """)

# Tabs principali
tab1, tab2, tab3 = st.tabs(["✉️ Email", "📱 LinkedIn", "📊 Cronologia"])

# TAB 1: EMAIL
with tab1:
    st.header("✉️ Email Professionale")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.text_input(
            "Azienda/Ente",
            placeholder="es. Museo MAXXI Roma"
        )
        offer = st.text_area(
            "Offerta",
            placeholder="es. sistema MOOD per exhibition interattiva"
        )
    
    with col2:
        tone = st.selectbox(
            "Tono",
            ["professionale", "consulenziale", "tecnico", "amichevole"]
        )
    
    if st.button("🚀 Genera Email", type="primary"):
        if not api_key:
            st.error("⚠️ Configura la Google API Key!")
        elif not company or not offer:
            st.error("⚠️ Compila tutti i campi!")
        else:
            with st.spinner("🤖 Gli agenti stanno lavorando..."):
                try:
                    # Esegui il comando CLI
                    cmd = f"""cd /Users/mainenti/datapizza-ai-0.0.2 && \
                    export GOOGLE_API_KEY={api_key} && \
                    python tools/multi_agent_cli.py email-pro \
                    --company "{company}" \
                    --offer "{offer}" \
                    --tone {tone} \
                    --profile configs/personal_profile.yaml"""
                    
                    result = os.popen(cmd).read()
                    
                    # Trova l'ultimo file generato
                    output_dir = Path("outputs/multi_agent")
                    if output_dir.exists():
                        json_files = sorted(output_dir.glob("email_*_full.json"), reverse=True)
                        if json_files:
                            latest = json_files[0]
                            with open(latest) as f:
                                data = json.load(f)
                            
                            st.success("✅ Email generata!")
                            
                            with st.expander("📝 Draft", expanded=False):
                                st.markdown(data.get("draft", ""))
                            
                            with st.expander("🔍 Critique", expanded=True):
                                st.markdown(data.get("critique", ""))
                            
                            with st.expander("✅ Final", expanded=True):
                                st.markdown(data.get("final", ""))
                                st.markdown("---")
                                st.success("**Pronta per l'invio!**")
                        else:
                            st.error("❌ Nessun file generato")
                    else:
                        st.error("❌ Cartella output non trovata")
                        
                except Exception as e:
                    st.error(f"❌ Errore: {e}")

# TAB 2: LINKEDIN
with tab2:
    st.header("📱 Post LinkedIn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input(
            "Argomento",
            placeholder="es. MOOD sistema AI per musei"
        )
        length = st.selectbox(
            "Lunghezza",
            ["breve", "medio", "lungo"],
            index=1
        )
    
    with col2:
        post_tone = st.selectbox(
            "Tono",
            ["professionale", "ispiratore", "tecnico", "storytelling"]
        )
        audience = st.text_input(
            "Audience",
            placeholder="es. curatori musei, technical director"
        )
    
    if st.button("🚀 Genera Post", type="primary"):
        if not api_key:
            st.error("⚠️ Configura la Google API Key!")
        elif not topic:
            st.error("⚠️ Inserisci almeno l'argomento!")
        else:
            with st.spinner("🤖 Gli agenti stanno lavorando..."):
                try:
                    cmd = f"""cd /Users/mainenti/datapizza-ai-0.0.2 && \
                    export GOOGLE_API_KEY={api_key} && \
                    python tools/multi_agent_cli.py post-pro \
                    --topic "{topic}" \
                    --length {length} \
                    --tone {post_tone} \
                    --audience "{audience}" \
                    --profile configs/personal_profile.yaml"""
                    
                    result = os.popen(cmd).read()
                    
                    output_dir = Path("outputs/multi_agent")
                    if output_dir.exists():
                        json_files = sorted(output_dir.glob("linkedin_post_*_full.json"), reverse=True)
                        if json_files:
                            latest = json_files[0]
                            with open(latest) as f:
                                data = json.load(f)
                            
                            st.success("✅ Post generato!")
                            
                            with st.expander("📝 Draft", expanded=False):
                                st.markdown(data.get("draft", ""))
                            
                            with st.expander("🔍 Critique", expanded=True):
                                st.markdown(data.get("critique", ""))
                            
                            with st.expander("✅ Final", expanded=True):
                                st.markdown(data.get("final", ""))
                                char_count = len(data.get("final", ""))
                                st.caption(f"📊 {char_count} caratteri")
                        else:
                            st.error("❌ Nessun file generato")
                    else:
                        st.error("❌ Cartella output non trovata")
                        
                except Exception as e:
                    st.error(f"❌ Errore: {e}")

# TAB 3: CRONOLOGIA
with tab3:
    st.header("📊 Cronologia Output")
    
    output_dir = Path("outputs/multi_agent")
    
    if output_dir.exists():
        json_files = sorted(output_dir.glob("*_full.json"), reverse=True)
        
        if json_files:
            st.success(f"📁 {len(json_files)} contenuti generati")
            
            filter_type = st.selectbox(
                "Filtra",
                ["Tutti", "Email", "LinkedIn", "Articoli"]
            )
            
            for json_file in json_files[:10]:  # Mostra solo ultimi 10
                # Filtraggio
                if filter_type == "Email" and "email" not in json_file.name:
                    continue
                if filter_type == "LinkedIn" and "linkedin" not in json_file.name:
                    continue
                if filter_type == "Articoli" and "article" not in json_file.name:
                    continue
                
                with open(json_file) as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                content_type = metadata.get("content_type", "unknown")
                timestamp = metadata.get("timestamp", "")
                
                with st.expander(f"📄 {content_type.upper()} - {timestamp}"):
                    t1, t2, t3 = st.tabs(["Draft", "Critique", "Final"])
                    
                    with t1:
                        st.markdown(data.get("draft", ""))
                    with t2:
                        st.markdown(data.get("critique", ""))
                    with t3:
                        st.markdown(data.get("final", ""))
        else:
            st.info("📭 Nessun contenuto ancora. Genera il primo!")
    else:
        st.info("📁 Cartella output non esiste ancora")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    🤖 Multi-Agent Dashboard | Powered by datapizza-ai & Google Gemini
</div>
""", unsafe_allow_html=True)
