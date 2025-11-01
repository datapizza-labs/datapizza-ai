"""
MOOD Multi-Agent System - Multi-Agent Content Generation
=========================================================

Copyright (c) 2025 Antonio Mainenti
Licensed under MIT License (see LICENSE_MOOD_CONTRIBUTIONS)

Author: Antonio Mainenti (https://github.com/ninuxi)
Project: MOOD - Adaptive Artistic Environment

ATTRIBUTION REQUIRED when using this code.

=========================================================

Sistema Multi-Agent per generazione contenuti di alta qualità.

Workflow:
1. WriterAgent: Scrive bozza contenuto (email, post, articolo)
2. CriticAgent: Analizza e trova problemi (punti deboli, esagerazioni, concetti poco chiari)
3. ReviserAgent: Riscrive contenuto basandosi sulle critiche

Uso:
    from datapizza.agents.multi_agent import MultiAgentContentTeam
    
    team = MultiAgentContentTeam(client=client, profile=profile)
    result = team.create_content(
        content_type="linkedin_post",
        topic="MOOD sistema AI per musei",
        target_audience="curatori, event manager",
        tone="professionale"
    )
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass

from datapizza.agents.agent import Agent
from datapizza.agents.image_generator_simple import ImageGenerator
from datapizza.core.clients.client import Client
from datapizza.memory.memory import Memory
from datapizza.tools.tools import tool
from .personal_profile import PersonalProfile


@dataclass
class CritiqueFeedback:
    """Feedback strutturato dall'agente critico."""
    weak_points: List[str]  # Punti deboli
    exaggerations: List[str]  # Esagerazioni
    unclear_concepts: List[str]  # Concetti poco chiari
    tone_issues: List[str]  # Problemi di tono
    suggestions: List[str]  # Suggerimenti miglioramento
    overall_score: float  # 0-10 qualità complessiva


class WriterAgent(Agent):
    """Agente specializzato nella scrittura di contenuti marketing."""
    
    name = "writer-agent"
    system_prompt = """You are an expert marketing content writer specializing in B2B technology products.
Your goal is to create compelling, professional content that:
- Captures attention with strong opening
- Clearly communicates value proposition
- Uses concrete examples and proof points
- Includes clear call-to-action
- Maintains appropriate tone for target audience

Write in Italian unless specified otherwise."""

    def __init__(self, client: Client, profile: Optional[PersonalProfile] = None):
        self.profile = profile
        sys_prompt = self._build_system_prompt(profile)
        super().__init__(name=self.name, client=client, system_prompt=sys_prompt, stateless=True)

    def _build_system_prompt(self, profile: Optional[PersonalProfile]) -> str:
        if not profile:
            return self.system_prompt
        
        lines = [self.system_prompt, "\n--- PROFILE CONTEXT ---"]
        lines.append(f"Writer name: {profile.name}")
        if profile.title:
            lines.append(f"Title: {profile.title}")
        
        # Featured product prominently
        if profile.featured_product:
            fp = profile.featured_product
            lines.append(f"\n🎯 PRIMARY PRODUCT: {fp.name}")
            if fp.tagline:
                lines.append(f"Tagline: {fp.tagline}")
            if fp.description:
                lines.append(f"Description: {fp.description}")
            if fp.key_features:
                lines.append("Key Features:")
                for feat in fp.key_features[:5]:  # Top 5
                    lines.append(f"  • {feat}")
            if fp.target_audience:
                lines.append("Target Audience: " + ", ".join(fp.target_audience[:3]))
            if fp.benefits:
                lines.append("Benefits: " + ", ".join(fp.benefits[:3]))
        
        return "\n".join(lines)

    @tool
    def write_email(self, company_name: str, offer: str, tone: str = "professionale") -> str:
        """Scrive una email di outreach professionale."""
        prompt = f"""Scrivi una email di cold outreach in italiano per {company_name}.

Offerta: {offer}
Tono: {tone}

Struttura richiesta:
1. Oggetto: Breve e specifico (max 60 caratteri)
2. Apertura: Riferimento personalizzato all'azienda (1-2 righe)
3. Proposizione valore: Cosa offri e perché è rilevante (2-3 righe)
4. Proof point: Risultato concreto o caso d'uso (1-2 righe)
5. CTA: Call-to-action chiara con link

Lunghezza totale: Max 150 parole."""
        
        response = self._client.invoke(input=prompt, tools=[], memory=Memory())
        return response.text or ""

    @tool
    def write_linkedin_post(
        self, 
        topic: str, 
        length: str = "medio",
        tone: str = "professionale",
        audience: str = "decision makers"
    ) -> str:
        """Scrive un post LinkedIn coinvolgente."""
        word_count = {"breve": "100-150", "medio": "200-300", "lungo": "400-500"}
        
        prompt = f"""Scrivi un post LinkedIn in italiano.

Topic: {topic}
Audience: {audience}
Tone: {tone}
Lunghezza: {word_count.get(length, '200-300')} parole

Struttura richiesta:
1. Hook: Prima frase che cattura attenzione
2. Problem/Insight: Problema o insight interessante (2-3 righe)
3. Solution/Value: La tua soluzione o valore offerto (3-4 righe)
4. Proof: Esempio concreto o metrica (1-2 righe)
5. CTA: Call-to-action engaging

Usa:
- Paragrafi brevi (max 2-3 righe)
- Emoji strategicamente (2-3 totali, non esagerare)
- Hashtag pertinenti alla fine (5-7)"""
        
        response = self._client.invoke(input=prompt, tools=[], memory=Memory())
        return response.text or ""

    @tool
    def write_article(self, topic: str, angle: str, word_count: int = 800) -> str:
        """Scrive un articolo di blog/thought leadership."""
        prompt = f"""Scrivi un articolo professionale in italiano.

Topic: {topic}
Angle/Prospettiva: {angle}
Lunghezza target: {word_count} parole

Struttura richiesta:
1. Titolo: Compelling e specifico
2. Intro: Hook + preview del valore (50-80 parole)
3. Sezioni: 3-4 sezioni con sottotitoli chiari
4. Esempi concreti: Almeno 2 case study o esempi pratici
5. Conclusione: Riassunto + CTA

Stile:
- Professionale ma accessibile
- Dati e proof points quando possibile
- Evita jargon inutile
- Usa liste puntate per leggibilità"""
        
        response = self._client.invoke(input=prompt, tools=[], memory=Memory())
        return response.text or ""


class CriticAgent(Agent):
    """Agente specializzato nell'analisi critica di contenuti."""
    
    name = "critic-agent"
    system_prompt = """You are a critical content reviewer with expertise in marketing and communication.
Your role is to find problems and suggest improvements in content.

Analyze content for:
1. Weak points: Claims without proof, vague statements, weak value propositions
2. Exaggerations: Unrealistic claims, hyperbole, unsubstantiated superlatives
3. Unclear concepts: Jargon, complex sentences, ambiguous statements
4. Tone issues: Inconsistencies, inappropriate formality, sales-y language
5. Structure problems: Poor flow, missing elements, weak CTA

Be constructive but rigorous. Provide specific examples and actionable feedback.
Respond in Italian."""

    def __init__(self, client: Client):
        super().__init__(name=self.name, client=client, system_prompt=self.system_prompt, stateless=True)

    @tool
    def critique_content(self, content: str, content_type: str = "email") -> str:
        """Analizza contenuto e fornisce feedback critico strutturato."""
        prompt = f"""Analizza questo {content_type} e trova problemi specifici.

CONTENUTO DA ANALIZZARE:
{content}

Fornisci feedback strutturato in questo formato:

### 🔴 PUNTI DEBOLI (3 principali)
1. [Punto debole specifico con esempio dal testo]
2. [Punto debole specifico con esempio dal testo]
3. [Punto debole specifico con esempio dal testo]

### ⚠️ ESAGERAZIONI (3 principali)
1. [Claim esagerato o non supportato con citazione]
2. [Claim esagerato o non supportato con citazione]
3. [Claim esagerato o non supportato con citazione]

### ❓ CONCETTI POCO CHIARI (3 principali)
1. [Concetto confuso o ambiguo con citazione]
2. [Concetto confuso o ambiguo con citazione]
3. [Concetto confuso o ambiguo con citazione]

### 🎯 TONE ISSUES (se presenti)
- [Problemi di tono o inconsistenze]

### 💡 SUGGERIMENTI PRIORITARI
1. [Suggerimento specifico e attuabile]
2. [Suggerimento specifico e attuabile]
3. [Suggerimento specifico e attuabile]

### 📊 SCORE COMPLESSIVO
Qualità: X/10
Rationale: [Breve spiegazione score]"""
        
        response = self._client.invoke(input=prompt, tools=[], memory=Memory())
        return response.text or ""


class ReviserAgent(Agent):
    """Agente specializzato nella revisione e ottimizzazione di contenuti."""
    
    name = "reviser-agent"
    system_prompt = """You are an expert content editor and optimizer.
Your role is to take original content and critical feedback, then produce an improved version.

Your revision must:
1. Address ALL points raised by the critic
2. Maintain the original intent and key messages
3. Improve clarity, impact, and professionalism
4. Strengthen weak points with concrete details
5. Remove or justify any claims criticized as exaggerated
6. Clarify any concepts marked as unclear
7. Ensure consistent, appropriate tone

The revised version should be significantly better while staying true to the original goal.
Write in Italian."""

    def __init__(self, client: Client, profile: Optional[PersonalProfile] = None):
        self.profile = profile
        sys_prompt = self.system_prompt
        if profile and profile.featured_product:
            fp = profile.featured_product
            sys_prompt += f"\n\nContext: You're revising content about {fp.name}."
        super().__init__(name=self.name, client=client, system_prompt=sys_prompt, stateless=True)

    @tool
    def revise_content(self, original: str, critique: str, content_type: str = "email") -> str:
        """Riscrive contenuto basandosi sul feedback critico."""
        prompt = f"""Riscrivi questo {content_type} incorporando il feedback critico.

CONTENUTO ORIGINALE:
{original}

FEEDBACK CRITICO:
{critique}

Istruzioni per la revisione:
1. Affronta TUTTI i punti deboli identificati
2. Rimuovi o giustifica con dati le esagerazioni
3. Chiarisci tutti i concetti marcati come poco chiari
4. Risolvi i problemi di tono
5. Mantieni l'intento e i messaggi chiave dell'originale
6. Migliora la struttura e il flow
7. Rafforza la CTA

OUTPUT: Fornisci SOLO la versione revisionata finale, senza commenti o spiegazioni."""
        
        response = self._client.invoke(input=prompt, tools=[], memory=Memory())
        return response.text or ""


class MultiAgentContentTeam:
    """Team di agenti collaborativi per generazione contenuti di alta qualità."""
    
    def __init__(self, client: Client, profile: Optional[PersonalProfile] = None):
        self.writer = WriterAgent(client=client, profile=profile)
        self.critic = CriticAgent(client=client)
        self.reviser = ReviserAgent(client=client, profile=profile)
        self.profile = profile
        
        # Initialize image generator (Gemini only)
        import os
        google_key = os.getenv("GOOGLE_API_KEY", "")
        self.image_generator = ImageGenerator(api_key=google_key)

    def create_email(
        self, 
        company_name: str, 
        offer: str, 
        tone: str = "professionale",
        iterations: int = 1
    ) -> Dict[str, Any]:
        """Workflow completo: Write → Critique → Revise per email."""
        print(f"\n🤖 WRITER: Generazione email per {company_name}...")
        draft = self.writer.write_email(company_name=company_name, offer=offer, tone=tone)
        
        print(f"\n🔍 CRITIC: Analisi critica della bozza...")
        critique = self.critic.critique_content(content=draft, content_type="email")
        
        print(f"\n✏️ REVISER: Revisione basata su feedback...")
        final = self.reviser.revise_content(original=draft, critique=critique, content_type="email")
        
        return {
            "draft": draft,
            "critique": critique,
            "final": final,
            "company": company_name,
            "offer": offer,
            "tone": tone
        }

    def create_linkedin_post(
        self,
        topic: str,
        length: str = "medio",
        tone: str = "professionale",
        audience: str = "decision makers"
    ) -> Dict[str, Any]:
        """Workflow completo: Write → Critique → Revise per LinkedIn post."""
        print(f"\n🤖 WRITER: Generazione post LinkedIn su '{topic}'...")
        draft = self.writer.write_linkedin_post(
            topic=topic, length=length, tone=tone, audience=audience
        )
        
        print(f"\n🔍 CRITIC: Analisi critica del post...")
        critique = self.critic.critique_content(content=draft, content_type="LinkedIn post")
        
        print(f"\n✏️ REVISER: Revisione basata su feedback...")
        final = self.reviser.revise_content(
            original=draft, critique=critique, content_type="LinkedIn post"
        )
        
        return {
            "draft": draft,
            "critique": critique,
            "final": final,
            "topic": topic,
            "length": length
        }

    def create_article(
        self,
        topic: str,
        angle: str,
        word_count: int = 800
    ) -> Dict[str, Any]:
        """Workflow completo: Write → Critique → Revise per articolo."""
        print(f"\n🤖 WRITER: Generazione articolo '{topic}'...")
        draft = self.writer.write_article(topic=topic, angle=angle, word_count=word_count)
        
        print(f"\n🔍 CRITIC: Analisi critica dell'articolo...")
        critique = self.critic.critique_content(content=draft, content_type="articolo")
        
        print(f"\n✏️ REVISER: Revisione basata su feedback...")
        final = self.reviser.revise_content(
            original=draft, critique=critique, content_type="articolo"
        )
        
        return {
            "draft": draft,
            "critique": critique,
            "final": final,
            "topic": topic,
            "word_count": word_count
        }
    
    def generate_instagram_post(
        self,
        topic: str,
        target_audience: str = "Museum Directors",
        style: str = "Educational"
    ) -> Dict[str, Any]:
        """Workflow completo: Write → Critique → Revise per Instagram post."""
        print(f"\n🤖 WRITER: Generazione Instagram post su '{topic}'...")
        
        # Generate draft using Writer agent (similar to email/linkedin)
        prompt = f"""Crea un post Instagram professionale per promuovere MOOD.

Topic: {topic}
Target Audience: {target_audience}
Style: {style}

Il post deve:
- Catturare l'attenzione nei primi 2 righe
- Spiegare il valore di MOOD per {target_audience}
- Includere call-to-action chiara
- Essere conciso (massimo 250 parole)
- Tono: {style.lower()}

Scrivi SOLO il testo del post, senza hashtag."""
        
        draft = self.writer._client.invoke(input=prompt, tools=[], memory=Memory()).text or ""
        
        print(f"\n🔍 CRITIC: Analisi critica del post Instagram...")
        critique = self.critic.critique_content(
            content=draft, 
            content_type="Instagram post"
        )
        
        print(f"\n✏️ REVISER: Revisione basata su feedback...")
        final = self.reviser.revise_content(
            original=draft, 
            critique=critique, 
            content_type="Instagram post"
        )
        
        return {
            "draft": {"content": draft},
            "critique": {"feedback": critique},
            "final": {"content": final},
            "topic": topic,
            "target_audience": target_audience,
            "style": style
        }
    
    def generate_instagram_post_with_image(
        self,
        topic: str,
        target_audience: str = "Museum Directors",
        style: str = "Educational",
        image_style: str = "modern",
        image_provider: str = "gemini"
    ) -> Dict[str, Any]:
        """
        Workflow completo per Instagram: testo + immagine generata.
        
        Args:
            topic: Argomento del post
            target_audience: Target audience
            style: Stile testo (Educational, Promotional, Inspirational)
            image_style: Stile immagine (modern, artistic, professional, etc.)
            image_provider: Provider immagine (gemini, banana, copilot)
        
        Returns:
            Dict con testo del post e immagine generata
        """
        # Generazione testo (workflow standard)
        print(f"\n📝 Generando testo Instagram per '{topic}'...")
        post_result = self.generate_instagram_post(
            topic=topic,
            target_audience=target_audience,
            style=style
        )
        
        # Estrai il testo finale per la descrizione dell'immagine
        final_text = post_result['final']['content']
        
        # Generazione immagine
        print(f"\n🎨 Generando immagine con Gemini per '{topic}'...")
        image_result = self.image_generator.generate_image(
            topic=topic,
            target_audience=target_audience,
            style=style,
            image_style=image_style,
            provider="gemini"
        )
        
        # Combina risultati
        return {
            "draft": post_result['draft'],
            "critique": post_result['critique'],
            "final": post_result['final'],
            "image": image_result,
            "topic": topic,
            "target_audience": target_audience,
            "text_style": style,
            "image_style": image_style
        }
    
    def generate_linkedin_personal_post(
        self,
        post_type: str = "Thought Leadership",
        topic: str = "",
        description: str = "",
        personal_brand: str = "Innovator & Developer"
    ) -> Dict[str, Any]:
        """
        Generazione post LinkedIn per account personale.
        
        Post types:
        - Thought Leadership: insights e reflections
        - Project Showcase: mostra il lavoro
        - Behind the Scenes: dietro le quinte del lavoro
        - Insights/Articles: articoli e analysis
        - Career Update: news professionali
        - Learning & Growth: cosa stai imparando
        
        Args:
            post_type: Tipo di post (vedi sopra)
            topic: Argomento/titolo del post
            description: Descrizione aggiuntiva
            personal_brand: Come ti descrivi professionalmente
        
        Returns:
            Dict con draft/critique/final del post LinkedIn
        """
        print(f"\n🤖 WRITER: Generazione LinkedIn post ({post_type})...")
        
        prompt = f"""Scrivi un post LinkedIn personale per un account individuale.

PERSONAL BRAND: {personal_brand}
POST TYPE: {post_type}
TOPIC: {topic}
DESCRIPTION: {description}

Requisiti per post LinkedIn personale:

1. TONE & VOICE
   - Autentico e personale (non corporate)
   - Conversazionale ma professionale
   - Mostra personality, non robotic
   - Usa "io" e racconti personali

2. STRUCTURE
   - Hook forte nei primi 2-3 righe (must engage)
   - Story o insight personale (3-5 frasi)
   - Lesson learned o takeaway (2-3 punti)
   - Call-to-action soft (domanda, observation, invitation)

3. CONTENT TIPS
   - Specificity > generality (numeri, esempi concreti)
   - Vulnerabilità > perfezionismo (ammetti sfide)
   - Generosity > selling (condividi valore, non vendi)
   - Short paragraphs/lines (mobile-friendly)

4. LENGTH
   - Massimo 2-3 paragrafi
   - Approx 150-250 parole
   - Break con spazi per readability

5. POST TYPE SPECIFIC:
   - Thought Leadership: condividi perspective unica su trend
   - Project Showcase: racconta il progetto con storie
   - Behind the Scenes: mostra il processo, le sfide
   - Insights/Articles: share learnings da esperienza
   - Career Update: annuncia news con context personale
   - Learning & Growth: cosa stai scoprendo/imparando

Scrivi un post autentico, engageable e ispirato. In italiano."""
        
        draft = self.writer._client.invoke(input=prompt, tools=[], memory=Memory()).text or ""
        
        print(f"\n🔍 CRITIC: Analisi critica del post LinkedIn...")
        critique = self.critic.critique_content(
            content=draft,
            content_type="LinkedIn personal post"
        )
        
        print(f"\n✏️ REVISER: Revisione basata su feedback...")
        final = self.reviser.revise_content(
            original=draft,
            critique=critique,
            content_type="LinkedIn personal post"
        )
        
        return {
            "draft": {"content": draft},
            "critique": {"feedback": critique},
            "final": {"content": final},
            "post_type": post_type,
            "topic": topic,
            "personal_brand": personal_brand
        }
