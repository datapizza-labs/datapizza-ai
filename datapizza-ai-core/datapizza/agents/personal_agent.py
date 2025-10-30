from typing import List, Optional

from datapizza.agents.agent import Agent, Plan, StepResult
from datapizza.clients.mock_client import MockClient
from datapizza.tools.tools import Tool, tool
from datapizza.memory.memory import Memory
from .personal_profile import PersonalProfile


class PersonalAgent(Agent):
    """Un agente personale che aiuta con promozione e outreach.

    Metodi come `post_linkedin`, `draft_email` e `find_targets` sono esposti
    come Tool e possono essere invocati dal planner.
    """

    name = "personal-agent"
    system_prompt = (
        "You are a personal growth and outreach assistant. Help the user find clients, draft outreaches and suggest promotional content."
    )

    def __init__(self, client: Optional[MockClient] = None, profile: Optional[PersonalProfile] = None):
        self.profile = profile
        sys_prompt = self._build_system_prompt(profile)
        client = client or MockClient(system_prompt=sys_prompt)
        super().__init__(name=self.name, client=client, system_prompt=sys_prompt, stateless=True)

    def _build_system_prompt(self, profile: Optional[PersonalProfile]) -> str:
        if not profile:
            return self.system_prompt
        lines = [self.system_prompt]
        lines.append(f"User name: {profile.name}")
        if profile.title:
            lines.append(f"Title: {profile.title}")
        if profile.about:
            lines.append(f"About: {profile.about}")
        if profile.services:
            lines.append("Services: " + ", ".join(profile.services))
        
        # Featured product prominently in system prompt
        if profile.featured_product:
            fp = profile.featured_product
            lines.append(f"\n🎯 FEATURED PRODUCT TO PROMOTE: {fp.name}")
            if fp.tagline:
                lines.append(f"Tagline: {fp.tagline}")
            if fp.description:
                lines.append(f"Description: {fp.description}")
            if fp.key_features:
                lines.append("Key Features:")
                for feat in fp.key_features:
                    lines.append(f"  - {feat}")
            if fp.target_audience:
                lines.append("Target Audience: " + "; ".join(fp.target_audience))
            if fp.benefits:
                lines.append("Benefits: " + "; ".join(fp.benefits))
            if fp.use_cases:
                lines.append("Use Cases: " + "; ".join(fp.use_cases[:3]))  # primi 3
            if fp.github_url:
                lines.append(f"GitHub: {fp.github_url}")
            if fp.cta_primary:
                lines.append(f"Primary CTA: {fp.cta_primary}")
        
        if profile.targets:
            t = profile.targets
            lines.append(
                "Targets: "
                + ", ".join(filter(None, [
                    f"industries={','.join(t.industries) if t.industries else ''}",
                    f"locations={','.join(t.locations) if t.locations else ''}",
                    f"size={t.size or ''}",
                    f"roles={','.join(t.roles) if t.roles else ''}",
                ]))
            )
        if profile.tone:
            lines.append(f"Preferred tone: email={profile.tone.email}, linkedin={profile.tone.linkedin}")
        if profile.cta_link:
            lines.append(f"Default CTA link: {profile.cta_link}")
        return "\n".join(lines)

    @tool
    def post_linkedin(
        self,
        topic: str,
        tone: str = "professionale",
        audience: str = "clienti",
        cta_link: Optional[str] = None,
        length: str = "breve",
    ) -> str:
        """Genera un post LinkedIn promozionale su un argomento.

        Parametri:
        - topic: argomento del post
        - tone: tono (es. professionale, amichevole, autorevole)
        - audience: destinatari
        - cta_link: link alla call/prenotazione
        - length: breve | medio | lungo
        """
        prompt = (
            "Scrivi un post LinkedIn in italiano.\n"
            f"Argomento: {topic}\nTono: {tone}\nAudience: {audience}\nLunghezza: {length}.\n"
            "Usa un'apertura forte, 1-2 frasi di valore e chiudi con una CTA chiara."
        )
        if not cta_link and self.profile and self.profile.cta_link:
            cta_link = self.profile.cta_link
        if self.profile and not tone:
            tone = self.profile.tone.linkedin
        if cta_link:
            prompt += f" Includi questo link come CTA finale: {cta_link}"

        try:
            # Se disponibile, usa l'LLM per migliore qualità
            resp = self._client.invoke(input=prompt, tools=[], memory=Memory())
            return resp.text or ""
        except Exception:
            return (
                f"🚀 Sto lanciando: {topic} — Contattami per collaborazioni! "
                + (f"Prenota qui: {cta_link} " if cta_link else "")
                + "#openforwork"
            )

    @tool
    def draft_email(
        self,
        client_name: str,
        offer: str,
        tone: str = "professionale",
        variant: str = "A",
        cta_link: Optional[str] = None,
        personalization: Optional[str] = None,
    ) -> str:
        """Crea una bozza email di outreach per un cliente target.

        Parametri:
        - client_name: nome azienda o referente
        - offer: offerta/servizio
        - tone: tono (professionale, consulenziale, amichevole)
        - variant: etichetta per A/B testing (A|B|...)
        - cta_link: link prenotazione/incontro
        - personalization: riga personalizzata (note sul prospect)
        """
        prompt = (
            "Scrivi una email di cold outreach in italiano, breve e chiara.\n"
            f"Tono: {tone}. Variante: {variant}.\n"
            f"Azienda/Referente: {client_name}. Offerta: {offer}.\n"
            "Struttura: apertura personalizzata, proposizione di valore concreta, 1 riga di prova/beneficio, CTA."
        )
        if not cta_link and self.profile and self.profile.cta_link:
            cta_link = self.profile.cta_link
        if self.profile and not tone:
            tone = self.profile.tone.email
        if personalization:
            prompt += f" Personalizzazione: {personalization}."
        if cta_link:
            prompt += f" Inserisci una CTA con questo link: {cta_link}."

        try:
            resp = self._client.invoke(input=prompt, tools=[], memory=Memory())
            return resp.text or ""
        except Exception:
            return (
                f"Ciao {client_name},\n\n"
                + (f"{personalization}\n\n" if personalization else "")
                + f"Mi chiamo [Tuo Nome] e offro {offer}. Credo potremmo collaborare per ottenere risultati concreti.\n\n"
                + (f"Prenota qui: {cta_link}\n\n" if cta_link else "")
                + "Posso organizzare una call di 20 minuti la prossima settimana?\n\nSaluti"
            )

    @tool
    def find_targets(self, industry: str, size: str = "SMB", role: str = "Founder") -> List[str]:
        """Ritorna una lista di aziende target (stub locale)."""
        # stub: in produzione qui chiameresti un DB o API
        base = [f"{industry}Co {i}" for i in range(1, 6)]
        return [f"{name} ({size})" for name in base]

