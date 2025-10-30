"""Web UI per PersonalAgent (Streamlit).

Funzioni:
- Carica profilo personale da YAML
- Prospects -> Enrich -> Emails (A/B) -> LinkedIn Post
- Chat con l'agente

Avvio:
  streamlit run tools/webapp/streamlit_app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import csv

import streamlit as st

# Path setup per import
ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "datapizza-ai-core"
DDG = ROOT / "datapizza-ai-tools" / "duckduckgo"
for p in (CORE, DDG):
    sys.path.insert(0, str(p))

from datapizza.agents.personal_agent import PersonalAgent
from datapizza.agents.personal_profile import PersonalProfile
from datapizza.clients.openai.openai_client import OpenAIClient  # type: ignore
from datapizza.clients.mock_client import MockClient  # type: ignore
from datapizza.tools.duckduckgo.base import DuckDuckGoSearchTool  # type: ignore


def get_client(prompt: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIClient(api_key=api_key, model=model, system_prompt=prompt)
    return MockClient(system_prompt=prompt)


def load_profile(path: Path) -> PersonalProfile | None:
    try:
        return PersonalProfile.load(path)
    except Exception:
        return None


st.set_page_config(page_title="Personal Agent", layout="wide")

st.sidebar.title("Profilo")
profile_path = st.sidebar.text_input(
    "File profilo YAML",
    value=str(ROOT / "configs" / "personal_profile.yaml"),
)
profile = load_profile(Path(profile_path))
if profile:
    st.sidebar.success(f"Profilo caricato: {profile.name}")
else:
    st.sidebar.warning("Profilo non trovato o invalido. Userò i default.")

client = get_client(PersonalAgent.system_prompt)
agent = PersonalAgent(client=client, profile=profile)

st.title("Personal Agent")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prospects", "Enrich", "Emails", "LinkedIn Post", "Chat",
])

with tab1:
    st.header("Prospecting")
    col1, col2, col3 = st.columns(3)
    industry = col1.text_input("Industry", value=(profile.targets.industries[0] if profile and profile.targets.industries else ""))
    location = col2.text_input("Location", value=(profile.targets.locations[0] if profile and profile.targets.locations else ""))
    size = col3.text_input("Size", value=(profile.targets.size or ""))
    role = st.text_input("Role", value=(profile.targets.roles[0] if profile and profile.targets.roles else ""))
    extra = st.text_input("Extra query")
    if st.button("Cerca"):
        ddg = DuckDuckGoSearchTool()
        results = ddg.search(" ".join(filter(None, [industry, "companies", location, size, role, extra])))
        st.session_state["prospects_results"] = results
        # Save CSV
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "prospects.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["company", "url", "snippet"])
            writer.writeheader()
            for r in results:
                writer.writerow({"company": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
        st.success("Salvato outputs/prospects.csv")
    if "prospects_results" in st.session_state:
        st.dataframe(st.session_state["prospects_results"])  # type: ignore

with tab2:
    st.header("Enrichment")
    src = st.text_input("CSV sorgente", value=str(ROOT / "outputs" / "prospects.csv"))
    if st.button("Arricchisci"):
        from urllib.parse import urlparse

        def domain_from_url(url: str) -> str:
            try:
                parsed = urlparse(url)
                host = parsed.netloc or parsed.path
                return host[4:] if host.startswith("www.") else host
            except Exception:
                return ""

        def email_candidates(domain: str):
            base = ["info", "hello", "contact", "hi", "sales"]
            return ", ".join([f"{b}@{domain}" for b in base]) if domain else ""

        rows = []
        with Path(src).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "prospects_enriched.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["company", "url", "domain", "linkedin_url", "email_candidates", "snippet", "notes"],
            )
            writer.writeheader()
            from datapizza.tools.duckduckgo.base import DuckDuckGoSearchTool  # lazy

            ddg = DuckDuckGoSearchTool()
            for r in rows:
                company = r.get("company") or r.get("title") or ""
                url = r.get("url") or r.get("href") or ""
                domain = domain_from_url(url)
                # linkedin search
                linkedin = ""
                try:
                    res = ddg.search(f"{company} linkedin company page")
                    for x in res:
                        if "linkedin.com" in (x.get("href") or ""):
                            linkedin = x.get("href") or ""
                            break
                except Exception:
                    pass
                emails = email_candidates(domain)
                writer.writerow(
                    {
                        "company": company,
                        "url": url,
                        "domain": domain,
                        "linkedin_url": linkedin,
                        "email_candidates": emails,
                        "snippet": r.get("snippet", ""),
                        "notes": "",
                    }
                )
        st.success("Salvato outputs/prospects_enriched.csv")

with tab3:
    st.header("Email drafts")
    src = st.text_input("CSV prospects enriched", value=str(ROOT / "outputs" / "prospects_enriched.csv"))
    offer = st.text_input("Offerta", value=(profile.offer_default if profile else ""))
    tone = st.text_input("Tono", value=(profile.tone.email if profile else "professionale"))
    ab = st.checkbox("A/B testing", value=True)
    cta = st.text_input("CTA link", value=(profile.cta_link if profile else ""))
    if st.button("Genera email"):
        companies = []
        with Path(src).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("company") or row.get("title")
                if name:
                    companies.append(name)
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "emails.csv"
        with out_csv.open("w", newline="") as f:
            if ab:
                writer = csv.DictWriter(f, fieldnames=["company", "email_draft_A", "email_draft_B"])
            else:
                writer = csv.DictWriter(f, fieldnames=["company", "email_draft"])
            writer.writeheader()
            for c in companies:
                if ab:
                    a = agent.draft_email(c, offer, tone=tone, variant="A", cta_link=cta)
                    b = agent.draft_email(c, offer, tone=tone, variant="B", cta_link=cta)
                    writer.writerow({"company": c, "email_draft_A": a, "email_draft_B": b})
                else:
                    d = agent.draft_email(c, offer, tone=tone, variant="A", cta_link=cta)
                    writer.writerow({"company": c, "email_draft": d})
        st.success("Salvato outputs/emails.csv")

with tab4:
    st.header("LinkedIn Post")
    topic = st.text_input("Topic")
    tone = st.text_input("Tono", value=(profile.tone.linkedin if profile else "professionale"))
    audience = st.text_input("Audience", value="clienti")
    length = st.selectbox("Lunghezza", ["breve", "medio", "lungo"], index=0)
    cta = st.text_input("CTA link", value=(profile.cta_link if profile else ""))
    if st.button("Genera Post"):
        post = agent.post_linkedin(topic, tone=tone, audience=audience, cta_link=cta, length=length)
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "linkedin_post.md").write_text(post)
        st.code(post)
        st.success("Salvato outputs/linkedin_post.md")

with tab5:
    st.header("Chat")
    prompt = st.text_input("Prompt")
    if st.button("Invia"):
        res = agent.run(prompt, tool_choice="auto")
        st.write(res.text if res else "")
