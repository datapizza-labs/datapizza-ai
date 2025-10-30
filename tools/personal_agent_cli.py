"""CLI per PersonalAgent.

Comandi:
  - prospects: cerca potenziali clienti con DuckDuckGo e salva CSV
  - email: genera bozze email di outreach da una lista di aziende
  - linkedin-post: genera un post LinkedIn
  - chat: modalità libera con l'agente (usa tool se supportato dal client)

Uso:
  python tools/personal_agent_cli.py prospects --industry "Design" --location "Milano"
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from urllib.parse import urlparse

import sys


# Risolve i path per importare i pacchetti locali (core + tool duckduckgo)
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "datapizza-ai-core"
DDG = ROOT / "datapizza-ai-tools" / "duckduckgo"
for p in (CORE, DDG):
    sys.path.insert(0, str(p))

from datapizza.agents.personal_agent import PersonalAgent
from datapizza.clients.openai.openai_client import OpenAIClient  # type: ignore
from datapizza.clients.mock_client import MockClient  # type: ignore
from datapizza.tools.duckduckgo.base import DuckDuckGoSearchTool  # type: ignore
from datapizza.agents.personal_profile import PersonalProfile  # type: ignore

# Try to import Google client
try:
    from datapizza.clients.google.google_client import GoogleClient  # type: ignore
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


def get_client(system_prompt: str):
    """Ritorna un client LLM: Google Gemini se GOOGLE_API_KEY, OpenAI se OPENAI_API_KEY, altrimenti MockClient."""
    # Try Google Gemini first
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and GOOGLE_AVAILABLE:
        model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")
        try:
            return GoogleClient(api_key=google_key, model=model, system_prompt=system_prompt)
        except Exception as e:
            print(f"⚠️ Google client error: {e}, falling back to OpenAI or Mock")
    
    # Fallback to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIClient(api_key=openai_key, model=model, system_prompt=system_prompt)
    
    # Final fallback to Mock
    return MockClient(system_prompt=system_prompt)


def _load_profile(profile_path: Optional[str]) -> Optional[PersonalProfile]:
    p = profile_path or str(ROOT / "configs" / "personal_profile.yaml")
    try:
        return PersonalProfile.load(p)
    except Exception:
        return None


def cmd_prospects(args: argparse.Namespace):
    ddg = DuckDuckGoSearchTool()
    terms = [args.industry, "companies", args.location]
    if args.size:
        terms.append(args.size)
    if args.role:
        terms.append(args.role)
    if args.extra:
        terms.append(args.extra)
    query = " ".join([t for t in terms if t]).strip()
    results = ddg.search(query)

    out_dir = Path(args.output_dir or (ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "prospects.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "url", "snippet"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "company": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )

    print(f"Salvato {len(results)} prospect in {out_csv}")


def cmd_email(args: argparse.Namespace):
    profile = _load_profile(args.profile)
    client = get_client(PersonalAgent.system_prompt)
    agent = PersonalAgent(client=client, profile=profile)

    # Sorgente aziende: CSV o lista da argomento
    companies: List[str] = []
    if args.from_csv:
        with Path(args.from_csv).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("company") or row.get("title")
                if name:
                    companies.append(name)
    else:
        companies = args.companies or []

    out_dir = Path(args.output_dir or (ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "emails.csv"
    fields = ["company"]
    if args.ab:
        fields += ["email_draft_A", "email_draft_B"]
    else:
        fields += ["email_draft"]

    cta_link = args.cta_link or (profile.cta_link if profile else None) or os.getenv("CALENDLY_LINK")

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in companies:
            if args.ab:
                draft_a = agent.draft_email(
                    c,
                    args.offer or (profile.offer_default if profile else ""),
                    tone=(args.tone or (profile.tone.email if profile else "professionale")),
                    variant="A",
                    cta_link=cta_link,
                )
                draft_b = agent.draft_email(
                    c,
                    args.offer or (profile.offer_default if profile else ""),
                    tone=(args.tone or (profile.tone.email if profile else "professionale")),
                    variant="B",
                    cta_link=cta_link,
                )
                writer.writerow({"company": c, "email_draft_A": draft_a, "email_draft_B": draft_b})
            else:
                draft = agent.draft_email(
                    c,
                    args.offer or (profile.offer_default if profile else ""),
                    tone=(args.tone or (profile.tone.email if profile else "professionale")),
                    variant="A",
                    cta_link=cta_link,
                )
                writer.writerow({"company": c, "email_draft": draft})

    print(f"Generate {len(companies)} email in {out_csv}")


def cmd_linkedin(args: argparse.Namespace):
    profile = _load_profile(args.profile)
    client = get_client(PersonalAgent.system_prompt)
    agent = PersonalAgent(client=client, profile=profile)
    cta_link = args.cta_link or (profile.cta_link if profile else None) or os.getenv("CALENDLY_LINK")
    post = agent.post_linkedin(
        args.topic,
        tone=(args.tone or (profile.tone.linkedin if profile else "professionale")),
        audience=args.audience,
        cta_link=cta_link,
        length=args.length,
    )

    out_dir = Path(args.output_dir or (ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "linkedin_post.md"
    out_md.write_text(post)
    print(f"Post LinkedIn salvato in {out_md}")


def cmd_chat(args: argparse.Namespace):
    profile = _load_profile(args.profile)
    client = get_client(PersonalAgent.system_prompt)
    # Abilita tool-call e pianificazione – usa un client reale per sfruttarli
    agent = PersonalAgent(client=client, profile=profile)
    # Eseguo una singola run; per sessioni iterative si può avvolgere in un loop
    result = agent.run(args.prompt, tool_choice="auto")
    print(result.text if result else "")


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _find_linkedin_for_company(company: str) -> str:
    try:
        ddg = DuckDuckGoSearchTool()
        results = ddg.search(f"{company} linkedin company page")
        for r in results:
            href = r.get("href", "")
            if "linkedin.com" in href:
                return href
    except Exception:
        return ""
    return ""


def _email_candidates_from_domain(domain: str) -> List[str]:
    if not domain:
        return []
    base = ["info", "hello", "contact", "hi", "sales"]
    return [f"{b}@{domain}" for b in base]


def cmd_enrich_prospects(args: argparse.Namespace):
    src = Path(args.from_csv)
    rows = []
    with src.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    out_dir = Path(args.output_dir or (ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "prospects_enriched.csv"

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "company",
            "url",
            "domain",
            "linkedin_url",
            "email_candidates",
            "snippet",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            company = r.get("company") or r.get("title") or ""
            url = r.get("url") or r.get("href") or ""
            domain = _domain_from_url(url)
            linkedin = _find_linkedin_for_company(company)
            emails = ", ".join(_email_candidates_from_domain(domain))
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

    print(f"Prospects arricchiti salvati in {out_csv}")


def _smtp_send(from_addr: str, to_addr: str, subject: str, body: str):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    if not (host and user and pwd and from_addr):
        raise RuntimeError("Config SMTP mancante (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL)")
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context) as server:
        server.login(user, pwd)
        server.sendmail(from_addr, [to_addr], msg.as_string())


def cmd_send(args: argparse.Namespace):
    src = Path(args.from_csv)
    from_email = args.from_email or os.getenv("FROM_EMAIL")
    sent_log = Path(args.output_dir or (ROOT / "outputs")) / "sent_log.csv"
    sent_log.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src.open() as f, sent_log.open("w", newline="") as logf:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(logf, fieldnames=["company", "email", "status", "error"])
        writer.writeheader()
        for row in reader:
            email = row.get("email") or ""
            if not email:
                writer.writerow({"company": row.get("company", ""), "email": "", "status": "skipped", "error": "missing email"})
                continue
            body = row.get("email_draft") or row.get("email_draft_A") or row.get("email_draft_B") or ""
            try:
                _smtp_send(from_email or "", email, args.subject or "Proposta", body)
                writer.writerow({"company": row.get("company", ""), "email": email, "status": "sent", "error": ""})
                count += 1
            except Exception as e:
                writer.writerow({"company": row.get("company", ""), "email": email, "status": "error", "error": str(e)})

    print(f"Invii completati: {count}. Log: {sent_log}")


def cmd_campaign(args: argparse.Namespace):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir or (ROOT / "outputs")) / f"campaign-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) prospects
    a = argparse.Namespace(
        industry=args.industry,
        location=args.location,
        size=args.size,
        role=args.role,
        extra=args.extra,
        output_dir=str(out_dir),
    )
    cmd_prospects(a)

    # 2) enrich
    b = argparse.Namespace(
        from_csv=str(out_dir / "prospects.csv"),
        output_dir=str(out_dir),
    )
    cmd_enrich_prospects(b)

    # 3) emails (A/B opzionale)
    c = argparse.Namespace(
        offer=args.offer,
        from_csv=str(out_dir / "prospects_enriched.csv"),
        companies=None,
        output_dir=str(out_dir),
        ab=args.ab,
        tone=args.tone,
        cta_link=args.cta_link or os.getenv("CALENDLY_LINK"),
    )
    cmd_email(c)

    # 4) linkedin post
    d = argparse.Namespace(
        topic=args.post_topic,
        tone=args.post_tone,
        audience=args.post_audience,
        length=args.post_length,
        cta_link=args.cta_link or os.getenv("CALENDLY_LINK"),
        output_dir=str(out_dir),
    )
    cmd_linkedin(d)

    print(f"Campagna completata. Files in {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(prog="personal-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prospects", help="Cerca prospect e salva CSV")
    p1.add_argument("--industry", required=True, help="Settore es. 'Design'")
    p1.add_argument("--location", required=True, help="Località es. 'Milano'")
    p1.add_argument("--size", help="Dimensione (SMB, Mid, Enterprise)")
    p1.add_argument("--role", help="Ruolo target (es. Founder, Head of...)")
    p1.add_argument("--extra", help="Testo extra per la query")
    p1.add_argument("--output-dir", help="Cartella output")
    p1.set_defaults(func=cmd_prospects)

    p2 = sub.add_parser("email", help="Genera bozze email da lista/CSV")
    p2.add_argument("--offer", required=True, help="La tua offerta da promuovere")
    p2.add_argument("--from-csv", help="CSV con colonna 'title' o 'company'")
    p2.add_argument("--companies", nargs="*", help="Nomi aziende in linea")
    p2.add_argument("--tone", default="professionale", help="Tono email")
    p2.add_argument("--ab", action="store_true", help="Genera variante A/B")
    p2.add_argument("--cta-link", help="Link CTA prenotazione/call (di default CALENDLY_LINK)")
    p2.add_argument("--profile", help="Percorso profilo personale YAML")
    p2.add_argument("--output-dir", help="Cartella output")
    p2.set_defaults(func=cmd_email)

    p3 = sub.add_parser("linkedin-post", help="Genera un post LinkedIn")
    p3.add_argument("--topic", required=True, help="Argomento del post")
    p3.add_argument("--tone", default="professionale", help="Tono del post")
    p3.add_argument("--audience", default="clienti", help="Audience del post")
    p3.add_argument("--length", default="breve", help="Lunghezza: breve|medio|lungo")
    p3.add_argument("--cta-link", help="Link CTA prenotazione/call (di default CALENDLY_LINK)")
    p3.add_argument("--profile", help="Percorso profilo personale YAML")
    p3.add_argument("--output-dir", help="Cartella output")
    p3.set_defaults(func=cmd_linkedin)

    p4 = sub.add_parser("chat", help="Chatta con l'agente (tool auto)")
    p4.add_argument("--prompt", required=True, help="Prompt da inviare all'agente")
    p4.add_argument("--profile", help="Percorso profilo personale YAML")
    p4.set_defaults(func=cmd_chat)

    p5 = sub.add_parser("enrich-prospects", help="Arricchisci prospects con dominio, linkedin, email candidates")
    p5.add_argument("--from-csv", required=True, help="CSV di input (da prospects)")
    p5.add_argument("--output-dir", help="Cartella output")
    p5.set_defaults(func=cmd_enrich_prospects)

    p6 = sub.add_parser("send", help="Invia email da CSV via SMTP (opzionale)")
    p6.add_argument("--from-csv", required=True, help="CSV con colonne email_draft/email_draft_A/email_draft_B ed 'email'")
    p6.add_argument("--from-email", help="Mittente (default FROM_EMAIL)")
    p6.add_argument("--subject", help="Oggetto email (default 'Proposta')")
    p6.add_argument("--output-dir", help="Cartella output per log")
    p6.set_defaults(func=cmd_send)

    p7 = sub.add_parser("campaign", help="Esegue prospects -> enrich -> email -> linkedin-post")
    p7.add_argument("--industry", required=True)
    p7.add_argument("--location", required=True)
    p7.add_argument("--offer", required=True)
    p7.add_argument("--post-topic", required=True)
    p7.add_argument("--size")
    p7.add_argument("--role")
    p7.add_argument("--extra")
    p7.add_argument("--tone", default="professionale")
    p7.add_argument("--ab", action="store_true")
    p7.add_argument("--cta-link")
    p7.add_argument("--post-tone", default="professionale")
    p7.add_argument("--post-audience", default="clienti")
    p7.add_argument("--post-length", default="breve")
    p7.add_argument("--output-dir", help="Cartella base output")
    p7.set_defaults(func=cmd_campaign)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
