"""CLI per sistema Multi-Agent collaborativo.

Comandi:
  - write-critique-revise: Workflow completo per contenuti di alta qualità
  - email-pro: Email con feedback critico e revisione
  - post-pro: LinkedIn post con feedback critico e revisione
  - article-pro: Articolo con feedback critico e revisione

Uso:
  python tools/multi_agent_cli.py email-pro \
    --company "Museo MAXXI Roma" \
    --offer "sistema MOOD per exhibition" \
    --profile configs/personal_profile.yaml

  python tools/multi_agent_cli.py post-pro \
    --topic "MOOD AI per musei" \
    --length lungo \
    --profile configs/personal_profile.yaml
"""
from __future__ import annotations

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "datapizza-ai-core"
GOOGLE = ROOT / "datapizza-ai-clients" / "datapizza-ai-clients-google"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(GOOGLE))

from datapizza.agents.multi_agent import MultiAgentContentTeam
from datapizza.agents.personal_profile import PersonalProfile

# Try Google Gemini
try:
    from datapizza.clients.google.google_client import GoogleClient
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Fallback OpenAI
try:
    from datapizza.clients.openai.openai_client import OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_client():
    """Ritorna client LLM: Gemini > OpenAI > Error."""
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and GOOGLE_AVAILABLE:
        model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")
        try:
            return GoogleClient(api_key=google_key, model=model)
        except Exception as e:
            print(f"⚠️ Google client error: {e}")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and OPENAI_AVAILABLE:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIClient(api_key=openai_key, model=model)
    
    raise RuntimeError(
        "❌ Nessun client LLM disponibile! "
        "Imposta GOOGLE_API_KEY o OPENAI_API_KEY"
    )


def load_profile(profile_path: Optional[str]) -> Optional[PersonalProfile]:
    """Carica profilo personale."""
    p = profile_path or str(ROOT / "configs" / "personal_profile.yaml")
    try:
        return PersonalProfile.load(p)
    except Exception as e:
        print(f"⚠️ Errore caricamento profilo: {e}")
        return None


def save_result(result: dict, output_dir: Path, content_type: str):
    """Salva risultato in formato JSON e separati (draft, critique, final)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{content_type}_{timestamp}"
    
    # JSON completo
    json_file = output_dir / f"{prefix}_full.json"
    with json_file.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n📦 JSON completo: {json_file}")
    
    # Draft
    draft_file = output_dir / f"{prefix}_draft.md"
    with draft_file.open("w") as f:
        f.write(result["draft"])
    print(f"📝 Bozza: {draft_file}")
    
    # Critique
    critique_file = output_dir / f"{prefix}_critique.md"
    with critique_file.open("w") as f:
        f.write(result["critique"])
    print(f"🔍 Critica: {critique_file}")
    
    # Final
    final_file = output_dir / f"{prefix}_final.md"
    with final_file.open("w") as f:
        f.write(result["final"])
    print(f"✅ Versione finale: {final_file}")


def cmd_email_pro(args: argparse.Namespace):
    """Genera email con workflow Write → Critique → Revise."""
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT EMAIL WORKFLOW")
    print("="*70)
    
    client = get_client()
    profile = load_profile(args.profile)
    team = MultiAgentContentTeam(client=client, profile=profile)
    
    result = team.create_email(
        company_name=args.company,
        offer=args.offer,
        tone=args.tone
    )
    
    # Output directory
    out_dir = Path(args.output_dir or (ROOT / "outputs" / "multi_agent"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_result(result, out_dir, "email")
    
    print("\n" + "="*70)
    print("✅ EMAIL COMPLETATA")
    print("="*70)
    print(f"\n📧 Destinatario: {args.company}")
    print(f"🎯 Offerta: {args.offer}")
    print(f"\n--- VERSIONE FINALE ---\n")
    print(result["final"])


def cmd_post_pro(args: argparse.Namespace):
    """Genera LinkedIn post con workflow Write → Critique → Revise."""
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT LINKEDIN POST WORKFLOW")
    print("="*70)
    
    client = get_client()
    profile = load_profile(args.profile)
    team = MultiAgentContentTeam(client=client, profile=profile)
    
    result = team.create_linkedin_post(
        topic=args.topic,
        length=args.length,
        tone=args.tone,
        audience=args.audience
    )
    
    # Output directory
    out_dir = Path(args.output_dir or (ROOT / "outputs" / "multi_agent"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_result(result, out_dir, "linkedin_post")
    
    print("\n" + "="*70)
    print("✅ POST LINKEDIN COMPLETATO")
    print("="*70)
    print(f"\n📱 Topic: {args.topic}")
    print(f"👥 Audience: {args.audience}")
    print(f"\n--- VERSIONE FINALE ---\n")
    print(result["final"])


def cmd_article_pro(args: argparse.Namespace):
    """Genera articolo con workflow Write → Critique → Revise."""
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT ARTICLE WORKFLOW")
    print("="*70)
    
    client = get_client()
    profile = load_profile(args.profile)
    team = MultiAgentContentTeam(client=client, profile=profile)
    
    result = team.create_article(
        topic=args.topic,
        angle=args.angle,
        word_count=args.words
    )
    
    # Output directory
    out_dir = Path(args.output_dir or (ROOT / "outputs" / "multi_agent"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_result(result, out_dir, "article")
    
    print("\n" + "="*70)
    print("✅ ARTICOLO COMPLETATO")
    print("="*70)
    print(f"\n📰 Topic: {args.topic}")
    print(f"🎯 Angle: {args.angle}")
    print(f"📊 Parole: ~{args.words}")
    print(f"\n--- VERSIONE FINALE (primi 500 caratteri) ---\n")
    print(result["final"][:500] + "...")


def build_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        prog="multi-agent",
        description="Sistema Multi-Agent per contenuti di alta qualità"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # email-pro command
    p1 = sub.add_parser("email-pro", help="Email con workflow Write→Critique→Revise")
    p1.add_argument("--company", required=True, help="Nome azienda destinataria")
    p1.add_argument("--offer", required=True, help="Offerta da proporre")
    p1.add_argument("--tone", default="professionale", help="Tono email")
    p1.add_argument("--profile", help="Path profilo personale YAML")
    p1.add_argument("--output-dir", help="Cartella output")
    p1.set_defaults(func=cmd_email_pro)
    
    # post-pro command
    p2 = sub.add_parser("post-pro", help="LinkedIn post con workflow Write→Critique→Revise")
    p2.add_argument("--topic", required=True, help="Topic del post")
    p2.add_argument("--length", default="medio", choices=["breve", "medio", "lungo"], help="Lunghezza post")
    p2.add_argument("--tone", default="professionale", help="Tono del post")
    p2.add_argument("--audience", default="decision makers", help="Target audience")
    p2.add_argument("--profile", help="Path profilo personale YAML")
    p2.add_argument("--output-dir", help="Cartella output")
    p2.set_defaults(func=cmd_post_pro)
    
    # article-pro command
    p3 = sub.add_parser("article-pro", help="Articolo con workflow Write→Critique→Revise")
    p3.add_argument("--topic", required=True, help="Topic dell'articolo")
    p3.add_argument("--angle", required=True, help="Angolazione/prospettiva unica")
    p3.add_argument("--words", type=int, default=800, help="Target word count")
    p3.add_argument("--profile", help="Path profilo personale YAML")
    p3.add_argument("--output-dir", help="Cartella output")
    p3.set_defaults(func=cmd_article_pro)
    
    return p


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
