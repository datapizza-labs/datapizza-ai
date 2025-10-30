"""Runner semplice per il `PersonalAgent`.

Esegue alcune azioni dimostrative usando `MockClient`.
"""
import sys
from pathlib import Path

# Assicura che la root del workspace sia nel PYTHONPATH quando esegui questo script
ROOT = Path(__file__).resolve().parents[1]
# Aggiungi la cartella datapizza-ai-core al PYTHONPATH per permettere import come `datapizza`.
CORE = ROOT / "datapizza-ai-core"
sys.path.insert(0, str(CORE))

from datapizza.agents.personal_agent import PersonalAgent


def main():
    agent = PersonalAgent()

    print("=== Esempio: trova target ===")
    targets = agent.find_targets("Design", size="SMB")
    print(targets)

    print("\n=== Esempio: bozza email ===")
    email = agent.draft_email("ACME Corp", "soluzioni di design conversazionale")
    print(email)

    print("\n=== Esempio: post LinkedIn ===")
    post = agent.post_linkedin("il mio nuovo servizio di design")
    print(post)


if __name__ == "__main__":
    main()
