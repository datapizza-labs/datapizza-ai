"""
🤖 Autonomous Coordinator
==========================
Coordinatore intelligente che decide cosa fare autonomamente e cosa richiedere approvazione.

Livelli di autonomia:
- AUTONOMOUS: Esegue senza approvazione (es: aggiornare dipendenze, documentazione)
- APPROVAL: Richiede OK (es: modifiche codice, integrazioni hardware)
- CRITICAL: Richiede review dettagliato (es: security, breaking changes)

Autore: Antonio Mainenti
Data: 29 Ottobre 2025
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from google import genai


class ActionPriority(Enum):
    """Priorità delle azioni"""
    CRITICAL = "critical"      # Fare subito
    HIGH = "high"              # Entro 24h
    MEDIUM = "medium"          # Entro settimana
    LOW = "low"                # Quando possibile
    INFO = "info"              # Solo informativo


class AutonomyLevel(Enum):
    """Livello di autonomia richiesto"""
    AUTONOMOUS = "autonomous"   # Fa da solo
    APPROVAL = "approval"       # Chiede OK rapido
    CRITICAL = "critical"       # Richiede review


class AutonomousCoordinator:
    """
    Coordinatore che analizza ricerche e azioni, decide autonomia e genera sintesi executive.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY non trovata")
        
        self.client = genai.Client(api_key=self.api_key.strip())
        self.output_dir = Path("outputs/autonomous")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.action_log = self.output_dir / "action_log.json"
        self._load_action_log()
    
    def _load_action_log(self):
        """Carica log azioni precedenti"""
        if self.action_log.exists():
            with open(self.action_log, 'r', encoding='utf-8') as f:
                self.actions = json.load(f)
        else:
            self.actions = {
                "autonomous_actions": [],
                "pending_approvals": [],
                "completed_actions": []
            }
    
    def _save_action_log(self):
        """Salva log azioni"""
        with open(self.action_log, 'w', encoding='utf-8') as f:
            json.dump(self.actions, f, indent=2, ensure_ascii=False)
    
    def create_executive_digest(self, research_summary: Dict, dev_proposals: Optional[List[Dict]] = None) -> Dict:
        """
        Crea un digest ultra-compatto: 30 secondi di lettura, decisioni chiare.
        
        Args:
            research_summary: Output di WebResearchAgent
            dev_proposals: Output di MOODDeveloperAgent (opzionale)
            
        Returns:
            Dict con digest executive e azioni categorizzate per autonomia
        """
        # Combina input
        research_text = research_summary.get('executive_summary', '')
        findings_text = "\n\n".join([
            f"{f['topic']}: {f['findings'][:200]}..."
            for f in research_summary.get('findings', [])[:3]  # Solo top 3
        ])
        
        dev_text = ""
        if dev_proposals:
            dev_text = "\n\n".join([
                f"{p.get('title', 'Proposta')}: {p.get('summary', '')[:200]}..."
                for p in dev_proposals[:2]  # Solo top 2
            ])
        
        # Prompt per LLM: analizza e categorizza
        prompt = f"""You are an executive AI assistant. Analyze this research/development output and create an ULTRA-COMPACT digest.

**Research Summary:**
{research_text}

**Key Findings (top 3):**
{findings_text}

**Dev Proposals:**
{dev_text if dev_text else "None"}

Create a 30-SECOND READ digest with:

1. **🎯 Top Priority (1 action only)**
   - What's the single most important thing to do NOW?
   - Be ultra-specific (tool name, version, command if possible)

2. **⚡ Quick Wins (max 3)**
   - Low-effort, high-impact actions
   - Can be done autonomously (updating docs, installing libs, config changes)

3. **🔔 Needs Your Approval (max 2)**
   - Medium-risk actions requiring OK
   - Be specific: what exactly needs approval?

4. **📊 Just FYI (1 line)**
   - Background trends to be aware of, no action needed

5. **🤖 I'll Handle Autonomously**
   - What I can do myself without bothering you

For EACH action, classify:
- **Priority**: CRITICAL | HIGH | MEDIUM | LOW | INFO
- **Autonomy**: AUTONOMOUS (I do it) | APPROVAL (needs your OK) | CRITICAL (detailed review)
- **ETA**: "5 min" | "1 hour" | "1 day" | etc.

Keep it BRUTALLY concise. Max 150 words total.
Format in clear markdown with emojis.
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            digest_text = response.text
            
            # Parse e categorizza azioni (semplificato - in produzione userebbe regex/parsing)
            digest = {
                "timestamp": datetime.now().isoformat(),
                "read_time": "30 seconds",
                "digest_text": digest_text,
                "autonomous_actions": self._extract_autonomous_actions(digest_text),
                "approval_needed": self._extract_approval_actions(digest_text),
                "fyi_only": self._extract_fyi(digest_text)
            }
            
            # Salva digest
            digest_file = self.output_dir / f"executive_digest_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
            with open(digest_file, 'w', encoding='utf-8') as f:
                f.write(f"# 🎯 Executive Digest\n\n")
                f.write(f"**Generated:** {digest['timestamp']}\n\n")
                f.write(f"**Read Time:** {digest['read_time']}\n\n")
                f.write("---\n\n")
                f.write(digest_text)
            
            return digest
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_autonomous_actions(self, text: str) -> List[Dict]:
        """Estrae azioni che l'agente può fare da solo"""
        # Simplified - in produzione userebbe NLP/parsing più sofisticato
        actions = []
        
        # Cerca sezioni "I'll Handle Autonomously" o "Quick Wins"
        if "Quick Wins" in text or "I'll Handle" in text:
            actions.append({
                "id": f"auto_{datetime.now().timestamp()}",
                "description": "Actions extracted from digest",
                "autonomy": AutonomyLevel.AUTONOMOUS.value,
                "priority": ActionPriority.HIGH.value,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            })
        
        return actions
    
    def _extract_approval_actions(self, text: str) -> List[Dict]:
        """Estrae azioni che richiedono approvazione"""
        actions = []
        
        if "Needs Your Approval" in text or "APPROVAL" in text:
            actions.append({
                "id": f"approval_{datetime.now().timestamp()}",
                "description": "Actions needing approval from digest",
                "autonomy": AutonomyLevel.APPROVAL.value,
                "priority": ActionPriority.HIGH.value,
                "status": "pending_approval",
                "created_at": datetime.now().isoformat()
            })
        
        return actions
    
    def _extract_fyi(self, text: str) -> List[str]:
        """Estrae info FYI"""
        if "Just FYI" in text or "📊" in text:
            return ["Background trends extracted from digest"]
        return []
    
    def execute_autonomous_action(self, action_id: str) -> bool:
        """
        Esegue un'azione autonoma senza approvazione.
        
        Args:
            action_id: ID dell'azione da eseguire
            
        Returns:
            True se successo
        """
        # Trova azione
        action = None
        for a in self.actions["autonomous_actions"]:
            if a["id"] == action_id:
                action = a
                break
        
        if not action:
            return False
        
        # Esegui (placeholder - qui andrebbero le implementazioni reali)
        action["status"] = "completed"
        action["completed_at"] = datetime.now().isoformat()
        
        # Sposta in completed
        self.actions["completed_actions"].append(action)
        self.actions["autonomous_actions"] = [
            a for a in self.actions["autonomous_actions"] if a["id"] != action_id
        ]
        
        self._save_action_log()
        return True
    
    def request_approval(self, action_id: str) -> Dict:
        """
        Genera richiesta approvazione compatta.
        
        Returns:
            Dict con domanda Y/N e contesto minimo
        """
        action = None
        for a in self.actions["pending_approvals"]:
            if a["id"] == action_id:
                action = a
                break
        
        if not action:
            return {"error": "Action not found"}
        
        return {
            "action_id": action_id,
            "question": f"OK per: {action['description']}?",
            "quick_context": action.get('context', 'No details'),
            "eta": action.get('eta', 'Unknown'),
            "risk": action.get('priority', 'medium')
        }
    
    def approve_action(self, action_id: str, approved: bool = True) -> bool:
        """Approva o rifiuta un'azione"""
        for action in self.actions["pending_approvals"]:
            if action["id"] == action_id:
                if approved:
                    action["status"] = "approved"
                    action["approved_at"] = datetime.now().isoformat()
                    # Sposta in autonomous per esecuzione
                    self.actions["autonomous_actions"].append(action)
                else:
                    action["status"] = "rejected"
                    action["rejected_at"] = datetime.now().isoformat()
                    self.actions["completed_actions"].append(action)
                
                self.actions["pending_approvals"] = [
                    a for a in self.actions["pending_approvals"] if a["id"] != action_id
                ]
                
                self._save_action_log()
                return True
        
        return False
    
    def get_pending_approvals(self) -> List[Dict]:
        """Ritorna azioni in attesa di approvazione"""
        return self.actions["pending_approvals"]
    
    def get_status_summary(self) -> Dict:
        """Ritorna sommario stato azioni"""
        return {
            "autonomous_pending": len(self.actions["autonomous_actions"]),
            "awaiting_approval": len(self.actions["pending_approvals"]),
            "completed_today": len([
                a for a in self.actions["completed_actions"]
                if datetime.fromisoformat(a.get("completed_at", "2000-01-01")).date() == datetime.now().date()
            ]),
            "total_completed": len(self.actions["completed_actions"])
        }


def main():
    """Test del coordinator"""
    coordinator = AutonomousCoordinator()
    
    # Esempio: digest da ricerca
    mock_research = {
        "executive_summary": "Top innovations: Multi-agent frameworks, Edge AI, LLM orchestration",
        "findings": [
            {"topic": "AI agents frameworks 2025", "findings": "PrimisAI Nexus enables secure Python execution in multi-agent systems..."},
            {"topic": "Edge AI", "findings": "Jetson Orin Nano provides low-latency inference..."}
        ]
    }
    
    digest = coordinator.create_executive_digest(mock_research)
    
    print("=" * 60)
    print("EXECUTIVE DIGEST")
    print("=" * 60)
    print(digest.get("digest_text", "Error generating digest"))
    print("\n" + "=" * 60)
    print("STATUS")
    print("=" * 60)
    print(coordinator.get_status_summary())


if __name__ == "__main__":
    main()
