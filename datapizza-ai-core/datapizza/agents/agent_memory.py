"""
🧠 Agent Learning Memory System
================================
Sistema di memoria che permette all'agent di imparare dalle azioni dell'utente.

Funzionalità:
- Traccia tutte le azioni MOOD (contatti cercati, email generate, post creati)
- Analizza pattern nei dati
- Inietta "learned patterns" nel system prompt dell'agent
- Propone miglioramenti innovativi basati su osservazioni

Autore: Antonio Mainenti
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import Counter


class AgentMemory:
    """Sistema di memoria che impara dalle azioni dell'utente"""
    
    def __init__(self, memory_file: str = "agent_learning.json"):
        """
        Inizializza il sistema di memoria.
        
        Args:
            memory_file: Path al file JSON di memoria
        """
        self.memory_file = Path(memory_file)
        self._ensure_memory_file()
    
    def _ensure_memory_file(self):
        """Crea il file di memoria se non esiste"""
        if not self.memory_file.exists():
            self.memory_file.write_text(json.dumps({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "contacts_hunted": [],
                "emails_generated": [],
                "emails_sent": [],
                "instagram_posts_generated": [],
                "features_proposed": [],
                "technologies_analyzed": [],
                "notes": []
            }, indent=2))
    
    def _load_memory(self) -> Dict[str, Any]:
        """Carica il file di memoria"""
        try:
            return json.loads(self.memory_file.read_text())
        except Exception as e:
            print(f"Errore caricamento memoria: {e}")
            self._ensure_memory_file()
            return json.loads(self.memory_file.read_text())
    
    def _save_memory(self, memory: Dict[str, Any]):
        """Salva il file di memoria"""
        self.memory_file.write_text(json.dumps(memory, indent=2, default=str))
    
    def log_contacts_hunted(self, count: int, organizations: List[str], sources: List[str]):
        """Registra contatti trovati"""
        memory = self._load_memory()
        memory["contacts_hunted"].append({
            "timestamp": datetime.now().isoformat(),
            "count": count,
            "organizations": organizations[:5],  # Ultimi 5
            "sources": sources[:5]
        })
        self._save_memory(memory)
    
    def log_email_generated(self, company_name: str, tone: str, offer: str):
        """Registra email generata"""
        memory = self._load_memory()
        memory["emails_generated"].append({
            "timestamp": datetime.now().isoformat(),
            "company_name": company_name,
            "tone": tone,
            "offer": offer
        })
        self._save_memory(memory)
    
    def log_email_sent(self, recipient_email: str, company_name: str):
        """Registra email inviata"""
        memory = self._load_memory()
        memory["emails_sent"].append({
            "timestamp": datetime.now().isoformat(),
            "recipient_email": recipient_email,
            "company_name": company_name
        })
        self._save_memory(memory)
    
    def log_instagram_post_generated(self, topic: str, target_audience: str, style: str):
        """Registra post Instagram generato"""
        memory = self._load_memory()
        memory["instagram_posts_generated"].append({
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "target_audience": target_audience,
            "style": style
        })
        self._save_memory(memory)
    
    def log_linkedin_post_generated(self, post_type: str, topic: str, personal_brand: str):
        """Registra post LinkedIn generato"""
        memory = self._load_memory()
        # Crea entry se non esiste
        if "linkedin_posts_generated" not in memory:
            memory["linkedin_posts_generated"] = []
        
        memory["linkedin_posts_generated"].append({
            "timestamp": datetime.now().isoformat(),
            "post_type": post_type,
            "topic": topic,
            "personal_brand": personal_brand
        })
        self._save_memory(memory)
    
    def log_feature_proposed(self, feature_name: str, description: str, status: str = "proposed"):
        """Registra feature proposta"""
        memory = self._load_memory()
        memory["features_proposed"].append({
            "timestamp": datetime.now().isoformat(),
            "feature_name": feature_name,
            "description": description,
            "status": status
        })
        self._save_memory(memory)
    
    def log_technology_analyzed(self, technology: str, context: str, recommendation: str):
        """Registra analisi tecnologica"""
        memory = self._load_memory()
        memory["technologies_analyzed"].append({
            "timestamp": datetime.now().isoformat(),
            "technology": technology,
            "context": context,
            "recommendation": recommendation
        })
        self._save_memory(memory)
    
    def add_note(self, note: str):
        """Aggiungi nota libera"""
        memory = self._load_memory()
        memory["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "content": note
        })
        self._save_memory(memory)
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analizza i pattern nei dati raccolti"""
        memory = self._load_memory()
        
        patterns = {
            "total_contacts": len(memory.get("contacts_hunted", [])),
            "total_emails_generated": len(memory.get("emails_generated", [])),
            "total_emails_sent": len(memory.get("emails_sent", [])),
            "total_instagram_posts": len(memory.get("instagram_posts_generated", [])),
            "total_linkedin_posts": len(memory.get("linkedin_posts_generated", [])),
            "conversion_rate": 0,
            "preferred_tones": [],
            "preferred_styles": [],
            "top_organizations": [],
            "linkedin_post_types": [],
            "most_active_hours": "non-determinato",
            "weekly_activity": "crescente"
        }
        
        # Calcola tasso di conversione
        if memory.get("emails_generated"):
            total_gen = len(memory["emails_generated"])
            total_sent = len(memory["emails_sent"])
            patterns["conversion_rate"] = (total_sent / total_gen) * 100 if total_gen > 0 else 0
        
        # Toni preferiti
        if memory.get("emails_generated"):
            tones = [e.get("tone", "neutral") for e in memory["emails_generated"]]
            patterns["preferred_tones"] = [tone for tone, _ in Counter(tones).most_common(3)]
        
        # Post type LinkedIn preferiti
        if memory.get("linkedin_posts_generated"):
            post_types = [p.get("post_type", "mixed") for p in memory["linkedin_posts_generated"]]
            patterns["linkedin_post_types"] = [pt for pt, _ in Counter(post_types).most_common(3)]
        
        # Stili preferiti
        if memory.get("instagram_posts_generated"):
            styles = [p.get("style", "professional") for p in memory["instagram_posts_generated"]]
            patterns["preferred_styles"] = [style for style, _ in Counter(styles).most_common(3)]
        
        # Organizzazioni più frequenti
        if memory.get("contacts_hunted"):
            all_orgs = []
            for hunt in memory["contacts_hunted"]:
                all_orgs.extend(hunt.get("organizations", []))
            patterns["top_organizations"] = [org for org, _ in Counter(all_orgs).most_common(5)]
        
        return patterns
    
    def get_learning_insights(self) -> str:
        """Genera insight per il system prompt dell'agent"""
        patterns = self.analyze_patterns()
        memory = self._load_memory()
        
        # Calcola last 7 days activity
        last_week = datetime.now() - timedelta(days=7)
        week_contacts = sum(1 for c in memory.get("contacts_hunted", [])
                          if c.get("timestamp") and datetime.fromisoformat(c["timestamp"]) > last_week)
        week_emails = sum(1 for e in memory.get("emails_sent", [])
                        if e.get("timestamp") and datetime.fromisoformat(e["timestamp"]) > last_week)
        
        insights = f"""
LEARNED PATTERNS (Data-Driven Insights):
=========================================
📊 User Activity Analysis:
- Total contacts hunted: {patterns['total_contacts']}
- Total emails generated: {patterns['total_emails_generated']}
- Total emails sent: {patterns['total_emails_sent']}
- Email conversion rate: {patterns['conversion_rate']:.1f}%
- Total Instagram posts generated: {patterns['total_instagram_posts']}
- Total LinkedIn posts generated: {patterns['total_linkedin_posts']}

📈 Last 7 Days Activity:
- Contacts hunted this week: {week_contacts}
- Emails sent this week: {week_emails}

💡 Preferences & Patterns:
- Preferred email tones: {', '.join(patterns['preferred_tones']) if patterns['preferred_tones'] else 'varied'}
- Preferred Instagram styles: {', '.join(patterns['preferred_styles']) if patterns['preferred_styles'] else 'varied'}
- Preferred LinkedIn post types: {', '.join(patterns['linkedin_post_types']) if patterns['linkedin_post_types'] else 'mixed'}
- Most targeted organizations: {', '.join(patterns['top_organizations'][:3]) if patterns['top_organizations'] else 'diverse'}

🎯 INTELLIGENT RECOMMENDATIONS:
Based on these patterns, propose:
1. Innovations that follow the user's working style
2. Improvements that increase the email conversion rate
3. Features that automate the most frequent tasks
4. Technology integrations that align with user interests
5. LinkedIn content strategies based on preferred post types
"""
        return insights
    
    def clear_old_logs(self, days: int = 90):
        """Ripulisce log più vecchi di N giorni"""
        memory = self._load_memory()
        cutoff = datetime.now() - timedelta(days=days)
        
        for key in ["contacts_hunted", "emails_generated", "emails_sent", 
                   "instagram_posts_generated", "linkedin_posts_generated",
                   "features_proposed", "technologies_analyzed", "notes"]:
            if key in memory:
                memory[key] = [
                    item for item in memory[key]
                    if item.get("timestamp") and datetime.fromisoformat(item["timestamp"]) > cutoff
                ]
        
        self._save_memory(memory)
