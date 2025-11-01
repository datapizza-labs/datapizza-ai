"""
🔍 Web Research Agent
=====================
Agente autonomo per ricerca periodica di novità tecnologiche, trend e opportunità.

Funzionalità:
- Ricerca automatica su web di tecnologie emergenti
- Monitoring di GitHub trends e repository rilevanti
- Analisi news AI/ML/Interactive Art
- Identificazione opportunità di integrazione
- Report digest periodici

Autore: Antonio Mainenti
Data: 29 Ottobre 2025
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from google import genai
import requests

# Import DuckDuckGo search per ricerche web reali
try:
    from ddgs import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print("⚠️  DuckDuckGo non disponibile. Installare con: pip install ddgs")


class WebResearchAgent:
    """
    Agente per ricerca automatica periodica di novità tecnologiche.
    
    Strategia:
    1. Usa DuckDuckGo search tool per query mirate
    2. Analizza risultati con LLM per rilevanza
    3. Genera report digest con insights
    4. Salva storico ricerche per pattern analysis
    """
    
    RESEARCH_TOPICS = [
        "AI agents frameworks 2025",
        "multi-agent systems Python",
        "LLM orchestration tools",
        "interactive art AI technology",
        "generative AI new releases",
        "edge computing AI inference",
        "real-time computer vision 2025",
        "OSC protocol innovations",
        "Python AI libraries trending"
    ]
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "outputs/research", brave_api_key: Optional[str] = None):
        """
        Inizializza Web Research Agent.
        
        Args:
            api_key: Google API key (default: da env GOOGLE_API_KEY)
            output_dir: Directory per salvare report
            brave_api_key: Brave Search API key (default: da env BRAVE_API_KEY) - 2000 query/mese gratis
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY non trovata")
        
        # Pulisci la chiave da eventuali spazi o newline
        self.api_key = self.api_key.strip()
        
        self.client = genai.Client(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Brave Search API (opzionale, fallback su DuckDuckGo)
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.brave_available = bool(self.brave_api_key)
        
        self.history_file = self.output_dir / "research_history.json"
        self._load_history()
    
    def _load_history(self):
        """Carica storico ricerche precedenti"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "last_research": None,
                "total_researches": 0,
                "topics_covered": [],
                "findings": []
            }
    
    def _save_history(self):
        """Salva storico ricerche"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def should_run_research(self) -> bool:
        """
        Verifica se è il momento di eseguire una ricerca.
        Criteri: 3 volte a settimana (lunedì, mercoledì, venerdì)
        """
        if not self.history.get("last_research"):
            return True
        
        last_research = datetime.fromisoformat(self.history["last_research"])
        now = datetime.now()
        days_since = (now - last_research).days
        
        # Se sono passati più di 2 giorni, esegui ricerca
        if days_since >= 2:
            return True
        
        # Oppure verifica se oggi è un giorno di ricerca e non l'hai già fatto
        today_weekday = now.weekday()  # 0=lunedì, 2=mercoledì, 4=venerdì
        research_days = [0, 2, 4]
        
        if today_weekday in research_days and last_research.date() != now.date():
            return True
        
        return False
    
    def search_topic(self, topic: str) -> str:
        """
        Ricerca web reale per un topic usando provider multipli con fallback:
        1. Brave Search API (2000 query/mese gratis, risultati migliori)
        2. DuckDuckGo (illimitato, fallback)
        
        Args:
            topic: Topic da ricercare
            
        Returns:
            Sintesi risultati ricerca
        """
        search_results = ""
        provider_used = "none"
        
        # Prova prima con Brave Search (se disponibile)
        if self.brave_available:
            try:
                search_results, provider_used = self._search_brave(topic)
            except Exception as e:
                print(f"⚠️  Brave Search fallito ({e}), fallback su DuckDuckGo")
        
        # Fallback su DuckDuckGo se Brave non disponibile o fallito
        if not search_results and DUCKDUCKGO_AVAILABLE:
            try:
                search_results, provider_used = self._search_duckduckgo(topic)
            except Exception as e:
                print(f"⚠️  DuckDuckGo Search fallito: {e}")
                search_results = ""
        
        # Analizza i risultati con LLM
        if search_results:
            prompt = f"""You are a tech research assistant analyzing web search results.

Topic: {topic}

Search Results (via {provider_used}):
{search_results}

Based on these real-time search results, provide:
1. **Latest Developments** (2-3 key innovations or releases mentioned)
2. **Emerging Trends** (what's gaining traction based on the sources)
3. **Integration Opportunities** (how these could be applied to multi-agent AI systems)
4. **Key Resources** (specific tools, repos, or frameworks mentioned with links)

Keep concise (max 200 words) and cite sources when possible.
Format in markdown.
"""
        else:
            # Fallback finale: usa solo conoscenza del modello se search non disponibile
            provider_used = "LLM knowledge only"
            prompt = f"""You are a tech research assistant monitoring the latest developments in AI and interactive technology.

Research Topic: {topic}

Based on your knowledge and recent trends, provide:
1. **Latest Developments** (2-3 key innovations or releases in the last 3 months)
2. **Emerging Trends** (what's gaining traction in this space)
3. **Integration Opportunities** (how these could be applied to multi-agent AI systems or interactive art projects)
4. **Key Resources** (notable GitHub repos, tools, or frameworks)

Keep the response concise (max 200 words) and focused on actionable insights.
Format in markdown.
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            result = response.text
            # Aggiungi metadata provider
            result = f"*[Searched via: {provider_used}]*\n\n{result}"
            return result
        except Exception as e:
            return f"Error researching {topic}: {str(e)}"
    
    def _search_brave(self, query: str, max_results: int = 5) -> tuple[str, str]:
        """
        Cerca con Brave Search API.
        
        Args:
            query: Query di ricerca
            max_results: Numero massimo risultati
            
        Returns:
            Tuple (risultati_formattati, provider_name)
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {
            "q": query,
            "count": max_results,
            "freshness": "pw"  # Past week - risultati recenti
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("web", {}).get("results", [])
        
        if not results:
            return "", "Brave Search (no results)"
        
        formatted = "\n\n".join([
            f"**{r.get('title', 'No title')}**\n{r.get('description', 'No description')}\nSource: {r.get('url', '')}\nPublished: {r.get('age', 'Unknown')}"
            for r in results
        ])
        
        return formatted, "Brave Search"
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> tuple[str, str]:
        """
        Cerca con DuckDuckGo.
        
        Args:
            query: Query di ricerca
            max_results: Numero massimo risultati
            
        Returns:
            Tuple (risultati_formattati, provider_name)
        """
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
            if not results:
                return "", "DuckDuckGo (no results)"
            
            formatted = "\n\n".join([
                f"**{r['title']}**\n{r['body']}\nSource: {r['href']}"
                for r in results
            ])
            
            return formatted, "DuckDuckGo"
    
    def run_research_cycle(self) -> Dict:
        """
        Esegue un ciclo completo di ricerca su tutti i topic.
        
        Returns:
            Dict con risultati della ricerca
        """
        print("🔍 Avvio ciclo di ricerca...")
        timestamp = datetime.now().isoformat()
        
        findings = []
        for i, topic in enumerate(self.RESEARCH_TOPICS, 1):
            print(f"  [{i}/{len(self.RESEARCH_TOPICS)}] Ricerca: {topic}")
            result = self.search_topic(topic)
            findings.append({
                "topic": topic,
                "timestamp": timestamp,
                "findings": result
            })
        
        # Genera report digest
        digest = self._generate_digest(findings, timestamp)
        
        # Salva risultati
        self._save_research_results(digest, findings, timestamp)
        
        # Aggiorna history
        self.history["last_research"] = timestamp
        self.history["total_researches"] += 1
        self.history["topics_covered"].extend(self.RESEARCH_TOPICS)
        self.history["findings"].append({
            "timestamp": timestamp,
            "topics_count": len(self.RESEARCH_TOPICS),
            "digest_file": f"research_digest_{timestamp[:10]}.md"
        })
        self._save_history()
        
        print(f"✅ Ricerca completata. Report salvato in {self.output_dir}")
        return digest
    
    def _generate_digest(self, findings: List[Dict], timestamp: str) -> Dict:
        """
        Genera digest sintetico dei risultati usando LLM.
        
        Args:
            findings: Lista di risultati ricerca
            timestamp: Timestamp ricerca
            
        Returns:
            Dict con digest e insights
        """
        # Combina tutti i findings
        combined_findings = "\n\n".join([
            f"### {f['topic']}\n{f['findings']}"
            for f in findings
        ])
        
        prompt = f"""You are synthesizing research findings from multiple tech topics.

Here are the findings:

{combined_findings}

Create a concise executive summary with:
1. **Top 3 Most Promising Innovations** (across all topics)
2. **Recommended Actions** (2-3 concrete next steps)
3. **Priority Integrations** (what to implement first in a multi-agent AI system)

Keep it actionable and specific. Max 300 words.
Format in markdown.
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            return {
                "timestamp": timestamp,
                "executive_summary": response.text,
                "topics_researched": len(findings),
                "findings": findings
            }
        except Exception as e:
            return {
                "timestamp": timestamp,
                "executive_summary": f"Error generating digest: {str(e)}",
                "topics_researched": len(findings),
                "findings": findings
            }
    
    def _save_research_results(self, digest: Dict, findings: List[Dict], timestamp: str):
        """Salva risultati ricerca in file markdown e JSON"""
        date_str = timestamp[:10]
        
        # Salva digest markdown
        digest_file = self.output_dir / f"research_digest_{date_str}.md"
        with open(digest_file, 'w', encoding='utf-8') as f:
            f.write(f"# 🔍 Research Digest - {date_str}\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            f.write(f"**Topics Researched:** {len(findings)}\n\n")
            f.write("---\n\n")
            f.write("## Executive Summary\n\n")
            f.write(digest["executive_summary"])
            f.write("\n\n---\n\n")
            f.write("## Detailed Findings\n\n")
            for finding in findings:
                f.write(f"### {finding['topic']}\n\n")
                f.write(finding['findings'])
                f.write("\n\n")
        
        # Salva JSON completo
        json_file = self.output_dir / f"research_full_{date_str}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(digest, f, indent=2, ensure_ascii=False)
    
    def get_latest_insights(self, days: int = 7) -> List[Dict]:
        """
        Recupera insights delle ultime ricerche.
        
        Args:
            days: Numero di giorni da considerare
            
        Returns:
            Lista di findings recenti
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        
        for finding in self.history.get("findings", []):
            if datetime.fromisoformat(finding["timestamp"]) > cutoff:
                recent.append(finding)
        
        return recent
    
    def get_latest_research_summary(self) -> Optional[Dict]:
        """
        Recupera un sommario dell'ultima ricerca per visualizzazione dashboard.
        
        Returns:
            Dict con info ultima ricerca o None se non ci sono ricerche
        """
        if not self.history.get("findings"):
            return None
        
        latest = self.history["findings"][-1]
        timestamp = datetime.fromisoformat(latest["timestamp"])
        
        # Carica il digest completo
        digest_file = self.output_dir / latest["digest_file"].replace(".md", ".json").replace("research_digest", "research_full")
        
        if digest_file.exists():
            with open(digest_file, 'r', encoding='utf-8') as f:
                digest_data = json.load(f)
                
            return {
                "timestamp": timestamp,
                "date": timestamp.strftime("%d %B %Y"),
                "time": timestamp.strftime("%H:%M"),
                "topics_count": latest["topics_count"],
                "executive_summary": digest_data.get("executive_summary", ""),
                "findings": digest_data.get("findings", [])
            }
        
        return {
            "timestamp": timestamp,
            "date": timestamp.strftime("%d %B %Y"),
            "time": timestamp.strftime("%H:%M"),
            "topics_count": latest["topics_count"],
            "executive_summary": "Report non disponibile",
            "findings": []
        }


def main():
    """Entry point per esecuzione manuale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("--force", action="store_true", help="Forza ricerca anche se non è il giorno schedulato")
    parser.add_argument("--output", type=str, default="outputs/research", help="Directory output")
    
    args = parser.parse_args()
    
    agent = WebResearchAgent(output_dir=args.output)
    
    if args.force or agent.should_run_research():
        print("🚀 Esecuzione ricerca schedulata...")
        digest = agent.run_research_cycle()
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        print(digest["executive_summary"])
    else:
        print("⏸️  Non è ancora il momento per una nuova ricerca.")
        print(f"Ultima ricerca: {agent.history.get('last_research', 'mai')}")
        print("Prossima ricerca: lunedì, mercoledì o venerdì")


if __name__ == "__main__":
    main()
