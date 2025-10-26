"""
🔍 Contact Hunter Agent
======================
Agente autonomo che cerca contatti email da siti web di musei, gallerie, festival.

Funzionalità:
- Scraping intelligente di pagine web
- Estrazione email, nomi, ruoli
- Identificazione decision maker (curatori, direttori)
- Validazione email
- Arricchimento dati (LinkedIn, about pages)

Autore: Antonio Mainenti
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time

@dataclass
class Contact:
    """Rappresenta un contatto trovato"""
    email: str
    name: Optional[str] = None
    role: Optional[str] = None
    organization: str = ""
    source_url: str = ""
    confidence: float = 0.0  # 0-1, quanto siamo sicuri sia un decision maker
    
    def to_dict(self):
        return {
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "organization": self.organization,
            "source_url": self.source_url,
            "confidence": self.confidence
        }


class ContactHunterAgent:
    """
    Agente che cerca autonomamente contatti email da siti web.
    
    Strategia:
    1. Cerca pagine chiave: /contatti, /about, /staff, /team
    2. Estrae email da HTML
    3. Identifica nomi vicino alle email
    4. Deduce ruoli da parole chiave (curatore, direttore, etc)
    5. Assegna confidence score
    """
    
    # Parole chiave per identificare decision maker
    DECISION_MAKER_KEYWORDS = [
        'director', 'direttore', 'direttrice',
        'curator', 'curatore', 'curatrice',
        'president', 'presidente',
        'ceo', 'founder', 'co-founder',
        'artistic', 'artistico',
        'general manager', 'responsabile'
    ]
    
    # Domini email comuni (personali, non aziendali)
    COMMON_PERSONAL_DOMAINS = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 
        'outlook.com', 'icloud.com', 'protonmail.com'
    ]
    
    # Pattern per trovare email
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Parole chiave per decision maker (alta priorità)
    DECISION_MAKER_KEYWORDS = [
        "direttore", "director", "direttrice",
        "curatore", "curator", "curatrice",
        "responsabile", "head", "chief",
        "presidente", "president",
        "amministratore", "ceo",
        "artistic director", "direttore artistico"
    ]
    
    # Ruoli secondari (media priorità)
    SECONDARY_ROLES = [
        "ufficio stampa", "press office",
        "comunicazione", "communication",
        "marketing",
        "eventi", "events",
        "coordinatore", "coordinator"
    ]
    
    # Email generiche da filtrare (bassa priorità)
    GENERIC_EMAILS = [
        "info@", "contact@", "reception@", 
        "booking@", "prenotazioni@", "biglietteria@",
        "amministrazione@", "amministrativo@"
    ]
    
    # Pagine dove cercare
    CONTACT_PAGES = [
        "/contatti", "/contact", "/contacts",
        "/chi-siamo", "/about", "/about-us",
        "/staff", "/team", "/people",
        "/governance", "/direzione",
        "/ufficio-stampa", "/press"
    ]
    
    def __init__(self, delay: float = 2.0):
        """
        Args:
            delay: Secondi di attesa tra richieste (rispetto per i server)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) MOOD Contact Research Bot'
        })
    
    def hunt_contacts(self, base_url: str, organization_name: str) -> List[Contact]:
        """
        Cerca contatti su un sito web.
        
        Args:
            base_url: URL base (es. https://www.maxxi.art)
            organization_name: Nome organizzazione (es. "Museo MAXXI")
            
        Returns:
            Lista di contatti trovati, ordinati per confidence
        """
        print(f"\n🔍 Hunting contacts for: {organization_name}")
        print(f"   Base URL: {base_url}")
        
        all_contacts = []
        pages_to_check = [base_url] + [urljoin(base_url, page) for page in self.CONTACT_PAGES]
        
        for url in pages_to_check:
            try:
                contacts = self._extract_contacts_from_page(url, organization_name)
                all_contacts.extend(contacts)
                print(f"   ✓ {url}: Found {len(contacts)} contacts")
                
                time.sleep(self.delay)  # Rispetto per il server
                
            except Exception as e:
                print(f"   ✗ {url}: Error - {e}")
                continue
        
        # Deduplica e ordina per confidence
        unique_contacts = self._deduplicate_contacts(all_contacts)
        sorted_contacts = sorted(unique_contacts, key=lambda c: c.confidence, reverse=True)
        
        print(f"\n✅ Found {len(sorted_contacts)} unique contacts")
        return sorted_contacts
    
    def _extract_contacts_from_page(self, url: str, organization: str) -> List[Contact]:
        """Estrae contatti da una singola pagina"""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        contacts = []
        
        # Trova tutte le email nella pagina
        page_text = soup.get_text()
        emails = re.findall(self.EMAIL_PATTERN, page_text)
        
        for email in set(emails):  # Set per evitare duplicati
            # Cerca contesto intorno all'email (nome e ruolo)
            context = self._find_context_for_email(soup, email)
            
            contact = Contact(
                email=email.lower(),
                name=context.get("name"),
                role=context.get("role"),
                organization=organization,
                source_url=url,
                confidence=self._calculate_confidence(email, context.get("role"))
            )
            
            contacts.append(contact)
        
        return contacts
    
    def _find_context_for_email(self, soup: BeautifulSoup, email: str) -> Dict:
        """Trova nome e ruolo vicino all'email"""
        context = {"name": None, "role": None}
        
        # Trova l'elemento che contiene l'email
        for element in soup.find_all(string=re.compile(re.escape(email))):
            parent = element.parent
            
            # Cerca nome e ruolo nei dintorni
            # Strategia: guarda parent, siblings, e container
            text_chunks = []
            
            # Parent e suoi siblings
            if parent:
                text_chunks.append(parent.get_text())
                for sibling in parent.find_previous_siblings(limit=3):
                    text_chunks.append(sibling.get_text())
                for sibling in parent.find_next_siblings(limit=3):
                    text_chunks.append(sibling.get_text())
            
            # Cerca nome (parole con maiuscola)
            combined_text = " ".join(text_chunks)
            name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
            names = re.findall(name_pattern, combined_text)
            if names:
                context["name"] = names[0]
            
            # Cerca ruolo
            for keyword in self.DECISION_MAKER_KEYWORDS + self.SECONDARY_ROLES:
                if keyword.lower() in combined_text.lower():
                    context["role"] = keyword
                    break
        
        return context
    
    def _calculate_confidence(self, email: str, role: Optional[str]) -> float:
        """
        Calcola confidence score 0-1 che il contatto sia rilevante.
        
        Logica:
        - Email generica (info@): 0.2
        - Email specifica senza ruolo: 0.5
        - Email con ruolo secondario: 0.7
        - Email con ruolo decision maker: 0.9
        """
        score = 0.5  # Base
        
        # Penalizza email generiche
        if any(email.startswith(generic) for generic in self.GENERIC_EMAILS):
            score = 0.2
        
        # Bonus per ruolo identificato
        if role:
            role_lower = role.lower()
            if any(kw in role_lower for kw in self.DECISION_MAKER_KEYWORDS):
                score = 0.9
            elif any(kw in role_lower for kw in self.SECONDARY_ROLES):
                score = 0.7
        
        # Bonus per email nominativa (nome.cognome@)
        if re.match(r'^[a-z]+\.[a-z]+@', email):
            score += 0.1
        
        return min(score, 1.0)
    
    def _deduplicate_contacts(self, contacts: List[Contact]) -> List[Contact]:
        """Rimuove duplicati, tenendo quello con confidence più alta"""
        seen = {}
        
        for contact in contacts:
            email = contact.email
            if email not in seen or contact.confidence > seen[email].confidence:
                seen[email] = contact
        
        return list(seen.values())
    
    def validate_email(self, email: str) -> bool:
        """
        Validazione basilare email (senza invio).
        Per validazione completa serve servizio esterno.
        """
        # Pattern regex
        if not re.match(self.EMAIL_PATTERN, email):
            return False
        
        # Domain non deve essere comune (gmail, yahoo, etc per organizzazioni)
        domain = email.split('@')[1]
        
        if domain in self.COMMON_PERSONAL_DOMAINS:
            return False  # Probabile email personale, non aziendale
        
        return True


# Esempio di utilizzo
if __name__ == "__main__":
    hunter = ContactHunterAgent()
    
    # Test con Museo MAXXI
    contacts = hunter.hunt_contacts(
        base_url="https://www.maxxi.art",
        organization_name="Museo MAXXI Roma"
    )
    
    print("\n📊 Results:")
    print("=" * 60)
    
    for i, contact in enumerate(contacts[:10], 1):  # Top 10
        print(f"\n{i}. {contact.email}")
        if contact.name:
            print(f"   Name: {contact.name}")
        if contact.role:
            print(f"   Role: {contact.role}")
        print(f"   Confidence: {contact.confidence:.0%}")
        print(f"   Source: {contact.source_url}")
