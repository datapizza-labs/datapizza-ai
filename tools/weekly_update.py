"""
⏰ Weekly Auto-Update Scheduler
=================================
Esegue automaticamente ricerca contatti ogni settimana.

Funzionalità:
- Scan settimanale di tutti i target in target_organizations.yaml
- Cerca nuovi contatti
- Aggiorna contatti esistenti
- Log risultati
- Notifica via email se trova nuovi decision maker

Esecuzione:
- Manuale: python tools/weekly_update.py
- Automatico: cron job ogni domenica alle 23:00

Autore: Antonio Mainenti
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "datapizza-ai-core"))

import yaml
from datetime import datetime
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from datapizza.agents.contact_hunter import ContactHunterAgent, Contact
from datapizza.database.contacts_db import ContactDatabase


class WeeklyUpdater:
    """Aggiorna automaticamente database contatti ogni settimana"""
    
    def __init__(self, 
                 targets_file: str = "configs/target_organizations.yaml",
                 db_path: str = "data/contacts.db",
                 email_config: str = "configs/email_config.yaml"):
        
        self.hunter = ContactHunterAgent(delay=3.0)  # Rispettoso con i server
        self.db = ContactDatabase(db_path)
        
        # Carica target
        with open(targets_file) as f:
            self.targets = yaml.safe_load(f)['organizations']
        
        # Carica config email per notifiche
        with open(email_config) as f:
            self.email_config = yaml.safe_load(f)
    
    def run_weekly_update(self):
        """Esegue update completo di tutti i target"""
        
        print("=" * 70)
        print("⏰ WEEKLY AUTO-UPDATE - Contact Hunter")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        total_new = 0
        total_updated = 0
        high_value_contacts = []  # Confidence > 0.8
        
        # Processa solo organizzazioni ad alta priorità
        high_priority_orgs = [org for org in self.targets if org.get('priority') == 'alta']
        
        print(f"\n📊 Processing {len(high_priority_orgs)} high-priority organizations\n")
        
        for i, org in enumerate(high_priority_orgs, 1):
            try:
                print(f"\n[{i}/{len(high_priority_orgs)}] {org['name']}")
                print("-" * 70)
                
                # Cerca contatti
                contacts = self.hunter.hunt_contacts(
                    base_url=org['website'],
                    organization_name=org['name']
                )
                
                # Salva nel database
                new_contacts = 0
                updated_contacts = 0
                
                for contact in contacts:
                    # Controlla se già esiste
                    existing = self.db.get_contacts(organization=org['name'])
                    existing_emails = [c['email'] for c in existing]
                    
                    if contact.email not in existing_emails:
                        new_contacts += 1
                        if contact.confidence > 0.8:
                            high_value_contacts.append(contact)
                    else:
                        updated_contacts += 1
                    
                    self.db.add_contact(contact)
                
                total_new += new_contacts
                total_updated += updated_contacts
                
                # Log update
                self.db.log_update(
                    organization=org['name'],
                    new_found=new_contacts,
                    updated=updated_contacts,
                    notes=f"Weekly auto-update. Found {len(contacts)} total contacts."
                )
                
                print(f"   ✅ New: {new_contacts}, Updated: {updated_contacts}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        # Risultati finali
        print("\n" + "=" * 70)
        print("📊 WEEKLY UPDATE COMPLETED")
        print("=" * 70)
        print(f"🆕 New contacts found: {total_new}")
        print(f"🔄 Contacts updated: {total_updated}")
        print(f"⭐ High-value contacts (>0.8): {len(high_value_contacts)}")
        
        # Stats generali
        stats = self.db.get_stats()
        print(f"\n📈 Database Stats:")
        print(f"   Total contacts: {stats['total_contacts']}")
        print(f"   By status: {stats.get('by_status', {})}")
        
        # Invia notifica se nuovi high-value contact
        if high_value_contacts:
            self._send_notification(high_value_contacts, total_new, total_updated)
        
        print(f"\n✅ Update completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _send_notification(self, high_value: List[Contact], new: int, updated: int):
        """Invia email notifica per nuovi contatti importanti"""
        
        smtp_config = self.email_config['smtp']
        sender = self.email_config['sender']
        creds = self.email_config['credentials']
        
        # Crea email
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{sender['name']} <{sender['email']}>"
        msg["To"] = sender["email"]  # Invia a te stesso
        msg["Subject"] = f"🎯 Weekly Update: {len(high_value)} New High-Value Contacts Found!"
        
        # HTML body
        contacts_html = ""
        for c in high_value[:10]:  # Top 10
            contacts_html += f"""
            <div style="background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #667eea;">
                <strong>{c.email}</strong><br>
                {f'👤 {c.name}<br>' if c.name else ''}
                {f'💼 {c.role}<br>' if c.role else ''}
                🏛️ {c.organization}<br>
                ⭐ Confidence: {c.confidence:.0%}
            </div>
            """
        
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2 style="color: #667eea;">🎯 Weekly Contact Update - MOOD Outreach</h2>
    
    <div style="background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
        <h3>📊 Summary</h3>
        <ul>
            <li><strong>{new}</strong> new contacts found</li>
            <li><strong>{updated}</strong> contacts updated</li>
            <li><strong>{len(high_value)}</strong> high-value contacts (confidence > 80%)</li>
        </ul>
    </div>
    
    <h3>⭐ Top High-Value Contacts:</h3>
    {contacts_html}
    
    <hr style="margin: 30px 0;">
    
    <p style="color: #666;">
        <em>Next update: Next Sunday 23:00</em><br>
        Dashboard: <a href="http://localhost:8502">Open Dashboard</a>
    </p>
</body>
</html>
"""
        
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        try:
            # Invia email
            server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
            if smtp_config["use_tls"]:
                server.starttls()
            server.login(sender["email"], creds["password"])
            server.send_message(msg)
            server.quit()
            
            print(f"\n📧 Notification email sent to {sender['email']}")
            
        except Exception as e:
            print(f"\n⚠️  Could not send notification: {e}")
    
    def close(self):
        """Chiudi connessioni"""
        self.db.close()


def setup_cron():
    """Crea cron job per esecuzione automatica settimanale"""
    
    script_path = Path(__file__).absolute()
    python_path = sys.executable
    
    # Comando cron: ogni domenica alle 23:00
    cron_command = f"0 23 * * 0 {python_path} {script_path}"
    
    print("📅 Setup Cron Job per Auto-Update Settimanale")
    print("=" * 70)
    print("\n1. Apri il crontab:")
    print("   crontab -e")
    print("\n2. Aggiungi questa riga:")
    print(f"   {cron_command}")
    print("\n3. Salva e chiudi")
    print("\nℹ️  Questo eseguirà l'update ogni domenica alle 23:00")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Weekly Contact Hunter Auto-Update")
    parser.add_argument("--setup-cron", action="store_true", help="Show cron setup instructions")
    parser.add_argument("--test", action="store_true", help="Test mode (only 3 organizations)")
    
    args = parser.parse_args()
    
    if args.setup_cron:
        setup_cron()
    else:
        updater = WeeklyUpdater()
        
        # Test mode: solo primi 3 target
        if args.test:
            print("🧪 TEST MODE: Processing only 3 organizations\n")
            updater.targets = [org for org in updater.targets if org.get('priority') == 'alta'][:3]
        
        try:
            updater.run_weekly_update()
        finally:
            updater.close()
