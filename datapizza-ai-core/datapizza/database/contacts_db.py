"""
📊 Contact & Campaign Database
================================
Database SQLite per tracciare contatti, email inviate, risposte, follow-up.

Schema:
- contacts: contatti trovati dal Hunter
- campaigns: campagne email inviate
- emails_sent: tracking email individuali
- updates_log: log aggiornamenti settimanali

Autore: Antonio Mainenti
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class Contact:
    """Contatto trovato"""
    email: str
    name: Optional[str] = None
    role: Optional[str] = None
    organization: str = ""
    source_url: str = ""
    confidence: float = 0.0
    first_found: str = ""
    last_updated: str = ""
    status: str = "new"  # new, contacted, responded, bounced
    
@dataclass  
class EmailSent:
    """Email inviata"""
    email_to: str
    subject: str
    body: str
    sent_at: str
    campaign_id: Optional[int] = None
    opened: bool = False
    clicked: bool = False
    responded: bool = False
    bounced: bool = False


class ContactDatabase:
    """Gestisce database contatti e campagne"""
    
    def __init__(self, db_path: str = "data/contacts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Use check_same_thread=False for Streamlit compatibility
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Crea schema database"""
        cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Crea schema database"""
        cursor = self.conn.cursor()
        
        # Tabella contatti
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            role TEXT,
            organization TEXT,
            source_url TEXT,
            confidence REAL,
            first_found TEXT,
            last_updated TEXT,
            status TEXT DEFAULT 'new',
            notes TEXT
        )
        """)
        
        # Tabella campagne
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT,
            description TEXT,
            total_contacts INTEGER DEFAULT 0,
            emails_sent INTEGER DEFAULT 0,
            emails_opened INTEGER DEFAULT 0,
            emails_clicked INTEGER DEFAULT 0,
            responses INTEGER DEFAULT 0
        )
        """)
        
        # Tabella email inviate
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails_sent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER,
            contact_id INTEGER,
            email_to TEXT NOT NULL,
            subject TEXT,
            body TEXT,
            sent_at TEXT,
            opened BOOLEAN DEFAULT 0,
            clicked BOOLEAN DEFAULT 0,
            responded BOOLEAN DEFAULT 0,
            bounced BOOLEAN DEFAULT 0,
            response_text TEXT,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
            FOREIGN KEY (contact_id) REFERENCES contacts(id)
        )
        """)
        
        # Tabella log aggiornamenti
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS updates_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_at TEXT,
            organization TEXT,
            new_contacts_found INTEGER,
            contacts_updated INTEGER,
            notes TEXT
        )
        """)
        
        self.conn.commit()
    
    def add_contact(self, contact: Contact) -> int:
        """Aggiunge o aggiorna contatto"""
        cursor = self.conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Set timestamps if not already set
        if not hasattr(contact, 'last_updated') or not contact.last_updated:
            contact.last_updated = now
        if not hasattr(contact, 'first_found') or not contact.first_found:
            contact.first_found = now
        if not hasattr(contact, 'status') or not contact.status:
            contact.status = "new"
        
        cursor.execute("""
        INSERT INTO contacts (email, name, role, organization, source_url, 
                            confidence, first_found, last_updated, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            name = COALESCE(excluded.name, name),
            role = COALESCE(excluded.role, role),
            confidence = excluded.confidence,
            last_updated = excluded.last_updated
        """, (
            contact.email, contact.name, contact.role, contact.organization,
            contact.source_url, contact.confidence, contact.first_found,
            contact.last_updated, contact.status
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_contacts(self, 
                    organization: Optional[str] = None,
                    status: Optional[str] = None,
                    min_confidence: float = 0.0) -> List[Dict]:
        """Recupera contatti con filtri"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM contacts WHERE confidence >= ?"
        params = [min_confidence]
        
        if organization:
            query += " AND organization = ?"
            params.append(organization)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY confidence DESC, last_updated DESC"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def create_campaign(self, name: str, description: str = "") -> int:
        """Crea nuova campagna"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO campaigns (name, created_at, description)
        VALUES (?, ?, ?)
        """, (name, datetime.now().isoformat(), description))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def log_email_sent(self, email_sent: EmailSent, contact_id: int):
        """Registra email inviata"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO emails_sent (campaign_id, contact_id, email_to, subject, 
                                body, sent_at, opened, clicked, responded, bounced)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            email_sent.campaign_id, contact_id, email_sent.email_to,
            email_sent.subject, email_sent.body, email_sent.sent_at,
            email_sent.opened, email_sent.clicked, email_sent.responded,
            email_sent.bounced
        ))
        
        # Aggiorna stato contatto
        cursor.execute("""
        UPDATE contacts SET status = 'contacted' WHERE id = ?
        """, (contact_id,))
        
        # Aggiorna stats campagna
        if email_sent.campaign_id:
            cursor.execute("""
            UPDATE campaigns SET emails_sent = emails_sent + 1 
            WHERE id = ?
            """, (email_sent.campaign_id,))
        
        self.conn.commit()
    
    def log_update(self, organization: str, new_found: int, updated: int, notes: str = ""):
        """Registra aggiornamento settimanale"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO updates_log (updated_at, organization, new_contacts_found, 
                                contacts_updated, notes)
        VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), organization, new_found, updated, notes))
        
        self.conn.commit()
    
    def get_contacts_needing_followup(self, days_since_sent: int = 7) -> List[Dict]:
        """Trova contatti che necessitano follow-up"""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT c.*, e.sent_at, e.subject as last_subject
        FROM contacts c
        JOIN emails_sent e ON c.id = e.contact_id
        WHERE c.status = 'contacted'
        AND e.responded = 0
        AND e.bounced = 0
        AND datetime(e.sent_at) <= datetime('now', '-' || ? || ' days')
        AND NOT EXISTS (
            SELECT 1 FROM emails_sent e2 
            WHERE e2.contact_id = c.id 
            AND e2.sent_at > e.sent_at
        )
        ORDER BY e.sent_at ASC
        """, (days_since_sent,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Statistiche generali"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Totale contatti
        cursor.execute("SELECT COUNT(*) as total FROM contacts")
        stats['total_contacts'] = cursor.fetchone()['total']
        
        # Per stato
        cursor.execute("""
        SELECT status, COUNT(*) as count 
        FROM contacts 
        GROUP BY status
        """)
        stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Email inviate
        cursor.execute("SELECT COUNT(*) as total FROM emails_sent")
        stats['total_emails_sent'] = cursor.fetchone()['total']
        
        # Tasso apertura
        cursor.execute("""
        SELECT 
            SUM(CASE WHEN opened = 1 THEN 1 ELSE 0 END) as opened,
            COUNT(*) as total
        FROM emails_sent
        """)
        row = cursor.fetchone()
        if row and row['total'] > 0:
            stats['open_rate'] = round(row['opened'] / row['total'] * 100, 1)
        else:
            stats['open_rate'] = 0.0
        
        return stats
    
    def close(self):
        """Chiudi connessione"""
        self.conn.close()


if __name__ == "__main__":
    # Test database
    db = ContactDatabase()
    
    # Test add contact
    contact = Contact(
        email="curator@maxxi.art",
        name="Maria Rossi",
        role="Curatore",
        organization="Museo MAXXI Roma",
        source_url="https://www.maxxi.art/contatti",
        confidence=0.9
    )
    
    contact_id = db.add_contact(contact)
    print(f"✅ Contact added: ID {contact_id}")
    
    # Test get contacts
    contacts = db.get_contacts(min_confidence=0.5)
    print(f"📊 Found {len(contacts)} contacts")
    
    # Test stats
    stats = db.get_stats()
    print(f"📈 Stats: {stats}")
    
    db.close()
