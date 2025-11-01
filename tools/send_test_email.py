#!/usr/bin/env python3
"""
Test invio email reale con il sistema MOOD
"""

import smtplib
import yaml
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path


def load_email_config():
    """Carica configurazione email"""
    config_path = Path(__file__).parent.parent / "configs" / "email_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def send_real_email(recipient_email: str, subject: str, body: str):
    """
    Invia una email reale
    
    Args:
        recipient_email: Email destinatario
        subject: Oggetto email
        body: Corpo email (può contenere HTML)
    """
    print("📧 Invio Email Reale")
    print("=" * 50)
    
    # Carica configurazione
    config = load_email_config()
    
    sender_email = config['sender']['email']
    sender_name = config['sender']['name']
    password = config['credentials']['password']
    signature = config.get('signature', '')
    
    print(f"📤 Da: {sender_name} <{sender_email}>")
    print(f"📬 A: {recipient_email}")
    print(f"📝 Oggetto: {subject}")
    print()
    
    # Crea messaggio
    msg = MIMEMultipart('alternative')
    msg['From'] = f"{sender_name} <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    
    # Corpo email con signature
    full_body = f"{body}\n\n{signature}"
    
    # Parte testo
    text_part = MIMEText(full_body, 'plain')
    msg.attach(text_part)
    
    # Connessione SMTP
    try:
        print("🔄 Connessione al server SMTP...")
        server = smtplib.SMTP(config['smtp']['host'], config['smtp']['port'])
        
        if config['smtp']['use_tls']:
            print("🔒 Avvio TLS...")
            server.starttls()
        
        print("🔐 Autenticazione...")
        server.login(sender_email, password)
        
        print("📤 Invio email...")
        server.send_message(msg)
        server.quit()
        
        print()
        print("✅ EMAIL INVIATA CON SUCCESSO!")
        print()
        print(f"📬 Controlla la inbox di: {recipient_email}")
        print("   (Controlla anche SPAM se non la vedi)")
        
        return True
        
    except Exception as e:
        print()
        print(f"❌ ERRORE: {e}")
        return False


def main():
    """Test invio email"""
    
    print()
    print("🎯 Test Invio Email MOOD Agent")
    print("=" * 50)
    print()
    
    # Destinatario
    recipient = input("📧 Email destinatario (premi INVIO per usare la tua): ").strip()
    if not recipient:
        config = load_email_config()
        recipient = config['sender']['email']
        print(f"   → Uso: {recipient}")
    
    print()
    
    # Oggetto
    default_subject = f"🤖 Test MOOD Agent System - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    subject = input(f"📝 Oggetto (premi INVIO per default):\n   → ").strip()
    if not subject:
        subject = default_subject
        print(f"   Uso: {subject}")
    
    print()
    
    # Corpo
    print("✍️  Corpo email (premi INVIO per usare template):")
    body_input = input("   → ").strip()
    
    if not body_input:
        body = f"""Ciao! 👋

Questo è un test del sistema email del MOOD Agent.

Il sistema è stato configurato con successo e ora può:
• ✉️ Inviare email automatiche
• 📊 Notifiche per approvazioni critiche
• 🔍 Report di ricerca settimanali
• 🤝 Outreach automatizzato

Timestamp: {datetime.now().strftime('%d/%m/%Y alle %H:%M:%S')}

Tutto funziona perfettamente! 🎉"""
    else:
        body = body_input
    
    print()
    print("=" * 50)
    print()
    
    # Conferma
    print("📋 RIEPILOGO:")
    print(f"   Da: {load_email_config()['sender']['email']}")
    print(f"   A: {recipient}")
    print(f"   Oggetto: {subject}")
    print()
    
    confirm = input("🚀 Inviare email? (s/n): ").strip().lower()
    if confirm != 's':
        print("❌ Annullato")
        return
    
    print()
    
    # Invio
    success = send_real_email(recipient, subject, body)
    
    if success:
        print()
        print("=" * 50)
        print("🎉 Sistema email MOOD completamente funzionante!")
        print()


if __name__ == "__main__":
    main()
