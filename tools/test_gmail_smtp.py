"""
Test Gmail SMTP Configuration
Verifica che le credenziali Gmail funzionino correttamente
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from pathlib import Path

def load_config():
    """Carica configurazione email"""
    config_path = Path("configs/email_config.yaml")
    
    if not config_path.exists():
        print("❌ File configs/email_config.yaml non trovato!")
        print("\n📝 Crea il file seguendo la guida in docs/GMAIL_SETUP.md")
        return None
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_smtp():
    """Test invio email"""
    
    print("📧 Test Gmail SMTP Configuration")
    print("=" * 50)
    
    # Carica config
    config = load_config()
    if not config:
        return
    
    smtp_config = config["smtp"]
    sender = config["sender"]
    creds = config["credentials"]
    
    print(f"\n📤 Mittente: {sender['name']} <{sender['email']}>")
    print(f"🔒 Server: {smtp_config['host']}:{smtp_config['port']}")
    print(f"🔐 TLS: {smtp_config['use_tls']}")
    
    # Crea messaggio di test
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{sender['name']} <{sender['email']}>"
    msg["To"] = sender["email"]  # Invia a te stesso
    msg["Subject"] = "🧪 Test Email Automation - MOOD System"
    
    # Corpo email
    text_body = """
Test Email dal Sistema Multi-Agent

Questo è un messaggio di test per verificare che il sistema di email automation funzioni correttamente.

Se ricevi questa email, la configurazione Gmail SMTP è corretta! ✅

Prossimi passi:
1. ✅ SMTP configurato correttamente
2. 🚀 Pronto per inviare email professionali ai musei
3. 🎨 Sistema Writer → Critic → Reviser attivo

---
Antonio Mainenti
Creator of MOOD - Adaptive Artistic Environment
https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system
"""
    
    html_body = """
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <h2 style="color: #667eea;">🧪 Test Email dal Sistema Multi-Agent</h2>
    
    <p>Questo è un messaggio di test per verificare che il sistema di <strong>email automation</strong> funzioni correttamente.</p>
    
    <div style="background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
        <strong>✅ Se ricevi questa email, la configurazione Gmail SMTP è corretta!</strong>
    </div>
    
    <h3>🎯 Prossimi passi:</h3>
    <ul>
        <li>✅ SMTP configurato correttamente</li>
        <li>🚀 Pronto per inviare email professionali ai musei</li>
        <li>🎨 Sistema Writer → Critic → Reviser attivo</li>
    </ul>
    
    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
    
    <p style="color: #666; font-size: 0.9em;">
        <strong>Antonio Mainenti</strong><br>
        Live Sound Engineer | Spatial Audio → Audio AI Engineer<br>
        Creator of MOOD - Adaptive Artistic Environment<br><br>
        🌐 <a href="https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system">MOOD on GitHub</a><br>
        📧 oggettosonoro@gmail.com
    </p>
</body>
</html>
"""
    
    # Allega entrambe le versioni
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    
    print("\n🔄 Connessione al server SMTP...")
    
    try:
        # Connetti a Gmail
        server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
        
        if smtp_config["use_tls"]:
            print("🔒 Avvio TLS...")
            server.starttls()
        
        print("🔐 Autenticazione...")
        server.login(sender["email"], creds["password"])
        
        print("📤 Invio email di test...")
        server.send_message(msg)
        
        print("\n✅ EMAIL INVIATA CON SUCCESSO!")
        print(f"\n📬 Controlla la tua inbox: {sender['email']}")
        print("   (Controlla anche SPAM se non la vedi)")
        
        server.quit()
        
        print("\n" + "=" * 50)
        print("🎉 Configurazione Gmail completata!")
        print("🚀 Pronto per inviare email professionali per MOOD")
        
    except smtplib.SMTPAuthenticationError:
        print("\n❌ ERRORE: Autenticazione fallita")
        print("\n🔧 Possibili cause:")
        print("   1. App Password non corretta")
        print("   2. 2FA non attivato su Gmail")
        print("   3. Password copiata con spazi (rimuovili!)")
        print("\n📝 Segui la guida: docs/GMAIL_SETUP.md")
        
    except smtplib.SMTPException as e:
        print(f"\n❌ ERRORE SMTP: {e}")
        print("\n🔧 Verifica:")
        print("   - Server: smtp.gmail.com")
        print("   - Port: 587")
        print("   - TLS: true")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("\n📝 Verifica la configurazione in configs/email_config.yaml")

if __name__ == "__main__":
    test_smtp()
