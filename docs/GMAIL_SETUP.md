# 📧 Gmail SMTP Setup per Email Automation

Guida rapida per configurare Gmail e inviare email automatiche dal sistema multi-agent.

## 🔐 Step 1: Crea App Password Gmail

Google non permette più l'accesso con password normale per app esterne. Serve una **App Password**.

### 1.1 Attiva 2FA (se non l'hai già)

1. Vai a: https://myaccount.google.com/security
2. Clicca su **"Verifica in due passaggi"** 
3. Segui la procedura (serve il telefono)

### 1.2 Genera App Password

1. Vai a: https://myaccount.google.com/apppasswords
2. Seleziona:
   - **App**: Posta
   - **Dispositivo**: Mac (o altro)
3. Clicca **"Genera"**
4. Ti mostrerà una password di 16 caratteri tipo: `xxxx xxxx xxxx xxxx`

⚠️ **IMPORTANTE**: 
- Copia questa password subito (la vedrai solo una volta!)
- Rimuovi gli spazi quando la usi nel config
- Non condividerla mai

### 1.3 Salva la Configurazione

Crea il file `configs/email_config.yaml`:

```yaml
# Gmail SMTP Configuration
smtp:
  host: smtp.gmail.com
  port: 587
  use_tls: true
  
sender:
  email: oggettosonoro@gmail.com
  name: Antonio Mainenti
  
credentials:
  # Incolla qui la tua App Password Gmail (16 caratteri senza spazi)
  password: "YOUR_16_CHAR_APP_PASSWORD_HERE"

# Signature
signature: |
  
  ---
  Antonio Mainenti
  Live Sound Engineer | Spatial Audio → Audio AI Engineer
  Creator of MOOD - Adaptive Artistic Environment
  
  🌐 GitHub: https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system
  📧 oggettosonoro@gmail.com

# Tracking (opzionale)
tracking:
  enabled: false  # Impostiamo a true più tardi se vuoi
  
# Limiti sicurezza
limits:
  max_per_hour: 20
  max_per_day: 100
  delay_between_sends: 3  # secondi tra un invio e l'altro
```

## 🧪 Step 2: Test Rapido

Testa che tutto funzioni:

```bash
cd /Users/mainenti/datapizza-ai-0.0.2

# Test invio email di prova
python tools/test_gmail_smtp.py
```

Ti invierà una email di test a `oggettosonoro@gmail.com` per verificare che funzioni.

## ⚠️ Troubleshooting

### Errore: "Username and Password not accepted"

**Causa**: App Password non corretta o 2FA non attivato

**Soluzione**:
1. Verifica 2FA attivo
2. Rigenera App Password
3. Copia senza spazi nel config

### Errore: "SMTP AUTH extension not supported"

**Causa**: Port o TLS non configurato

**Soluzione**: Usa sempre port `587` con `use_tls: true`

### Email finisce in SPAM

**Causa**: Gmail gratuito ha reputazione limitata per bulk email

**Soluzioni**:
1. Invia solo a contatti che conosci
2. Usa subject line non "spammy" (evita TUTTO MAIUSCOLO, troppi !!)
3. Personalizza sempre il contenuto (no email identiche)
4. Aggiungi un link "unsubscribe" nel footer

**Il sistema multi-agent già fa tutto questo automaticamente!** ✅

## 📊 Limiti Gmail Gratuito

- **500 email/giorno** (più che sufficiente)
- **100 destinatari per email**
- Massimo 20 MB per allegati

Per MOOD outreach (musei, gallerie) è perfetto!

## 🎯 Prossimi Step

Dopo aver creato l'App Password:

1. ✅ Crea `configs/email_config.yaml` con la tua password
2. ✅ Testa con `python tools/test_gmail_smtp.py`
3. ✅ Usa il dashboard per inviare email professionali!

## 🔗 Link Utili

- [Google App Passwords](https://myaccount.google.com/apppasswords)
- [Gmail SMTP Settings](https://support.google.com/mail/answer/7126229)

---

**Quando hai creato l'App Password, torna qui e dimmi "Password creata"!** 🚀
