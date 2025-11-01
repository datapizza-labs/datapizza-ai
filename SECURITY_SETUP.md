# 🔒 Protezione Segreti - Gitleaks Setup

Questo repository usa **gitleaks** per prevenire il commit accidentale di chiavi API, token e altri segreti.

## Come Funziona

- **Pre-commit hook**: Ogni volta che fai `git commit`, gitleaks scansiona i file staged alla ricerca di segreti.
- **Blocco automatico**: Se rileva una chiave API o token, il commit viene bloccato.
- **Configurazione**: `.gitleaks.toml` definisce i pattern da rilevare (Google API keys, OpenAI, Anthropic, ecc.).

## Installazione (già fatto in questo repo)

```bash
# Installa gitleaks
brew install gitleaks

# Il pre-commit hook è già configurato in .git/hooks/pre-commit
```

## File Protetti

Il `.gitignore` esclude automaticamente:
- `.env` (tutte le varianti)
- `configs/personal_profile.yaml`
- `configs/target_organizations.yaml`
- `configs/email_config.yaml`
- `outputs/`, `data/`, `docs/agent_reports/`
- Docs privati: `LINKEDIN_API_SETUP.md`, `MOOD_PROMOTION_GUIDE.md`, `GEMINI_SETUP.md`

## Uso delle Chiavi API

✅ **Corretto**:
```bash
# In .env locale (gitignored)
GOOGLE_API_KEY=your_key_here

# Carica nell'ambiente
source .env
```

❌ **Sbagliato**:
```python
# MAI hardcoded nel codice
api_key = "AIzaSyB..."
```

❌ **Sbagliato**:
```markdown
# MAI nei docs committati
export GOOGLE_API_KEY=AIzaSyB...
```

## Se il Pre-commit Blocca un Commit

1. **Rimuovi il segreto** dal file staged
2. **Sostituisci con placeholder** (es. `YOUR_API_KEY`)
3. **Metti il segreto nel .env** locale

Se sei sicuro che sia un falso positivo:
```bash
git commit --no-verify
```

## Test Manuale

Puoi scansionare il repo in qualsiasi momento:

```bash
# Scansiona tutto il repo
gitleaks detect --config=.gitleaks.toml --verbose

# Scansiona solo file staged
gitleaks protect --staged --config=.gitleaks.toml
```

## Regole Attive

- Google API keys (pattern: `AIza...`)
- OpenAI API keys (pattern: `sk-...`)
- Anthropic API keys (pattern: `sk-ant-...`)
- Pattern generici (`api_key=`, `apikey=`, ecc.)

## Allowlist (Esclusi dalla Scansione)

- `.venv/`, `node_modules/`, `.git/`
- Placeholder: `YOUR_API_KEY`, `your-api-key`, `EXAMPLE_KEY`

---

**Importante**: Questo sistema protegge solo i commit locali. Se una chiave è già stata pushata:
1. Revoca immediatamente la chiave
2. Crea una nuova chiave
3. Usa `git commit --amend` + `git push -f` per riscrivere la history (se il branch non è ancora mergiato)
