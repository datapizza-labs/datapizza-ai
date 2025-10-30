"""
LinkedIn OAuth 2.0 Authentication Helper
Ottiene access token per pubblicare su LinkedIn
"""

import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import json

# CONFIGURAZIONE - Inserisci i valori dalla tua LinkedIn App
CLIENT_ID = "YOUR_CLIENT_ID_HERE"  # Sostituisci con il tuo Client ID
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"  # Sostituisci con il tuo Client Secret
REDIRECT_URI = "http://localhost:8502/callback"

# Scopes necessari
SCOPES = ["openid", "profile", "w_member_social"]

# Token storage
access_token = None

class CallbackHandler(BaseHTTPRequestHandler):
    """Handler per ricevere il callback OAuth"""
    
    def do_GET(self):
        global access_token
        
        # Parse della query string
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        if "code" in params:
            # Scambia authorization code per access token
            code = params["code"][0]
            
            token_url = "https://www.linkedin.com/oauth/v2/accessToken"
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET
            }
            
            response = requests.post(token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data["access_token"]
                
                # Salva il token
                with open("linkedin_token.json", "w") as f:
                    json.dump(token_data, f, indent=2)
                
                # Risposta di successo
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                <html>
                <head><title>LinkedIn Auth Success</title></head>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: green;">&#x2713; Autenticazione completata!</h1>
                    <p>Access token salvato in <code>linkedin_token.json</code></p>
                    <p>Puoi chiudere questa finestra.</p>
                </body>
                </html>
                """)
                
                print("\n✅ Access Token ottenuto con successo!")
                print(f"Token: {access_token[:20]}...")
                print(f"Salvato in: linkedin_token.json")
            else:
                # Errore
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: red;">❌ Errore</h1>
                    <p>{response.text}</p>
                </body>
                </html>
                """.encode())
                
                print(f"\n❌ Errore: {response.text}")
        else:
            # Nessun code ricevuto
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Error: No code received</h1></body></html>")
    
    def log_message(self, format, *args):
        """Silenzia i log HTTP"""
        pass


def get_linkedin_token():
    """Avvia il processo di autenticazione OAuth"""
    
    print("🔐 LinkedIn OAuth 2.0 Authentication")
    print("=" * 50)
    print(f"\nClient ID: {CLIENT_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print(f"Scopes: {', '.join(SCOPES)}")
    
    # Verifica configurazione
    if CLIENT_ID == "YOUR_CLIENT_ID_HERE" or CLIENT_SECRET == "YOUR_CLIENT_SECRET_HERE":
        print("\n⚠️  ERRORE: Configura prima CLIENT_ID e CLIENT_SECRET!")
        print("\n📝 Istruzioni:")
        print("1. Vai a https://www.linkedin.com/developers/apps")
        print("2. Crea una nuova app o seleziona una esistente")
        print("3. Copia Client ID e Client Secret dalla tab 'Auth'")
        print("4. Modifica questo file e incolla i valori")
        return
    
    # Costruisci URL di autorizzazione
    auth_url = "https://www.linkedin.com/oauth/v2/authorization"
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES)
    }
    
    full_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    
    print("\n🌐 Avvio server locale su http://localhost:8502")
    print("📱 Apertura browser per autenticazione LinkedIn...\n")
    
    # Avvia server locale
    server = HTTPServer(("localhost", 8502), CallbackHandler)
    
    # Apri browser
    webbrowser.open(full_url)
    
    print("⏳ In attesa del callback...")
    print("   (Autorizza l'app nel browser che si è aperto)")
    
    # Gestisci una richiesta (il callback)
    server.handle_request()
    
    # Chiudi server
    server.server_close()
    
    if access_token:
        print("\n✅ Autenticazione completata!")
        print(f"\n📄 Token info:")
        with open("linkedin_token.json", "r") as f:
            data = json.load(f)
            print(f"   Access Token: {data['access_token'][:30]}...")
            print(f"   Expires in: {data.get('expires_in', 'N/A')} seconds")
            print(f"   Scope: {data.get('scope', 'N/A')}")
        
        print("\n🎯 Prossimi passi:")
        print("   1. Copia linkedin_token.json in configs/")
        print("   2. L'agent userà automaticamente questo token")
    else:
        print("\n❌ Autenticazione fallita")


if __name__ == "__main__":
    get_linkedin_token()
