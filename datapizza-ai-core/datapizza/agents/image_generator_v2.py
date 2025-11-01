"""
🖼️ Image Generation Module V2 - Multi-Provider
==============================================
Generazione di immagini con supporto per:
- Gemini (Google) - Usa imagegen API
- Banana.dev - Inference API serverless
- Microsoft Copilot Designer - Bing Image Creator

Autore: Antonio Mainenti
"""

import os
import requests
import json
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import base64


class ImageGeneratorBase(ABC):
    """Base class per generatori di immagini"""
    
    def __init__(self, cache_dir: str = "generated_images"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / f"{self.__class__.__name__}_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Carica metadati delle immagini generate"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Salva metadati"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _create_image_prompt(self, post_text: str, topic: str, style: str) -> str:
        """Crea prompt ottimizzato per generazione immagini"""
        style_descriptions = {
            "modern": "contemporary, clean lines, minimalist",
            "minimalist": "simple, elegant, whitespace-focused",
            "abstract": "surreal, artistic, conceptual",
            "professional": "corporate, polished, business-appropriate",
            "artistic": "creative, expressive, gallery-worthy",
            "vibrant": "colorful, energetic, dynamic",
            "geometric": "geometric shapes, mathematical precision",
            "nature-inspired": "natural elements, organic shapes"
        }
        
        style_desc = style_descriptions.get(style, style)
        
        # Estrai keywords
        keywords = self._extract_keywords(post_text)
        
        prompt = f"""Create a high-quality social media image for: {topic}

Style: {style_desc}
Keywords: {', '.join(keywords[:3])}

Requirements:
- Professional, visually striking
- Suitable for Instagram/LinkedIn
- Square composition (1:1)
- No text or faces
- Modern aesthetic
- Color palette: contemporary

Context: {post_text[:150]}...

Generate an engaging image that captures the essence of this topic."""
        
        return prompt
    
    def _extract_keywords(self, text: str) -> list:
        """Estrae parole chiave dal testo"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'be', 'been', 'being',
            'che', 'di', 'da', 'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una',
            'e', 'o', 'in', 'con', 'per', 'da', 'è', 'sono', 'era', 'were'
        }
        
        words = text.lower().split()
        keywords = [
            w.strip('.,!?;:') for w in words 
            if len(w.strip('.,!?;:')) > 4 and w.lower() not in stop_words
        ]
        
        return list(set(keywords))[:10]
    
    @abstractmethod
    def generate_image(self, post_text: str, topic: str, style: str = "modern") -> Optional[Dict[str, str]]:
        """Genera immagine - implementare in sottoclassi"""
        pass


class GeminiImageGenerator(ImageGeneratorBase):
    """Generatore di immagini usando Google Gemini"""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "generated_images"):
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            print("⚠️ GOOGLE_API_KEY non configurata")
    
    def generate_image(self, post_text: str, topic: str, style: str = "modern") -> Optional[Dict[str, str]]:
        """
        Genera immagine usando Gemini 2.0 con output immagine.
        Nota: Gemini non genera direttamente immagini, ma le descrive
        per poi usarle come placeholder + descrizione.
        """
        if not self.api_key:
            print("⚠️ GOOGLE_API_KEY non configurata - skipping")
            return None
        
        try:
            prompt = self._create_image_prompt(post_text, topic, style)
            
            # Usa Gemini per creare descrizione dettagliata dell'immagine
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"""Genera una descrizione dettagliata e vivida di un'immagine per:

{prompt}

Descrivi l'immagine in modo che potrebbe essere usato per generarla con un altro tool. 
Sii specifico su: colori, composizione, elementi, stile, mood."""
                    }]
                }]
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            
            response = requests.post(
                url,
                headers=headers,
                params={"key": self.api_key},
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Errore Gemini: {response.text}")
                return None
            
            data = response.json()
            description = data['candidates'][0]['content']['parts'][0]['text']
            
            # Salva come immagine "virtuale" con descrizione
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = topic.lower().replace(" ", "_")[:20]
            filename = f"{topic_slug}_{timestamp}_gemini.txt"
            
            file_path = self.cache_dir / filename
            file_path.write_text(f"Generated with Gemini\n\n{description}")
            
            metadata_entry = {
                "provider": "gemini",
                "topic": topic,
                "style": style,
                "prompt": prompt,
                "description": description,
                "local_path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "type": "text_description"
            }
            
            self.metadata[f"{topic}_{timestamp}"] = metadata_entry
            self._save_metadata()
            
            print(f"✅ Descrizione generata con Gemini: {file_path}")
            
            return {
                "url": "gemini://generated",
                "local_path": str(file_path),
                "prompt": prompt,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "provider": "gemini"
            }
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            return None


class BananaImageGenerator(ImageGeneratorBase):
    """Generatore di immagini usando Banana.dev"""
    
    def __init__(self, api_key: Optional[str] = None, model_key: Optional[str] = None, 
                 cache_dir: str = "generated_images"):
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("BANANA_API_KEY", "")
        self.model_key = model_key or os.getenv("BANANA_MODEL_KEY", "")
        if not self.api_key or not self.model_key:
            print("⚠️ BANANA_API_KEY o BANANA_MODEL_KEY non configurate")
    
    def generate_image(self, post_text: str, topic: str, style: str = "modern") -> Optional[Dict[str, str]]:
        """
        Genera immagine usando Banana.dev Stable Diffusion API.
        
        Nota: Richiede account su banana.dev e modello Stable Diffusion
        """
        if not self.api_key or not self.model_key:
            print("⚠️ Credenziali Banana.dev non configurate")
            return None
        
        try:
            prompt = self._create_image_prompt(post_text, topic, style)
            
            # Banana.dev Stable Diffusion endpoint
            url = "https://api.banana.dev/start/v4/"
            
            payload = {
                "apiKey": self.api_key,
                "modelKey": self.model_key,
                "startRequest": {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, ugly, text",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "height": 768,
                    "width": 768,
                    "seed": -1
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code != 200:
                print(f"❌ Errore Banana: {response.text}")
                return None
            
            job_data = response.json()
            job_id = job_data.get('jobid')
            
            if not job_id:
                print("❌ Nessun job_id ricevuto da Banana")
                return None
            
            # Polling per risultato
            print(f"⏳ Generando immagine (job_id: {job_id})...")
            result_url = f"https://api.banana.dev/check/v4/"
            
            for attempt in range(60):  # Max 5 minuti
                check_payload = {
                    "apiKey": self.api_key,
                    "jobid": job_id
                }
                
                result_response = requests.post(result_url, json=check_payload, timeout=30)
                result_data = result_response.json()
                
                if result_data.get('jobStatus') == 'completed':
                    image_base64 = result_data.get('outputs', {}).get('image_base64', '')
                    
                    if image_base64:
                        # Salva immagine
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        topic_slug = topic.lower().replace(" ", "_")[:20]
                        filename = f"{topic_slug}_{timestamp}_banana.png"
                        
                        file_path = self.cache_dir / filename
                        image_bytes = base64.b64decode(image_base64)
                        file_path.write_bytes(image_bytes)
                        
                        metadata_entry = {
                            "provider": "banana",
                            "topic": topic,
                            "style": style,
                            "prompt": prompt,
                            "local_path": str(file_path),
                            "timestamp": datetime.now().isoformat(),
                            "job_id": job_id
                        }
                        
                        self.metadata[f"{topic}_{timestamp}"] = metadata_entry
                        self._save_metadata()
                        
                        print(f"✅ Immagine generata con Banana: {file_path}")
                        
                        return {
                            "url": "banana://generated",
                            "local_path": str(file_path),
                            "prompt": prompt,
                            "timestamp": datetime.now().isoformat(),
                            "provider": "banana",
                            "job_id": job_id
                        }
                
                import time
                time.sleep(5)
            
            print("❌ Timeout generazione Banana")
            return None
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            return None


class CopilotImageGenerator(ImageGeneratorBase):
    """Generatore di immagini usando Microsoft Copilot Designer"""
    
    def __init__(self, cache_dir: str = "generated_images"):
        super().__init__(cache_dir)
        print("📌 Copilot Designer: richiede autenticazione browser")
    
    def generate_image(self, post_text: str, topic: str, style: str = "modern") -> Optional[Dict[str, str]]:
        """
        Genera immagine usando Copilot Designer.
        
        Nota: Copilot Designer (designer.microsoft.com) richiede autenticazione
        e non ha API pubblica. Questo è un placeholder che reindirizza l'utente.
        """
        prompt = self._create_image_prompt(post_text, topic, style)
        
        # Genera URL Copilot Designer
        copilot_url = f"https://designer.microsoft.com/"
        
        # Crea file con istruzioni
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:20]
        filename = f"{topic_slug}_{timestamp}_copilot_prompt.txt"
        
        file_path = self.cache_dir / filename
        instructions = f"""🎨 Copilot Designer - Image Generation Instructions

Prompt da usare:
{prompt}

Istruzioni:
1. Vai su: https://designer.microsoft.com/
2. Accedi con account Microsoft
3. Incolla il prompt qui sopra
4. Clicca "Create"
5. Salva l'immagine

Generato: {datetime.now().isoformat()}
"""
        file_path.write_text(instructions)
        
        return {
            "url": copilot_url,
            "local_path": str(file_path),
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "provider": "copilot",
            "type": "manual",
            "instructions": "Usa Copilot Designer manualmente (non-API)"
        }


class MultiProviderImageGenerator:
    """Wrapper che seleziona il provider migliore"""
    
    def __init__(self, 
                 google_api_key: Optional[str] = None,
                 banana_api_key: Optional[str] = None,
                 banana_model_key: Optional[str] = None,
                 default_provider: Literal["gemini", "banana", "copilot"] = "gemini",
                 cache_dir: str = "generated_images"):
        
        self.default_provider = default_provider
        
        self.providers = {
            "gemini": GeminiImageGenerator(google_api_key, cache_dir),
            "banana": BananaImageGenerator(banana_api_key, banana_model_key, cache_dir),
            "copilot": CopilotImageGenerator(cache_dir)
        }
    
    def generate_image(self, post_text: str, topic: str, style: str = "modern",
                      provider: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Genera immagine usando provider specificato o default.
        
        Args:
            post_text: Testo del post
            topic: Argomento
            style: Stile (modern, artistic, professional, etc)
            provider: Provider da usare (gemini, banana, copilot)
        
        Returns:
            Dict con URL, local_path, prompt, timestamp
        """
        selected_provider = provider or self.default_provider
        
        if selected_provider not in self.providers:
            print(f"⚠️ Provider '{selected_provider}' non riconosciuto")
            selected_provider = "gemini"
        
        print(f"🖼️ Generando con {selected_provider.upper()}...")
        return self.providers[selected_provider].generate_image(post_text, topic, style)
    
    def set_default_provider(self, provider: str):
        """Cambia provider di default"""
        if provider in self.providers:
            self.default_provider = provider
            print(f"✅ Provider di default cambiato a: {provider}")
        else:
            print(f"❌ Provider '{provider}' non supportato")
