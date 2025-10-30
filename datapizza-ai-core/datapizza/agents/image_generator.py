"""
🖼️ Image Generation Module - DALL-E Integration
================================================
Generazione di immagini per social media e contenuti visuali.

Supporta:
- DALL-E 3 (OpenAI)
- Caching delle immagini
- Resize e ottimizzazione
- Metadata storage

Autore: Antonio Mainenti
"""

import os
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class ImageGenerator:
    """Generatore di immagini usando DALL-E 3"""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "generated_images"):
        """
        Inizializza il generatore di immagini.
        
        Args:
            api_key: OpenAI API key (default da env OPENAI_API_KEY)
            cache_dir: Directory per salvare immagini generate
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.api_url = "https://api.openai.com/v1/images/generations"
        
        # Metadata file per tracciare immagini generate
        self.metadata_file = self.cache_dir / "metadata.json"
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
    
    def generate_instagram_image(
        self,
        post_text: str,
        topic: str,
        style: str = "modern",
        size: str = "1024x1024"
    ) -> Optional[Dict[str, str]]:
        """
        Genera immagine per post Instagram basata sul testo.
        
        Args:
            post_text: Testo del post Instagram (usato per generare prompt)
            topic: Argomento del post (es. "MOOD AI system")
            style: Stile artistico (modern, minimalist, abstract, professional)
            size: Dimensione immagine (1024x1024, 1792x1024, 1024x1792)
        
        Returns:
            Dict con url, local_path, prompt, timestamp
        """
        if not self.api_key:
            print("⚠️ OPENAI_API_KEY non configurata - skipping image generation")
            return None
        
        try:
            # Crea prompt per immagine basato su testo e argomento
            image_prompt = self._create_image_prompt(post_text, topic, style)
            
            # Check cache
            cached = self._check_cache(image_prompt)
            if cached:
                print(f"♻️ Immagine trovata in cache: {cached['local_path']}")
                return cached
            
            print(f"🎨 Generando immagine con DALL-E 3...")
            
            # Chiama DALL-E 3
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": image_prompt,
                    "n": 1,
                    "size": size,
                    "quality": "standard",
                    "style": "natural"
                }
            )
            
            if response.status_code != 200:
                print(f"❌ Errore DALL-E: {response.text}")
                return None
            
            data = response.json()
            image_url = data['data'][0]['url']
            revised_prompt = data['data'][0].get('revised_prompt', image_prompt)
            
            # Download e salva immagine localmente
            local_path = self._save_image(image_url, topic)
            
            # Salva metadati
            timestamp = datetime.now().isoformat()
            metadata_entry = {
                "topic": topic,
                "style": style,
                "size": size,
                "prompt": image_prompt,
                "revised_prompt": revised_prompt,
                "url": image_url,
                "local_path": str(local_path),
                "timestamp": timestamp,
                "post_text_preview": post_text[:100]
            }
            
            # Usa hash del prompt come chiave
            prompt_hash = hash(image_prompt) % ((2**31) - 1)
            self.metadata[str(prompt_hash)] = metadata_entry
            self._save_metadata()
            
            print(f"✅ Immagine generata: {local_path}")
            
            return {
                "url": image_url,
                "local_path": str(local_path),
                "prompt": image_prompt,
                "revised_prompt": revised_prompt,
                "timestamp": timestamp
            }
            
        except Exception as e:
            print(f"❌ Errore nella generazione immagine: {e}")
            return None
    
    def _create_image_prompt(self, post_text: str, topic: str, style: str) -> str:
        """
        Crea un prompt DALL-E basato sul testo e argomento.
        
        Args:
            post_text: Testo del post
            topic: Argomento
            style: Stile artistico
        
        Returns:
            Prompt ottimizzato per DALL-E 3
        """
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
        
        # Estrai keywords dal testo
        keywords = self._extract_keywords(post_text)
        
        # Crea prompt
        prompt = f"""Create a high-quality Instagram-ready image for a post about: {topic}

Style: {style_desc}
Keywords from post: {', '.join(keywords[:3])}

Requirements:
- Professional quality, suitable for social media
- Visually striking and engaging
- Composition: centered main element with space around
- Color palette: modern and contemporary
- No text, logos, or faces
- Aspect ratio: square (1:1)
- Resolution: high quality for digital display

Topic context: {post_text[:150]}...

Generate an image that captures the essence of this topic in a {style} style."""
        
        return prompt
    
    def _extract_keywords(self, text: str) -> list:
        """Estrae parole chiave dal testo"""
        # Parole da ignorare
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
    
    def _check_cache(self, prompt: str) -> Optional[Dict]:
        """Verifica se un'immagine è già stata generata con questo prompt"""
        prompt_hash = hash(prompt) % ((2**31) - 1)
        return self.metadata.get(str(prompt_hash))
    
    def _save_image(self, url: str, topic: str) -> Path:
        """Scarica e salva immagine localmente"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Crea filename unico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:20]
        filename = f"{topic_slug}_{timestamp}.png"
        
        file_path = self.cache_dir / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return file_path
    
    def generate_multiple_images(
        self,
        topics: list,
        style: str = "modern"
    ) -> Dict[str, Dict]:
        """
        Genera più immagini in batch.
        
        Args:
            topics: Lista di (topic, post_text) tuples
            style: Stile artistico
        
        Returns:
            Dict con risultati per ogni topic
        """
        results = {}
        for topic, post_text in topics:
            print(f"\n📸 Generando immagine per: {topic}")
            result = self.generate_instagram_image(post_text, topic, style)
            results[topic] = result
        
        return results
