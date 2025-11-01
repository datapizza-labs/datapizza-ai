"""
🖼️ Image Generator - Gemini Only (Semplificato)
================================================
Generazione di descrizioni immagini con Google Gemini.
Non genera immagini reali, ma descrizioni dettagliate.

Autore: Antonio Mainenti
"""

import os
import json
from typing import Optional
from pathlib import Path
from google.genai import Client


class GeminiImageDescriptionGenerator:
    """Genera descrizioni di immagini usando Google Gemini"""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "generated_images"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY non trovata nelle variabili d'ambiente")
        
        self.client = Client(api_key=self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def generate_image_description(
        self, 
        topic: str, 
        target_audience: str = "",
        style: str = "modern",
        image_style: str = "professional"
    ) -> dict:
        """
        Genera una descrizione dettagliata di un'immagine per il topic dato.
        
        Args:
            topic: Argomento/tema dell'immagine
            target_audience: Pubblico target
            style: Stile della descrizione
            image_style: Stile visivo dell'immagine (professional, artistic, etc)
        
        Returns:
            dict con 'description' e 'keywords'
        """
        
        # Crea prompt per Gemini
        audience_text = f" per {target_audience}" if target_audience else ""
        prompt = f"""Genera una descrizione DETTAGLIATA e VIVIDA di un'immagine su: "{topic}"{audience_text}

Stile richiesto: {image_style}
Tono: {style}

La descrizione deve essere:
- Ricca di dettagli visivi
- Evocativa e ispirante
- Adatta per social media (Instagram/LinkedIn)
- Lunga 2-3 frasi
- In italiano

Inoltre, estrai 5 parole chiave principali.

Rispondi nel formato:
DESCRIZIONE: [descrizione qui]
KEYWORDS: [keyword1, keyword2, keyword3, keyword4, keyword5]"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            
            text = str(response.text) if response.text else ""
            
            # Parse risposta
            description = ""
            keywords = []
            
            if text and "DESCRIZIONE:" in text:
                desc_start = text.find("DESCRIZIONE:") + len("DESCRIZIONE:")
                desc_end = text.find("KEYWORDS:") if "KEYWORDS:" in text else len(text)
                description = text[desc_start:desc_end].strip()
            
            if text and "KEYWORDS:" in text:
                keywords_start = text.find("KEYWORDS:") + len("KEYWORDS:")
                keywords_text = text[keywords_start:].strip()
                keywords = [k.strip() for k in keywords_text.split(",")]
            
            return {
                "description": description,
                "keywords": keywords,
                "topic": topic,
                "style": image_style,
                "source": "gemini-2.0-flash-exp"
            }
        
        except Exception as e:
            print(f"❌ Errore nella generazione: {e}")
            return {
                "description": f"Immagine per: {topic}",
                "keywords": [topic],
                "error": str(e)
            }
    
    def save_description(self, image_data: dict, filename: Optional[str] = None) -> str:
        """Salva la descrizione in un file JSON"""
        if not filename:
            filename = f"image_{image_data['topic'].replace(' ', '_')}.json"
        
        filepath = self.cache_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(image_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)


# Interfaccia unica semplificata
class ImageGenerator:
    """Wrapper semplificato per generazione immagini"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.generator = GeminiImageDescriptionGenerator(api_key)
    
    def generate_image(
        self,
        topic: str,
        target_audience: str = "",
        style: str = "modern",
        image_style: str = "professional",
        provider: str = "gemini"  # Parametro ignorato (solo Gemini)
    ) -> dict:
        """Genera descrizione immagine"""
        return self.generator.generate_image_description(
            topic=topic,
            target_audience=target_audience,
            style=style,
            image_style=image_style
        )
