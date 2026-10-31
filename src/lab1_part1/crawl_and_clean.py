#!/usr/bin/env python3
"""
Crawler pour sites administratifs statiques (service-public.gouv.fr, campusfrance.org, etc.)
Usage: python crawl_and_clean.py
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import re
from pathlib import Path
from typing import Set, Dict, Optional
import logging

# Configuration
OUTPUT_FILE = "data/crawler_output.jsonl"
URLS_FILE = "raw_urls.txt"
MIN_TEXT_LENGTH = 200  # Longueur minimale pour filtrer les pages vides
DELAY_BETWEEN_REQUESTS = 1.0  # Délai entre requêtes (respect des serveurs)

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StaticWebCrawler:
    """Crawler optimisé pour sites administratifs statiques"""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
        })
        
        # Créer le répertoire de sortie
        Path("data").mkdir(exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        # Supprimer espaces multiples et sauts de ligne excessifs
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """Extrait le contenu pertinent d'une page"""
        
        # Supprimer scripts, styles, navigation, etc.
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extraction du titre
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.find('title'):
            title = soup.find('title').get_text(strip=True)
        
        # Extraction du contenu principal
        main_content = None
        
        # Chercher les zones de contenu principal (priorité)
        for selector in ['main', 'article', '[role="main"]', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Si pas de zone principale, prendre le body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return None
        
        # Extraire le texte
        text = main_content.get_text(separator='\n', strip=True)
        text = self.clean_text(text)
        
        # Filtrage par longueur
        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(f"Page trop courte ignorée: {url} ({len(text)} caractères)")
            return None
        
        return {
            "url": url,
            "title": title,
            "content": text,
            "source": urlparse(url).netloc,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Récupère une page web"""
        try:
            logger.info(f"Récupération: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return BeautifulSoup(response.text, 'html.parser')
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération de {url}: {e}")
            return None
    
    def save_to_jsonl(self, data: Dict):
        """Sauvegarde un enregistrement au format JSONL"""
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        logger.info(f"✓ Sauvegardé: {data['title'][:60]}...")
    
    def crawl_url(self, url: str):
        """Crawle une URL donnée"""
        
        # Éviter les doublons
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        
        # Récupération de la page
        soup = self.fetch_page(url)
        if not soup:
            return
        
        # Extraction du contenu
        content = self.extract_content(soup, url)
        if content:
            self.save_to_jsonl(content)
        
        # Délai pour respecter les serveurs
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    def crawl_from_file(self, urls_file: str):
        """Crawle toutes les URLs d'un fichier"""
        
        # Vérifier l'existence du fichier
        if not Path(urls_file).exists():
            logger.error(f"Fichier {urls_file} introuvable")
            logger.info("Création d'un fichier exemple...")
            with open(urls_file, 'w', encoding='utf-8') as f:
                f.write("# Ajoutez vos URLs ici (une par ligne)\n")
                f.write("https://www.service-public.fr/particuliers/vosdroits/N19804\n")
                f.write("https://www.campusfrance.org/fr/venir-etudier-en-France\n")
            return
        
        # Charger les URLs
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Démarrage du crawl de {len(urls)} URLs...")
        
        # Crawler chaque URL
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] Traitement de {url}")
            self.crawl_url(url)
        
        logger.info(f"Crawl terminé! {len(self.visited_urls)} pages traitées")
        logger.info(f"Données sauvegardées dans {OUTPUT_FILE}")


def main():
    """Point d'entrée principal"""
    crawler = StaticWebCrawler()
    crawler.crawl_from_file(URLS_FILE)


if __name__ == "__main__":
    main()