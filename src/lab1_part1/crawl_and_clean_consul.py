#!/usr/bin/env python3
"""
Crawler Selenium pour sites dynamiques (consulat.ma, sites Angular, etc.)
Inclut scroll forcé, délais Angular, et contournement basique de WAF
Usage: python crawl_and_clean_selenium.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import time
import re
from pathlib import Path
from typing import Dict, Optional
import logging
import random

# Configuration
OUTPUT_FILE = "data/crawler_output.jsonl"
URLS_FILE = "raw_urls_selenium.txt"
MIN_TEXT_LENGTH = 100  # Filtrage minimal pour consulat.ma
DELAY_ANGULAR = 3  # Délai pour chargement Angular (secondes)
SCROLL_PAUSE = 1.5  # Pause entre scrolls
MAX_SCROLLS = 5  # Nombre maximum de scrolls

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeleniumCrawler:
    """Crawler avec Selenium pour sites dynamiques et protégés"""
    
    def __init__(self):
        self.driver = None
        self.setup_driver()
        Path("data").mkdir(exist_ok=True)
    
    def setup_driver(self):
        """Configure le driver Chrome avec options anti-détection"""
        chrome_options = Options()
        
        # Options anti-détection et anti-WAF
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User-Agent réaliste
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Autres options utiles
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--lang=fr-FR')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Mode headless optionnel (décommenter si nécessaire)
        # chrome_options.add_argument('--headless')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Masquer webdriver property
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            
            logger.info("Driver Selenium initialisé avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du driver: {e}")
            raise
    
    def human_like_scroll(self):
        """Effectue un scroll progressif comme un humain"""
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        
        scroll_count = 0
        current_position = 0
        
        while current_position < total_height and scroll_count < MAX_SCROLLS:
            # Scroll aléatoire entre 300 et 800 pixels
            scroll_step = random.randint(300, 800)
            current_position += scroll_step
            
            self.driver.execute_script(f"window.scrollTo(0, {current_position});")
            time.sleep(SCROLL_PAUSE + random.uniform(0, 0.5))
            
            scroll_count += 1
            
            # Recalculer hauteur (contenu dynamique peut changer)
            total_height = self.driver.execute_script("return document.body.scrollHeight")
        
        # Retour en haut
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)
    
    def wait_for_angular(self):
        """Attend le chargement complet d'Angular"""
        try:
            # Attendre que Angular soit défini
            WebDriverWait(self.driver, DELAY_ANGULAR).until(
                lambda d: d.execute_script("return typeof angular !== 'undefined' || document.readyState === 'complete'")
            )
            
            # Attendre la fin des requêtes HTTP Angular
            time.sleep(DELAY_ANGULAR)
            
            # Vérifier si Angular est présent et attendre
            angular_present = self.driver.execute_script(
                "return typeof angular !== 'undefined'"
            )
            
            if angular_present:
                self.driver.execute_script(
                    "return angular.element(document).injector().get('$http').pendingRequests.length === 0"
                )
        
        except Exception as e:
            logger.debug(f"Angular non détecté ou erreur d'attente: {e}")
            time.sleep(DELAY_ANGULAR)
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def extract_content(self, url: str) -> Optional[Dict]:
        """Extrait le contenu d'une page chargée"""
        
        # Parser avec BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Supprimer éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()
        
        # Extraction du titre
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.find('title'):
            title = soup.find('title').get_text(strip=True)
        
        # Extraction du contenu principal
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return None
        
        # Extraire le texte
        text = main_content.get_text(separator='\n', strip=True)
        text = self.clean_text(text)
        
        # Filtrage minimal pour consulat.ma
        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(f"Page trop courte: {url} ({len(text)} caractères)")
            return None
        
        return {
            "url": url,
            "title": title,
            "content": text,
            "source": urlparse(url).netloc,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "selenium"
        }
    
    def save_to_jsonl(self, data: Dict):
        """Sauvegarde au format JSONL"""
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        logger.info(f"✓ Sauvegardé: {data['title'][:60]}...")
    
    def crawl_url(self, url: str, retry_count: int = 0):
        """Crawle une URL avec Selenium"""
        max_retries = 2
        
        try:
            logger.info(f"Chargement: {url}")
            
            # Délai aléatoire pour simuler comportement humain
            time.sleep(random.uniform(2, 4))
            
            # Charger la page
            self.driver.get(url)
            
            # Attendre Angular si présent
            self.wait_for_angular()
            
            # Scroll progressif
            self.human_like_scroll()
            
            # Attendre un peu après scroll
            time.sleep(2)
            
            # Vérifier si WAF/blocage
            page_text = self.driver.page_source.lower()
            if "rejected" in page_text or "blocked" in page_text or "access denied" in page_text:
                logger.warning(f"⚠️  Possible blocage WAF détecté sur {url}")
                
                if retry_count < max_retries:
                    logger.info(f"Nouvelle tentative ({retry_count + 1}/{max_retries})...")
                    time.sleep(random.uniform(5, 10))
                    return self.crawl_url(url, retry_count + 1)
                else:
                    logger.error(f"✗ Échec après {max_retries} tentatives: {url}")
                    return
            
            # Extraction du contenu
            content = self.extract_content(url)
            if content:
                self.save_to_jsonl(content)
            else:
                logger.warning(f"Aucun contenu extrait de {url}")
        
        except Exception as e:
            logger.error(f"Erreur lors du crawl de {url}: {e}")
            
            if retry_count < max_retries:
                logger.info(f"Nouvelle tentative ({retry_count + 1}/{max_retries})...")
                time.sleep(random.uniform(5, 10))
                return self.crawl_url(url, retry_count + 1)
    
    def crawl_from_file(self, urls_file: str):
        """Crawle toutes les URLs d'un fichier"""
        
        if not Path(urls_file).exists():
            logger.error(f"Fichier {urls_file} introuvable")
            logger.info("Création d'un fichier exemple...")
            with open(urls_file, 'w', encoding='utf-8') as f:
                f.write("# URLs pour crawler Selenium (sites dynamiques)\n")
                f.write("https://consulat.ma/fr/visa-etudes\n")
                f.write("https://ma.diplomatie.gouv.fr/\n")
            return
        
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Démarrage du crawl Selenium de {len(urls)} URLs...")
        
        try:
            for i, url in enumerate(urls, 1):
                logger.info(f"[{i}/{len(urls)}] Traitement de {url}")
                self.crawl_url(url)
        
        finally:
            self.close()
        
        logger.info(f"Crawl terminé! Données dans {OUTPUT_FILE}")
    
    def close(self):
        """Ferme le driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Driver fermé")


def main():
    """Point d'entrée principal"""
    crawler = SeleniumCrawler()
    crawler.crawl_from_file(URLS_FILE)


if __name__ == "__main__":
    main()