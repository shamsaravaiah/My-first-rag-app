#!/usr/bin/env python3
"""
Aprobo.com Complete Product Scraper - Vector DB Ready
Scrapes ALL products with structured data optimized for vector database ingestion
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import re
import hashlib
from urllib.parse import urljoin, urlparse
from datetime import datetime
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProductDocument:
    """Structured product document for vector DB"""
    # Primary identifiers
    id: str
    name: str
    url: str
    collection: str
    
    # Content for embeddings
    title: str
    description: str
    full_text: str
    summary: str
    
    # Structured data
    category: str
    subcategory: str
    applications: List[str]
    features: List[str]
    specifications: Dict[str, str]
    technical_data: Dict[str, str]
    
    # Metadata
    images: List[Dict[str, str]]
    related_products: List[str]
    keywords: List[str]
    
    # Vector DB metadata
    document_type: str = "product"
    source: str = "aprobo.com"
    scraped_at: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        """Generate metadata after initialization"""
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()
        if not self.content_hash:
            content_for_hash = f"{self.name}{self.description}{self.full_text}"
            self.content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()

class AproboVectorScraper:
    def __init__(self):
        self.base_url = "https://aprobo.com"
        self.start_url = "https://aprobo.com/en/produkter/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        self.documents = []
        self.scraped_urls = set()
        
        # Product categorization keywords
        self.category_keywords = {
            'acoustic': ['decibel', 'sound', 'acoustic', 'noise', 'db', 'soundseal'],
            'flooring': ['wood', 'parquet', 'laminate', 'linoleum', 'lvt', 'floor'],
            'textile': ['rug', 'carpet', 'wool', 'textile', 'tile'],
            'accessories': ['underlay', 'glue', 'tape', 'protection', 'molding']
        }
        
    def get_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch page with comprehensive error handling"""
        for attempt in range(retries):
            try:
                logger.info(f"Fetching: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = response.apparent_encoding
                
                soup = BeautifulSoup(response.content, 'html.parser')
                time.sleep(1.5)  # Respectful delay
                return soup
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to fetch {url}")
                    return None
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Advanced text cleaning for vector DB"""
        if not text:
            return ""
        
        # Remove HTML entities and normalize whitespace
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, name: str = "") -> List[str]:
        """Extract relevant keywords for better searchability"""
        keywords = set()
        
        # Add name words
        if name:
            keywords.update(re.findall(r'\b\w+\b', name.lower()))
        
        # Extract technical terms and measurements
        technical_patterns = [
            r'\b\d+\s*(?:mm|cm|m|db|kg|%)\b',  # Measurements
            r'\b(?:acoustic|sound|noise|step|drum|impact)\b',  # Acoustic terms
            r'\b(?:parquet|linoleum|laminate|wood|textile|concrete)\b',  # Materials
            r'\b(?:floating|glued|cast|integrated)\b',  # Installation types
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text.lower())
            keywords.update(matches)
        
        # Add product-specific terms
        product_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        keywords.update([term.lower() for term in product_terms if len(term) > 2])
        
        return list(keywords)[:20]  # Limit to top 20 keywords
    
    def categorize_product(self, name: str, description: str, full_text: str) -> tuple:
        """Intelligently categorize products"""
        combined_text = f"{name} {description} {full_text}".lower()
        
        # Determine primary category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                category_scores[category] = score
        
        primary_category = max(category_scores.keys(), default="general") if category_scores else "general"
        
        # Determine subcategory based on specific product names
        subcategory = "general"
        if "decibel" in name.lower():
            subcategory = f"decibel_{name.lower().split()[-1]}" if name.lower().split() else "decibel"
        elif "art wood" in combined_text:
            subcategory = "art_wood"
        elif "zealand" in combined_text:
            subcategory = "zealand_collection"
        elif any(term in combined_text for term in ["underlay", "tape", "glue"]):
            subcategory = "accessories"
        
        return primary_category, subcategory
    
    def extract_specifications(self, soup: BeautifulSoup, full_text: str) -> Dict[str, str]:
        """Extract technical specifications and measurements"""
        specs = {}
        
        # Look for measurement patterns
        measurement_patterns = [
            (r'(\d+(?:\.\d+)?)\s*mm', 'thickness_mm'),
            (r'(\d+(?:\.\d+)?)\s*db', 'sound_reduction_db'),
            (r'(\d+(?:\.\d+)?)\s*kg', 'weight_kg'),
            (r'(\d+(?:\.\d+)?)\s*%', 'improvement_percent'),
        ]
        
        for pattern, spec_name in measurement_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                specs[spec_name] = matches[0]
        
        # Look for material specifications
        if 'wool' in full_text.lower():
            specs['material'] = 'wool'
        elif 'wood' in full_text.lower():
            specs['material'] = 'wood'
        elif 'concrete' in full_text.lower():
            specs['material'] = 'concrete'
        
        # Installation method
        if 'floating' in full_text.lower():
            specs['installation'] = 'floating'
        elif 'glued' in full_text.lower():
            specs['installation'] = 'glued'
        elif 'cast' in full_text.lower():
            specs['installation'] = 'cast'
        
        return specs
    
    def create_rich_summary(self, name: str, description: str, applications: List[str], features: List[str]) -> str:
        """Create a rich summary optimized for vector search"""
        summary_parts = []
        
        if name:
            summary_parts.append(f"Product: {name}")
        
        if description:
            summary_parts.append(f"Description: {description}")
        
        if applications:
            app_text = " ".join(applications[:2])  # Top 2 applications
            summary_parts.append(f"Applications: {app_text}")
        
        if features:
            feat_text = " ".join(features[:2])  # Top 2 features  
            summary_parts.append(f"Features: {feat_text}")
        
        return " | ".join(summary_parts)
    
    def extract_comprehensive_product_data(self, soup: BeautifulSoup, url: str) -> Optional[ProductDocument]:
        """Extract all product data and structure for vector DB"""
        
        # Extract basic info
        name = ""
        name_selectors = ['h1', '.product-title', '.entry-title', 'title']
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                name = self.clean_and_normalize_text(element.get_text())
                break
        
        if not name:
            logger.warning(f"No product name found for {url}")
            return None
        
        # Extract collection
        collection = ""
        collection_indicators = soup.find_all(text=re.compile(r'(collection|concept)', re.IGNORECASE))
        if collection_indicators:
            for indicator in collection_indicators:
                parent = indicator.parent
                if parent and parent.name in ['h1', 'h2', 'h3', 'div']:
                    collection = self.clean_and_normalize_text(parent.get_text())
                    if len(collection) < 50:  # Reasonable collection name length
                        break
        
        # Extract all text content
        content_areas = soup.find_all(['div', 'section', 'article', 'p'])
        text_parts = []
        description_parts = []
        
        for area in content_areas:
            text = self.clean_and_normalize_text(area.get_text())
            if text and len(text) > 20 and text not in text_parts:
                text_parts.append(text)
                if len(description_parts) < 3 and len(text) > 50:
                    description_parts.append(text)
        
        full_text = " ".join(text_parts)
        description = " ".join(description_parts)
        
        # Extract structured data
        applications = []
        features = []
        
        for text in text_parts:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in ['intended', 'designed', 'suitable', 'combined with', 'application']):
                applications.append(text)
            elif any(keyword in text_lower for keyword in ['provides', 'offers', 'improves', 'reduces', 'acoustic', 'sound']):
                features.append(text)
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and not any(skip in src.lower() for skip in ['logo', 'icon', 'button']):
                images.append({
                    'url': urljoin(self.base_url, src),
                    'alt': self.clean_and_normalize_text(img.get('alt', '')),
                    'description': self.clean_and_normalize_text(img.get('title', ''))
                })
        
        # Extract related products
        related_products = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if '/product/' in href and href != url:
                related_products.append(self.clean_and_normalize_text(link.get_text()))
        
        # Categorize and extract specifications
        category, subcategory = self.categorize_product(name, description, full_text)
        specifications = self.extract_specifications(soup, full_text)
        keywords = self.extract_keywords(full_text, name)
        
        # Create summary
        summary = self.create_rich_summary(name, description, applications, features)
        
        # Generate unique ID
        product_id = hashlib.md5(url.encode()).hexdigest()
        
        # Create document
        document = ProductDocument(
            id=product_id,
            name=name,
            url=url,
            collection=collection,
            title=name,
            description=description[:500],  # Limit for better embedding
            full_text=full_text[:2000],     # Limit for better embedding
            summary=summary,
            category=category,
            subcategory=subcategory,
            applications=applications[:5],   # Top 5
            features=features[:5],          # Top 5
            specifications=specifications,
            technical_data=specifications,  # Alias for compatibility
            images=images,
            related_products=related_products[:10],
            keywords=keywords
        )
        
        return document
    
    def discover_all_urls(self) -> Set[str]:
        """Comprehensive URL discovery"""
        all_urls = set()
        
        # Start with main page
        main_soup = self.get_page(self.start_url)
        if not main_soup:
            return all_urls
        
        # Find collection URLs
        collection_urls = set()
        for link in main_soup.find_all('a', href=True):
            href = link.get('href', '')
            if 'product_collection' in href or 'collection' in link.get_text().lower():
                collection_urls.add(urljoin(self.base_url, href))
        
        logger.info(f"Found {len(collection_urls)} collections")
        
        # Visit each collection
        for collection_url in collection_urls:
            collection_soup = self.get_page(collection_url)
            if collection_soup:
                for link in collection_soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if '/product/' in href and '/product_collection/' not in href:
                        all_urls.add(urljoin(self.base_url, href))
        
        # Also check main page for direct product links
        for link in main_soup.find_all('a', href=True):
            href = link.get('href', '')
            if '/product/' in href:
                all_urls.add(urljoin(self.base_url, href))
        
        logger.info(f"Discovered {len(all_urls)} total product URLs")
        return all_urls
    
    def scrape_all_products(self) -> List[ProductDocument]:
        """Main scraping method optimized for vector DB"""
        logger.info("Starting comprehensive product scraping for Vector DB...")
        
        # Discover all product URLs
        product_urls = self.discover_all_urls()
        
        # Scrape each product
        for i, url in enumerate(sorted(product_urls), 1):
            logger.info(f"Processing product {i}/{len(product_urls)}: {url}")
            
            soup = self.get_page(url)
            if not soup:
                continue
            
            document = self.extract_comprehensive_product_data(soup, url)
            if document:
                self.documents.append(document)
                logger.info(f"✓ Extracted: {document.name} ({document.category}/{document.subcategory})")
            else:
                logger.warning(f"⚠ Failed to extract data from: {url}")
        
        logger.info(f"Scraping complete! {len(self.documents)} products ready for vector DB")
        return self.documents
    
    def export_for_vector_db(self, format_type: str = "jsonl") -> str:
        """Export data in vector DB ready formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "jsonl":
            filename = f"aprobo_vector_ready_{timestamp}.jsonl"
            with open(filename, 'w', encoding='utf-8') as f:
                for doc in self.documents:
                    f.write(json.dumps(asdict(doc), ensure_ascii=False) + '\n')
        
        elif format_type == "json":
            filename = f"aprobo_vector_ready_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([asdict(doc) for doc in self.documents], f, indent=2, ensure_ascii=False)
        
        elif format_type == "csv":
            filename = f"aprobo_vector_ready_{timestamp}.csv"
            if self.documents:
                # Flatten for CSV
                flattened = []
                for doc in self.documents:
                    flat = {
                        'id': doc.id,
                        'name': doc.name,
                        'url': doc.url,
                        'collection': doc.collection,
                        'title': doc.title,
                        'description': doc.description,
                        'full_text': doc.full_text[:1000] + "..." if len(doc.full_text) > 1000 else doc.full_text,
                        'summary': doc.summary,
                        'category': doc.category,
                        'subcategory': doc.subcategory,
                        'applications': ' | '.join(doc.applications),
                        'features': ' | '.join(doc.features),
                        'specifications': json.dumps(doc.specifications),
                        'keywords': ' | '.join(doc.keywords),
                        'image_count': len(doc.images),
                        'document_type': doc.document_type,
                        'source': doc.source,
                        'scraped_at': doc.scraped_at,
                        'content_hash': doc.content_hash
                    }
                    flattened.append(flat)
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened)
        
        logger.info(f"Vector DB ready data exported to: {filename}")
        return filename
    
    def create_embedding_ready_texts(self) -> List[Dict]:
        """Create optimized text chunks for embedding"""
        embedding_docs = []
        
        for doc in self.documents:
            # Main document embedding
            main_text = f"{doc.title}. {doc.description}. {doc.summary}"
            embedding_docs.append({
                'id': f"{doc.id}_main",
                'text': main_text,
                'metadata': {
                    'product_id': doc.id,
                    'product_name': doc.name,
                    'category': doc.category,
                    'subcategory': doc.subcategory,
                    'url': doc.url,
                    'type': 'main_description'
                }
            })
            
            # Applications embedding (if substantial)
            if doc.applications:
                app_text = f"Applications for {doc.name}: " + " ".join(doc.applications)
                if len(app_text) > 100:
                    embedding_docs.append({
                        'id': f"{doc.id}_applications",
                        'text': app_text,
                        'metadata': {
                            'product_id': doc.id,
                            'product_name': doc.name,
                            'category': doc.category,
                            'type': 'applications'
                        }
                    })
            
            # Features embedding (if substantial)
            if doc.features:
                feat_text = f"Features of {doc.name}: " + " ".join(doc.features)
                if len(feat_text) > 100:
                    embedding_docs.append({
                        'id': f"{doc.id}_features",
                        'text': feat_text,
                        'metadata': {
                            'product_id': doc.id,
                            'product_name': doc.name,
                            'category': doc.category,
                            'type': 'features'
                        }
                    })
        
        return embedding_docs
    
    def save_embedding_ready_data(self):
        """Save data optimized for embedding models"""
        embedding_docs = self.create_embedding_ready_texts()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aprobo_embeddings_ready_{timestamp}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for doc in embedding_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"Embedding-ready data saved to: {filename}")
        logger.info(f"Created {len(embedding_docs)} embedding documents")
        return filename

def main():
    """Run the complete vector DB ready scraping"""
    scraper = AproboVectorScraper()
    
    try:
        # Scrape all products
        documents = scraper.scrape_all_products()
        
        if not documents:
            logger.error("No products were scraped!")
            return
        
        # Export in multiple formats
        jsonl_file = scraper.export_for_vector_db("jsonl")
        json_file = scraper.export_for_vector_db("json") 
        csv_file = scraper.export_for_vector_db("csv")
        embedding_file = scraper.save_embedding_ready_data()
        
        # Print comprehensive summary
        print(f"\n{'='*60}")
        print(f"VECTOR DB READY SCRAPING COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Total products scraped: {len(documents)}")
        print(f"🏷️  Categories found: {len(set(doc.category for doc in documents))}")
        print(f"📁 Files created:")
        print(f"   • {jsonl_file} - JSONL for vector ingestion")
        print(f"   • {json_file} - JSON for analysis")
        print(f"   • {csv_file} - CSV for spreadsheets")
        print(f"   • {embedding_file} - Optimized chunks for embeddings")
        
        # Category breakdown
        categories = {}
        for doc in documents:
            categories[doc.category] = categories.get(doc.category, 0) + 1
        
        print(f"\n📈 Category breakdown:")
        for category, count in categories.items():
            print(f"   • {category}: {count} products")
        
        # Sample products
        print(f"\n📋 Sample products:")
        for i, doc in enumerate(documents[:5], 1):
            print(f"   {i}. {doc.name} ({doc.category}/{doc.subcategory})")
            print(f"      Keywords: {', '.join(doc.keywords[:5])}")
        
        print(f"\n✅ Ready for vector database ingestion!")
        
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()