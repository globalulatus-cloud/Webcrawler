import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import pandas as pd
import re
from collections import Counter, defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import jieba  # For Chinese word segmentation
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import ssl

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Download required NLTK data (run once)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Web Scraper & Text Analyzer",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-value {
        font-size: 2em;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


class WebScraper:
    def __init__(self, max_workers=5, timeout=10, respect_robots=True):
        self.max_workers = max_workers
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.visited = set()
        self.results = []
        self.lock = threading.Lock()
        self.stop_flag = False
        self.pause_flag = False
        
    def check_robots_txt(self, url):
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots:
            return True
            
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except:
            return True
    
    def detect_language_advanced(self, text):
        """Detect language with support for mixed content"""
        try:
            # Primary language detection
            primary_lang = detect(text)
            
            # Check for script types
            has_latin = bool(re.search(r'[a-zA-Z]', text))
            has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
            has_japanese = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text))
            has_korean = bool(re.search(r'[\uac00-\ud7af]', text))
            has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))
            has_cyrillic = bool(re.search(r'[\u0400-\u04ff]', text))
            
            scripts = []
            if has_latin: scripts.append('Latin')
            if has_chinese: scripts.append('Chinese')
            if has_japanese: scripts.append('Japanese')
            if has_korean: scripts.append('Korean')
            if has_arabic: scripts.append('Arabic')
            if has_cyrillic: scripts.append('Cyrillic')
            
            # Determine if Asian or Latin dominant
            has_asian = has_chinese or has_japanese or has_korean
            
            return {
                'primary': primary_lang,
                'scripts': scripts,
                'is_mixed': len(scripts) > 1,
                'is_asian': has_asian and not has_latin,  # Pure Asian
                'is_latin': has_latin and not has_asian,   # Pure Latin
                'content_type': 'characters' if (has_asian and not has_latin) else 'words'
            }
        except LangDetectException:
            return {
                'primary': 'unknown',
                'scripts': ['Unknown'],
                'is_mixed': False,
                'is_asian': False,
                'is_latin': False,
                'content_type': 'words'
            }
    
    def extract_words_latin(self, text):
        """Extract words and phrases from Latin script text"""
        words = []
        phrases = []
        
        try:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            tokens = [w for w in tokens if w.isalnum() and len(w) > 2]
            
            # Remove stopwords
            try:
                stop_words = set(stopwords.words('english'))
                tokens = [w for w in tokens if w not in stop_words]
            except:
                pass
            
            words.extend(tokens)
            
            # Extract noun phrases using POS tagging
            try:
                pos_tags = pos_tag(tokens)
                for i in range(len(pos_tags) - 1):
                    if pos_tags[i][1].startswith('NN') and pos_tags[i+1][1].startswith('NN'):
                        phrases.append(f"{pos_tags[i][0]} {pos_tags[i+1][0]}")
            except:
                pass
            
            # Extract n-grams (2-3 words)
            for i in range(len(tokens) - 1):
                phrases.append(f"{tokens[i]} {tokens[i+1]}")
                if i < len(tokens) - 2:
                    phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
        except Exception as e:
            st.error(f"Error in Latin extraction: {e}")
        
        return words, phrases
    
    def extract_words_asian(self, text):
        """Extract characters and sequences from Asian scripts"""
        characters = []
        sequences = []
        
        # Extract Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        characters.extend(chinese_chars)
        
        # Chinese word segmentation
        try:
            chinese_text = ''.join(chinese_chars)
            if chinese_text:
                words = jieba.cut(chinese_text)
                sequences.extend([w for w in words if len(w) > 1])
        except:
            pass
        
        # Extract Japanese characters
        japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text)
        characters.extend(japanese_chars)
        
        # Extract Korean characters
        korean_chars = re.findall(r'[\uac00-\ud7af]', text)
        characters.extend(korean_chars)
        
        # Create 2-3 character sequences
        for i in range(len(chinese_chars) - 1):
            sequences.append(f"{chinese_chars[i]}{chinese_chars[i+1]}")
            if i < len(chinese_chars) - 2:
                sequences.append(f"{chinese_chars[i]}{chinese_chars[i+1]}{chinese_chars[i+2]}")
        
        return characters, sequences
    
    def extract_content(self, url):
        """Extract and analyze content from a URL"""
        if self.stop_flag:
            return None
            
        while self.pause_flag:
            time.sleep(0.1)
        
        try:
            # Check robots.txt
            if not self.check_robots_txt(url):
                return None
            
            # Fetch page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Detect language
            lang_info = self.detect_language_advanced(text)
            
            # Extract words and phrases based on language
            all_items = []  # Words or Characters based on language
            all_phrases = []
            
            if 'Latin' in lang_info['scripts']:
                words, phrases = self.extract_words_latin(text)
                all_items.extend(words)
                all_phrases.extend(phrases)
            
            if any(script in lang_info['scripts'] for script in ['Chinese', 'Japanese', 'Korean']):
                chars, sequences = self.extract_words_asian(text)
                all_items.extend(chars)
                all_phrases.extend(sequences)
            
            # Find all links for crawling
            links = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    links.append(full_url)
            
            return {
                'url': url,
                'language': lang_info['primary'],
                'scripts': ', '.join(lang_info['scripts']),
                'is_mixed': lang_info['is_mixed'],
                'content_type': lang_info['content_type'],  # 'words' or 'characters'
                'text_length': len(text),
                'items': all_items,  # Words or Characters
                'phrases': all_phrases,
                'links': links
            }
            
        except Exception as e:
            st.warning(f"Error scraping {url}: {str(e)}")
            return None
    
    def crawl_website(self, start_url, max_depth=2, progress_callback=None):
        """Crawl entire website with depth limit"""
        self.visited = set()
        self.results = []
        self.stop_flag = False
        
        # Queue: (url, depth)
        queue = [(start_url, 0)]
        total_urls = 1
        processed = 0
        
        while queue and not self.stop_flag:
            # Process batch concurrently
            batch = []
            while queue and len(batch) < self.max_workers:
                url, depth = queue.pop(0)
                if url not in self.visited and depth <= max_depth:
                    batch.append((url, depth))
                    self.visited.add(url)
            
            if not batch:
                break
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.extract_content, url): (url, depth) 
                          for url, depth in batch}
                
                for future in as_completed(futures):
                    url, depth = futures[future]
                    result = future.result()
                    
                    if result:
                        result['depth'] = depth
                        with self.lock:
                            self.results.append(result)
                        
                        # Add new links to queue if not at max depth
                        if depth < max_depth:
                            for link in result['links']:
                                if link not in self.visited:
                                    queue.append((link, depth + 1))
                                    total_urls += 1
                    
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_urls, url)
            
            time.sleep(0.5)  # Rate limiting
        
        return self.results
    
    def crawl_specific_urls(self, urls, progress_callback=None):
        """Crawl specific list of URLs"""
        self.visited = set()
        self.results = []
        self.stop_flag = False
        
        total = len(urls)
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_content, url): url for url in urls}
            
            for future in as_completed(futures):
                url = futures[future]
                result = future.result()
                
                if result:
                    result['depth'] = 0
                    with self.lock:
                        self.results.append(result)
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total, url)
        
        return self.results


# Initialize session state
if 'scraper' not in st.session_state:
    st.session_state.scraper = WebScraper()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False


# UI Layout
st.title("🌐 Web Scraper & Text Analyzer")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    max_workers = st.slider("Concurrent Workers", 1, 10, 5,
                           help="Number of pages to scrape simultaneously")
    timeout = st.slider("Request Timeout (seconds)", 5, 30, 10)
    respect_robots = st.checkbox("Respect robots.txt", value=True)
    
    st.session_state.scraper.max_workers = max_workers
    st.session_state.scraper.timeout = timeout
    st.session_state.scraper.respect_robots = respect_robots
    
    st.markdown("---")
    st.info("💡 **Tips:**\n- Start with lower depth for faster results\n- Increase workers for faster crawling\n- Enable robots.txt for ethical scraping")

# Mode selection
col1, col2 = st.columns(2)
with col1:
    mode = st.radio("Scraping Mode", ["Entire Website", "Specific Pages"], horizontal=True)

st.markdown("---")

# Input section
if mode == "Entire Website":
    st.subheader("🌍 Entire Website Crawler")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        website_url = st.text_input("Website URL", placeholder="https://example.com")
    with col2:
        crawl_depth = st.number_input("Crawl Depth", min_value=1, max_value=5, value=2)
    
    st.caption("Crawl depth determines how many levels deep to follow links (1 = homepage only)")
    
else:
    st.subheader("📋 Specific Pages Scraper")
    
    url_list = st.text_area(
        "Enter URLs (one per line)",
        height=200,
        placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3"
    )

# Control buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Start Scraping", type="primary", disabled=st.session_state.is_running):
        st.session_state.is_running = True
        st.session_state.results = None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = current / max(total, 1)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {current}/{total} - {url[:50]}...")
        
        # Start scraping
        if mode == "Entire Website":
            if website_url:
                results = st.session_state.scraper.crawl_website(
                    website_url, crawl_depth, update_progress
                )
                st.session_state.results = results
            else:
                st.error("Please enter a website URL")
        else:
            urls = [u.strip() for u in url_list.split('\n') if u.strip()]
            if urls:
                results = st.session_state.scraper.crawl_specific_urls(urls, update_progress)
                st.session_state.results = results
            else:
                st.error("Please enter at least one URL")
        
        st.session_state.is_running = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

with col2:
    if st.button("⏸️ Pause", disabled=not st.session_state.is_running):
        st.session_state.scraper.pause_flag = not st.session_state.scraper.pause_flag

with col3:
    if st.button("⏹️ Stop", disabled=not st.session_state.is_running):
        st.session_state.scraper.stop_flag = True
        st.session_state.is_running = False

with col4:
    if st.session_state.results:
        if st.button("🔄 Clear Results"):
            st.session_state.results = None
            st.rerun()

st.markdown("---")

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Determine if we're showing words or characters
    content_types = [r['content_type'] for r in results]
    is_character_content = 'characters' in content_types
    
    # Statistics
    st.subheader("📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pages Crawled", len(results))
    
    with col2:
        total_items = sum(len(r['items']) for r in results)
        label = "Total Characters" if is_character_content else "Total Words"
        st.metric(label, f"{total_items:,}")
    
    with col3:
        total_phrases = sum(len(r['phrases']) for r in results)
        st.metric("Total Phrases", f"{total_phrases:,}")
    
    with col4:
        avg_items = total_items / len(results) if results else 0
        label = "Avg Characters/Page" if is_character_content else "Avg Words/Page"
        st.metric(label, f"{avg_items:.1f}")
    
    st.markdown("---")
    
    # Language distribution
    st.subheader("🌐 Language Distribution")
    lang_counts = Counter(r['language'] for r in results)
    lang_df = pd.DataFrame(lang_counts.items(), columns=['Language', 'Count'])
    st.bar_chart(lang_df.set_index('Language'))
    
    st.markdown("---")
    
    # Results table
    st.subheader("📄 Scraped Pages")
    
    # Prepare data for display
    display_data = []
    for r in results:
        content_label = "Characters" if r['content_type'] == 'characters' else "Words"
        display_data.append({
            'URL': r['url'],
            'Language': r['language'],
            'Scripts': r['scripts'],
            'Mixed': '✓' if r['is_mixed'] else '',
            'Content Type': r['content_type'].title(),
            f'{content_label} Count': len(r['items']),
            'Phrase Count': len(r['phrases']),
            'Text Length': r['text_length'],
            'Depth': r['depth']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare detailed CSV
        detailed_data = []
        for r in results:
            content_label = "Characters" if r['content_type'] == 'characters' else "Words"
            top_items = Counter(r['items']).most_common(20)
            top_phrases = Counter(r['phrases']).most_common(20)
            
            detailed_data.append({
                'URL': r['url'],
                'Language': r['language'],
                'Scripts': r['scripts'],
                'Is Mixed': r['is_mixed'],
                'Content Type': r['content_type'].title(),
                f'{content_label} Count': len(r['items']),
                'Phrase Count': len(r['phrases']),
                'Text Length': r['text_length'],
                f'Top {content_label}': '; '.join([f"{w}({c})" for w, c in top_items]),
                'Top Phrases': '; '.join([f"{p}({c})" for p, c in top_phrases]),
                'Depth': r['depth']
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        csv = detailed_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Detailed CSV",
            data=csv,
            file_name=f"scrape_results_{int(time.time())}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Word/Character frequency export
        all_items = []
        for r in results:
            all_items.extend(r['items'])
        
        item_freq = Counter(all_items).most_common(100)
        freq_label = "Character" if is_character_content else "Word"
        freq_df = pd.DataFrame(item_freq, columns=[freq_label, 'Frequency'])
        freq_csv = freq_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=f"📊 Download {freq_label} Frequency",
            data=freq_csv,
            file_name=f"{freq_label.lower()}_frequency_{int(time.time())}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Configure your scraping parameters above and click 'Start Scraping' to begin!")
    
    # Feature showcase
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Fast Concurrent Crawling**
        - Parallel processing
        - Adjustable workers
        - Rate limiting
        """)
    
    with col2:
        st.markdown("""
        **🌍 Multi-Language Support**
        - Latin script → Words
        - Asian script → Characters
        - Mixed content handling
        """)
    
    with col3:
        st.markdown("""
        **📊 Advanced Analysis**
        - Phrase extraction
        - Frequency analysis
        - Language detection
        """)

st.markdown("---")
st.caption("Built with Streamlit • Advanced Web Scraping & NLP Analysis")
