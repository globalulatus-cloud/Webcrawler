import streamlit as st
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import re
from langdetect import detect
import time

st.set_page_config(page_title="Office-Safe Website Crawler", layout="wide")
st.title("Office-Safe Website Crawler (Browser-Based)")

urls_input = st.text_area(
    "Enter URLs (one per line)",
    height=150,
    placeholder="https://join.baeminconnect.com/"
)

max_depth = st.slider("Crawl depth", 0, 3, 1)
start = st.button("Start Crawl")

def extract_words(text, lang):
    if lang in ["ja", "ko", "zh", "zh-cn", "zh-tw"]:
        return re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    return re.findall(r"[A-Za-z0-9']+", text)

def crawl(urls):
    edge_options = Options()
    edge_options.add_argument("user-data-dir=C:/Users/%USERNAME%/AppData/Local/Microsoft/Edge/User Data")
    edge_options.add_argument("--profile-directory=Default")

    driver = webdriver.Edge(options=edge_options)

    visited = set()
    rows = []

    for start_url in urls:
        domain = urlparse(start_url).netloc
        stack = [(start_url, 0)]

        while stack:
            url, depth = stack.pop()
            if url in visited or depth > max_depth:
                continue

            visited.add(url)
            st.write(f"Crawling: {url}")

            driver.get(url)
            time.sleep(4)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)

            try:
                lang = detect(text)
            except:
                lang = "unknown"

            tokens = extract_words(text, lang)

            if lang in ["ja", "ko", "zh", "zh-cn", "zh-tw"]:
                word_count = 0
                char_count = len(tokens)
            else:
                word_count = len(tokens)
                char_count = 0

            img_count = len(soup.find_all("img"))

            rows.append({
                "url": url,
                "depth": depth,
                "language": lang,
                "word_count": word_count,
                "char_count": char_count,
                "img_count": img_count
            })

            for a in soup.find_all("a", href=True):
                next_url = urljoin(url, a["href"]).split("#")[0]
                if urlparse(next_url).netloc in ["", domain]:
                    stack.append((next_url, depth + 1))

    driver.quit()
    return rows

if start:
    urls = [u.strip() for u in urls_input.splitlines() if u.startswith("http")]
    data = crawl(urls)

    st.info(f"Pages collected: {len(data)}")

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)

        df.to_csv("page_list.csv", index=False, encoding="utf-8-sig")
        with open("page_list.csv", "rb") as f:
            st.download_button("Download CSV", f, file_name="page_list.csv")
