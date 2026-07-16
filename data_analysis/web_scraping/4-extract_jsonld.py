#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import json
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """Extract quote data from embedded JSON-LD schema blocks
    within a web page's HTML, parsing the structured
    metadata into structured dictionaries."""
    # Fetch the url
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    # Store the data collected
    quotes = []
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        data = json.loads(script.string)
        # a single page might have one object or a list of them
        if isinstance(data, list):
            items = data
        else:
            items = [data]

        for item in items:
            if item.get("@type") != "Quote":
                continue

            text = item.get("text")
            author = item.get("author", {}).get("name")

            tags = item.get("keywords", "")
            if isinstance(tags, str):
                tags = tags.split(",")

            quotes.append({
                "text": text,
                "author": author,
                "tags": tags,
            })
    return quotes
