#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic
import time
from urllib import parse



def scrape_paginated(base_url):
    """Scrape all pages of quotes by dynamically
    traversing 'Next' pagination links from a base URL,
    introducing delays between requests."""
    # Grab the raw HTML text from the page
    all_quotes = []  # stack quotes from every page here
    current_url = base_url  # start on page 1
 
    while current_url:
        html = fetch_html(current_url)  # raw HTML of the page
        all_quotes += scrape_basic(html)  # reuse the Task 1 parser
 
        soup = BeautifulSoup(html, "html.parser")
        next_link = soup.select_one("li.next a")  # the 'Next' button
 
        if next_link:
            # href is relative (/page/2/), join it with current URL
            current_url = parse.urljoin(current_url, next_link["href"])
            time.sleep(1)  # be polite, pause between requests
        else:
            current_url = None  # no 'Next' -> stop looping
 
    return all_quotes

    