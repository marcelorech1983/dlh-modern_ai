#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """Scrape all pages of quotes by dynamically
    traversing 'Next' pagination links from a base URL,
    introducing delays between requests."""
    # store quotes from every page here
    all_quotes = []
    current_url = base_url

    while current_url:
        # Grab the quotes from the current page and append them to our list
        all_quotes += scrape_basic(current_url)
        # We need to fetch the page again just to look for the "Next" button.
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")
        next_link = soup.select_one("li.next a")

        if next_link:
            # Turn the relative link (like /page/2/) into a full URL
            current_url = parse.urljoin(current_url, next_link["href"])
            # Pause between requests
            time.sleep(1)
        else:
            # No 'Next' stop looping
            current_url = None
    return all_quotes
