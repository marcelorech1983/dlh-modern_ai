#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """Scrape the first page of quotes from the specified URL,
    extracting the quote text, author, and tags using BeautifulSoup."""
    # Grab the raw HTML text from the page
    html = fetch_html(url)

    # Let BeautifulSoup parse the HTML so we can easily search it
    soup = BeautifulSoup(html, "html.parser")

    # Grab all the individual quote containers from the page
    quote_blocks = soup.find_all("div", class_="quote")

    quotes = []
    for block in quote_blocks:
       # Pull out the quote text and author name from this container
        text = block.find("span", class_="text").text
        author = block.find("small", class_="author").text

        # Find all the tag elements inside this specific quote block
        tag_elements = block.find_all("a", class_="tag")

        # Extract just the clean text from each tag and build a list
        tags = []
        for tag in tag_elements:
            tags.append(tag.text)

        # Put the extracted data into a dictionary for this quote
        quote_dict = {
            "text": text,
            "author": author,
            "tags": tags,
        }

        # Append this quote in our main results list
        quotes.append(quote_dict)

    return quotes