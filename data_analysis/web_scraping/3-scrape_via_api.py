#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """Grab all quotes by querying the site's
    paginated JSON API endpoints sequentially, compiling
    the text, author, and tags into a single list."""
    # Count pagination
    pg_number = 1
    # Ckeck if there is more pages
    more_pages = True
    # Store the data collected
    quote = []
    # While loop through every page
    while more_pages:
        page = base_url + "/api/quotes?page=" + str(pg_number)
        resp = fetch_html(page)
        data = json.loads(resp)
        # For loop to append the in correct way
        for i in data["quotes"]:
            quote_dict = {
                "text": i["text"],
                "author": i["author"]["name"],
                "tags": i["tags"],
            }
            quote.append(quote_dict)
        # If/Else to check the pages
        if data["has_next"]:
            pg_number += 1
        else:
            more_pages = False
    return quote
