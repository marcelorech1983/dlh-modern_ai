#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import requests


def fetch_html(url, headers=None, timeout=10):
    """Fetch the HTML content of a specified URL,
    with support for custom headers, timeout limits,
    and HTTP error raising."""
    # Send the request
    resp = requests.get(url, headers=headers, timeout=timeout)
    # If the server responded with an error stop here and raise an exception
    resp.raise_for_status()
    # Get just the raw HTML text from the response
    html = resp.text
    return html
