#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """Log into a website using a persistent session to retrieve a token,
    authenticate with credentials,
    and scrape quotes from the protected page."""
    session = requests.Session()
    # first GET the form, we need its hidden csrf_token to log in
    login_page = session.get(login_url)
    soup = BeautifulSoup(login_page.text, "html.parser")

    # the token changes every time, so it must be read, not hardcoded
    token_input = soup.find("input", {"name": "csrf_token"})
    csrf_token = token_input["value"]

    # site checks these 3 fields to accept the login
    payload = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token,
    }
    session.post(login_url, data=payload)

    # same session object -> cookie is sent automatically here
    home_url = login_url.replace("/login", "/")
    home_page = session.get(home_url)
    soup = BeautifulSoup(home_page.text, "html.parser")

    quotes = []
    for quote in soup.select(".quote"):
        text = quote.select_one(".text").get_text()
        author = quote.select_one(".author").get_text()

        tags = []
        for tag in quote.select(".tag"):
            tags.append(tag.get_text())

        quotes.append({
            "text": text,
            "author": author,
            "tags": tags,
        })
    return quotes
