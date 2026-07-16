#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import time
from selenium import webdriver


def scrape_products(url):
    """Launch a headless Chrome browser to scrape a static
    product page, using Selenium to extract titles, prices,
    descriptions, and ratings into a list of dictionaries."""
    # headless + fixed window size, exactly what the task asks for
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(1)

    products = []
    # loop through every product card on the page
    for card in driver.find_elements("class name", "thumbnail"):
        title = card.find_element("class name", "title")
        # star count is in data-rating, not in the visible text
        rating = card.find_element(
            "css selector", ".ratings p[data-rating]"
        )
        products.append({
            "title": title.get_attribute("title"),
            "price": card.find_element("class name", "price").text,
            "description": card.find_element(
                "class name", "description"
            ).text,
            "rating": rating.get_attribute("data-rating"),
        })

    return products
