# Data Collection - Web Scraping

Static and dynamic web scraping pipeline built on quotes.toscrape.com and webscraper.io test sites: fetching raw HTML, parsing with BeautifulSoup, handling pagination and JSON APIs, extracting JSON-LD, scraping behind a login, and driving JS-rendered pages (including infinite scroll) with Selenium.

## Tasks

| # | Task | File | Status |
|---|---|---|---|
| 0 | Fetch HTML | `0-fetch_html.py` | Done |
| 1 | Basic Static Scraping | `1-scrape_basic.py` | Done |
| 2 | Pagination Handling | `2-scrape_paginated.py` | Done |
| 3 | API-Based Scraping | `3-scrape_via_api.py` | Done |
| 4 | JSON-LD Extraction | `4-extract_jsonld.py` | Done |
| 5 | Login & Scrape | `5-login_and_scrape.py` | Not started |
| 6 | Scrape Static Products | `6-products_list.py` | Not started |
| 7 | Scrape Single Product Detail | `7-product_detail.py` | Not started |
| 8 | Scroll & Scrape Products | `8-scroll_and_scrape.py` | Not started |

5 of 9 tasks complete.

## Targets

- `https://quotes.toscrape.com/` — static quotes site, used for tasks 0–5 (pages, JSON API, JSON-LD, login-gated content)
- `https://webscraper.io/test-sites/e-commerce/static` — static product catalog, used for tasks 6–7
- `https://webscraper.io/test-sites/e-commerce/allinone` — JS-rendered infinite-scroll catalog, used for task 8

## Requirements

- Python 3.11 (Ubuntu 20.04 LTS), pycodestyle 2.11.1
- numpy 2.0.2, beautifulsoup4 4.12.2, selenium 4.34.2
- Chrome headless, 1920x1080, no sandbox (tasks 6–8)
- Every module and function documented
- Files start with `#!/usr/bin/env python3`, end with a newline, and are executable

## Author

Marcelo Rech — [github.com/marcelorech1983](https://github.com/marcelorech1983)
