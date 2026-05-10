#!/usr/bin/env python3
"""Scrape SHL Individual Test Solutions catalog."""
import html
import urllib.request
import re
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_rows(html: str, section: str) -> list:
    idx = html.find(section)
    if idx == -1:
        return []

    next_section_idx = html.find('<th class="custom__table-heading__title">', idx + len(section))
    section_html = html[idx:next_section_idx] if next_section_idx != -1 else html[idx:]

    rows = []
    row_pattern = re.compile(r'<tr data-entity-id="(\d+)">(.*?)</tr>', re.DOTALL)
    for m in row_pattern.finditer(section_html):
        entity_id = m.group(1)
        row_html = m.group(2)

        link = re.search(
            r'href="(/[^"]+product-catalog/view/[^"]+)"[^>]*>\s*([^<]+)', row_html
        )
        if not link:
            continue
        url = BASE_URL + link.group(1).rstrip("/")
        name = html.unescape(link.group(2).strip())

        tds = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
        remote_testing = len(tds) > 1 and '-yes' in tds[1]
        adaptive_irt = len(tds) > 2 and '-yes' in tds[2]

        test_types = re.findall(
            r'<span class="product-catalogue__key"[^>]*>\s*([A-Z])\s*</span>', row_html
        )

        rows.append({
            "entity_id": entity_id,
            "name": name,
            "url": url,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "test_types": test_types,
        })
    return rows


def scrape_catalog() -> list:
    seen_ids = set()
    all_items = []

    start = 0
    while True:
        url = f"{CATALOG_URL}?start={start}"
        print(f"Fetching {url}...")
        html = fetch(url)

        rows = parse_rows(html, "Individual Test Solutions")
        new_rows = [r for r in rows if r["entity_id"] not in seen_ids]
        for r in new_rows:
            seen_ids.add(r["entity_id"])
        all_items.extend(new_rows)
        print(f"  Found {len(rows)} rows, {len(new_rows)} new. Total: {len(all_items)}")

        current_pages = sorted(set(int(x) for x in re.findall(r'start=(\d+)', html)))
        next_pages = [p for p in current_pages if p > start]
        if not next_pages:
            break
        start = min(next_pages)
        time.sleep(0.5)

    return all_items


if __name__ == "__main__":
    items = scrape_catalog()
    print(f"\nTotal Individual Test Solutions: {len(items)}")
    output_path = "/Users/niharikakhanna/shl_assignment/catalog.json"
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved to {output_path}")
    for item in items[:3]:
        print(item)
