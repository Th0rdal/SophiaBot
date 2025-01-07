import requests
import json

def fetch_wikipedia_articles(category_name, max_articles, language="en"):
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    session = requests.Session()

    # Schritt 1: Seiten aus der Kategorie abrufen
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": max_articles
    }

    response = session.get(url=api_url, params=params)
    pages = response.json().get("query", {}).get("categorymembers", [])

    # Schritt 2: Text für jede Seite abrufen
    articles = []
    for page in pages:
        page_id = page["pageid"]
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True
        }
        response = session.get(url=api_url, params=params)
        page_data = response.json().get("query", {}).get("pages", {}).get(str(page_id), {})
        articles.append({"title": page_data.get("title"), "text": page_data.get("extract")})

    # Speichern
    output_file = f"./raw/{category_name}_dump.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(articles)} articles to {output_file}")

if __name__ == "__main__":
    # Kategorie "History" herunterladen
    fetch_wikipedia_articles("History", max_articles=80)
