import xml.etree.ElementTree as ET
import gzip

def parse_pubmed_file(filename):
    with gzip.open(filename, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()

    articles = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle") or ""
        abstract = article.findtext(".//AbstractText") or ""

        if title and abstract:
            articles.append({
                "title": title.strip(),
                "abstract": abstract.strip()
            })

    return articles

# Test it
if __name__ == "__main__":
    articles = parse_pubmed_file("pubmed25n0001.xml.gz")
    print(f"Parsed {len(articles)} articles.")
    print(articles[0])  # Check a sample