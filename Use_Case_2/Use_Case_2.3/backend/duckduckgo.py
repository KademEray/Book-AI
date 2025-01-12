import logging

from duckduckgo_search import DDGS


logging.basicConfig(
    filename='Use_Case_2/Use_Case_2.3/backend/backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


logger = logging.getLogger(__name__)

class DuckDuckGoSearch:
    """DuckDuckGo-Suche."""

    def perform_search(self, query):
        try:
            logger.info(f"Suche nach: {query}")
            max_results = 10  # Ziel: 10 relevante Ergebnisse
            collected_results = []
            logger.info(f"Suche nach: {query}")
            # Führe die Suche durch
            results = DDGS().text(query, region='de-de', safesearch='Off', max_results=50)  # Höhere Anzahl abrufen, um zu filtern
            #fallback without region search und query format anpassen
            if not results:
                results = DDGS().text(query, region='wt-wt', safesearch='Off', max_results=50)  # Höhere Anzahl abrufen, um zu filtern
            if not results:
                results = DDGS().text(query, safesearch='Off', max_results=50)  # Höhere Anzahl abrufen, um zu filtern
            
            logger.info(f"Anzahl der Suchergebnisse: {len(results)}")
            logger.info(f"Ergebnisse: {results}")

            if not results:
                logger.warning("Keine Ergebnisse gefunden.")
                return {"results": []}

            # Filtere Ergebnisse mit Titel, Link und Snippet
            for result in results:
                if (
                    "title" in result and result["title"].strip() and
                    "href" in result and result["href"].strip() and
                    "body" in result and result["body"].strip()
                ):
                    collected_results.append({
                        "title": result["title"].strip(),
                        "link": result["href"].strip(),
                        "snippet": result["body"].strip()
                    })

                # Beende die Schleife, sobald 10 Ergebnisse erreicht wurden
                if len(collected_results) >= max_results:
                    break

            # Rückgabe der gesammelten Ergebnisse
            logger.info(f"Gefundene relevante Ergebnisse: {len(collected_results)}")
            return {"results": collected_results}

        except Exception as e:
            return {"error": f"Fehler bei der DuckDuckGo-Suche: {str(e)}"}
        