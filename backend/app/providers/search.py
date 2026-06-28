import httpx

from app.providers.base import WebResult


class TavilySearchProvider:
    name = "tavily"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self.enabled = bool(api_key)

    async def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        if not self.api_key:
            return []
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
        return [
            WebResult(
                title=item.get("title", "Untitled source"),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score"),
            )
            for item in response.json().get("results", [])
            if item.get("url")
        ]
