import asyncio

import aiohttp

from core.exceptions import DomainError
from core.models import RawLatexDocument
from ports.outbound import LatexSourcePort


class AiohttpLatexClient(LatexSourcePort):
    """
    Concurrent HTTP client adapter for ingesting raw LaTeX sources.
    Respects Hexagonal limits by mapping external I/O directly into mathematical
    domain entities without intermediate mutable state.
    """

    async def fetch_sources(self, source_identifiers: list[str]) -> list[RawLatexDocument]:
        """
        Executes concurrent retrieval of all URIs in O(1) blocking time.
        Iterates over coroutines via gather without 'break' or 'continue'.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_single(session, uri) for uri in source_identifiers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_documents: list[RawLatexDocument] = []

            # Deterministic flow control without structural loop breakage
            for result in results:
                if isinstance(result, RawLatexDocument):
                    valid_documents.append(result)

            return valid_documents

    async def _fetch_single(self, session: aiohttp.ClientSession, uri: str) -> RawLatexDocument:
        """
        Retrieves a single payload and instantiates the pure value object.
        """
        try:
            async with session.get(uri) as response:
                if response.status != 200:
                    raise DomainError(f"Failed to fetch {uri}: HTTP {response.status}")
                text = await response.text()
                return RawLatexDocument(raw_text=text)
        except aiohttp.ClientError as e:
            raise DomainError(f"Network error fetching {uri}: {str(e)}") from e
