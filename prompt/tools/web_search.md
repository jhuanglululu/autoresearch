Search the web (DuckDuckGo) for pages, papers, and discussion relevant to your
assignment. Read-only.

Returns a ranked list of results, each with a title, URL, and a short snippet.
The snippet is a teaser, not evidence — follow up with `fetch_page` to read a
promising result in full before you rely on or cite it.

Input:
- `query` (required): the search query (topic or keywords).
- `max_results` (optional): number of results to return (default 5).

To turn a result into wiki evidence, fetch it, then archive the text with
`wiki_capture_source` and cite it from a summary.
