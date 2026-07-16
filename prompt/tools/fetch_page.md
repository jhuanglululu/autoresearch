Fetch a web page over HTTP(S) and return its readable text, with the markup,
scripts, and navigation stripped out. Read-only.

Input:
- `url` (required): the URL of the page to fetch.

Limits: the request follows redirects under a timeout, and only HTML/text is
returned (non-text content types are refused). Oversized pages are truncated to
a byte cap — fetch a specific page, not a whole site.

Use `web_search` to find a URL, fetch it here to read it, then archive what
matters with `wiki_capture_source` (origin becomes the URL) so a summary can
cite it.
