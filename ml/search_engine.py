MAX_SEARCH_SUGGESTIONS = 8

def search_movies(query, titles, max_results=MAX_SEARCH_SUGGESTIONS):
    """
    Return ranked autocomplete suggestions: prefix matches first,
    then substring matches, then word-start matches.
    """
    q = query.strip().lower()
    if not q:
        return []

    prefix, contains, word_start = [], [], []
    for title in titles:
        lower_title = title.lower()
        if lower_title.startswith(q):
            prefix.append(title)
        elif q in lower_title:
            contains.append(title)
        elif any(word.startswith(q) for word in lower_title.replace('-', ' ').replace(':', ' ').split()):
            word_start.append(title)

    seen = set()
    results = []
    for group in (prefix, contains, word_start):
        for title in group:
            if title not in seen:
                seen.add(title)
                results.append(title)
                if len(results) >= max_results:
                    return results
    return results


def find_exact_title(query, titles):
    """Return the canonical title when the query matches exactly (case-insensitive)."""
    q = query.strip().lower()
    if not q:
        return None
    for title in titles:
        if title.lower() == q:
            return title
    return None