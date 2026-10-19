import html
import pandas as pd

OVERVIEW_MAX_CHARS = 120

def format_similarity_score(score: float) -> str:
    """Format hybrid score as percentage (clamped 0–99 for display)."""
    pct = max(0, min(99, round(float(score) * 100)))
    return f"Similarity: {pct}%"

def format_overview_html(overview):
    """
    Render a short overview (~120 chars) with an expandable Read More section
    when the full text is longer.
    """
    if overview is None or (isinstance(overview, float) and pd.isna(overview)):
        return '<div class="movie-overview">N/A</div>'

    text = str(overview).strip()
    if not text:
        return '<div class="movie-overview">N/A</div>'

    safe_text = html.escape(text)

    if len(text) <= OVERVIEW_MAX_CHARS:
        return f'<div class="movie-overview">{safe_text}</div>'

    preview = text[:OVERVIEW_MAX_CHARS]
    if ' ' in preview:
        preview = preview.rsplit(' ', 1)[0]
    preview = preview.rstrip('.,;:')
    safe_preview = html.escape(preview)

    return f'''<div class="movie-overview">
        <details class="overview-toggle">
            <summary class="overview-summary">
                <span class="overview-short">{safe_preview}...</span>
                <span class="overview-read-more-btn">Read More</span>
            </summary>
            <span class="overview-expanded">{safe_text}</span>
        </details>
    </div>'''