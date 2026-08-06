"""One-off script: embeds and stores the demo knowledge base into MongoDB.

Run with: uv run python scripts/ingest_rag_docs.py
"""

import asyncio

from app.services.rag_service import store_chunk

DOCUMENTS = [
    {
        "source": "trend-report-sneakers-q1-2026",
        "text": (
            "Sneaker colorways in Q1 2026 have shifted heavily toward muted earth "
            "tones — olive, sand, and rust — replacing the neon palettes that "
            "dominated 2024. Limited-edition collaborations between streetwear "
            "labels and heritage sportswear brands are driving the highest resale "
            "premiums, with some pairs reselling at 3-4x retail within the first week."
        ),
    },
    {
        "source": "trend-report-streetwear-q1-2026",
        "text": (
            "Streetwear silhouettes are trending oversized again after several "
            "seasons of slim-fit dominance. Cargo pants, boxy graphic tees, and "
            "utility vests are the top-searched apparel categories this quarter. "
            "Gorpcore — outdoor/technical wear repurposed as everyday fashion — "
            "continues to grow, especially among 18-24 year olds."
        ),
    },
    {
        "source": "trend-report-tech-q1-2026",
        "text": (
            "Compact AI wearables are the fastest-growing tech category this year, "
            "with pin-style and pendant form factors outpacing smart glasses in "
            "search volume. Battery life and always-listening privacy concerns "
            "remain the top two complaints in product reviews. Foldable phone "
            "adoption has plateaued outside of premium markets."
        ),
    },
    {
        "source": "company-background",
        "text": (
            "TrendGully was founded to help brands spot emerging consumer trends "
            "before they go mainstream, combining social listening data with "
            "retail sales signals. The company focuses primarily on fashion, "
            "sneakers, and lifestyle tech categories, serving brand strategists "
            "and merchandising teams."
        ),
    },
    {
        "source": "trend-report-social-video",
        "text": (
            "Short-form video remains the dominant discovery channel for new "
            "products, but engagement is shifting from pure entertainment content "
            "toward 'search-style' videos where users treat platforms like a "
            "product search engine. This has increased the importance of clear "
            "product naming and on-screen text in trend-driving content."
        ),
    },
]


async def main():
    for doc in DOCUMENTS:
        was_new = await store_chunk(text=doc["text"], source=doc["source"])
        print(f"{'Stored' if was_new else 'Skipped (already exists)'}: {doc['source']}")


if __name__ == "__main__":
    asyncio.run(main())
