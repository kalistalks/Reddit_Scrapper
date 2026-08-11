import streamlit as st
import sqlite3
import json
import pandas as pd
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is importable when running via `streamlit run gui/gui.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import get_config

# Configure Streamlit page
st.set_page_config(
    page_title="Marine Bilge Pump Sentiment Explorer",
    page_icon="⚓",
    layout="wide"
)

def _extract_json_from_text(text: str) -> str:
    """Extract JSON payload from markdown-fenced or plain text."""
    if not text:
        return text

    stripped = text.strip()

    # Fenced block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # First object/array fallback
    match = re.search(r"[\{\[].*[\}\]]", stripped, re.DOTALL)
    if match:
        return match.group(0).strip()

    return stripped


@st.cache_data
def load_posts_with_insights(
    db_path: str,
    insights_dir: str,
    provider: str,
    data_version: float
) -> pd.DataFrame:
    """Load posts with insight_processed=1 and join with insight data."""

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Query posts with insights processed
    query = """
    SELECT id, url, title, body, relevance_score, pain_score, emotion_score,
           COALESCE(technical_depth_score, 0) as technical_depth_score,
            COALESCE(sentiment_label, '') as sentiment_label,
            COALESCE(sentiment_score, 0) as sentiment_score,
            COALESCE(sentiment_confidence, 0) as sentiment_confidence,
            COALESCE(sentiment_aspects, '') as sentiment_aspects,
            COALESCE(product_mentioned, '') as product_mentioned,
            COALESCE(complaint_summary, '') as complaint_summary,
            COALESCE(praise_summary, '') as praise_summary,
           subreddit, created_utc, processed_at
    FROM posts
    WHERE insight_processed = 1
    """

    posts_df = pd.read_sql_query(query, conn)
    conn.close()

    # Load insight data from JSONL files
    insights_data = {}
    insights_path = Path(insights_dir)

    for jsonl_file in insights_path.glob("insight_result_*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    custom_id = data.get('custom_id')
                    if not custom_id:
                        continue

                    if provider == "anthropic":
                        # Anthropic format: {"custom_id": "...", "content": "...", "result_type": "..."}
                        if data.get("result_type") != "succeeded":
                            continue
                        content = data.get("content", "")
                        insight_json = json.loads(_extract_json_from_text(content))
                        insights_data[custom_id] = insight_json
                    elif provider == "openai":
                        # OpenAI format
                        if (
                            data.get('response') and
                            data['response'].get('body') and
                            data['response']['body'].get('choices')
                        ):
                            content = data['response']['body']['choices'][0]['message']['content']
                            insight_json = json.loads(_extract_json_from_text(content))
                            insights_data[custom_id] = insight_json

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # Add insight data to posts dataframe
    posts_df['pain_point'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('pain_point', ''))
    posts_df['tags'] = posts_df['id'].map(lambda x: ', '.join(insights_data.get(x, {}).get('tags', [])))
    posts_df['roi_weight'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('roi_weight', 0))
    posts_df['justification'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('justification', ''))
    posts_df['product_opportunity'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('product_opportunity', ''))
    posts_df['affected_audience'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('affected_audience', ''))
    posts_df['existing_alternatives'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('existing_alternatives', ''))
    posts_df['build_complexity'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('build_complexity', ''))
    posts_df['technical_moat'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('technical_moat', ''))
    posts_df['business_model'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('business_model', ''))
    posts_df['business_type'] = posts_df['id'].map(lambda x: insights_data.get(x, {}).get('business_type', ''))
    posts_df['sentiment_magnitude'] = posts_df['sentiment_score'].abs()

    return posts_df

def display_post_card(post: pd.Series):
    """Display a single post as a card."""
    with st.container():
        st.markdown("---")
        # Header with title and scores
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 9])

        with col1:
            st.metric("Sentiment", str(post.get('sentiment_label', '')).title() or "Unknown")
        with col2:
            st.metric("Score", f"{post['sentiment_score']:.2f}")
        with col3:
            st.metric("Confidence", f"{post['sentiment_confidence']:.2f}")
        with col4:
            st.metric("Relevance", f"{post['relevance_score']:.2f}")
        with col5:
            st.metric("Heat", f"{post['sentiment_magnitude']:.2f}")
        with col6:
            if post.get('complaint_summary'):
                st.error(post['complaint_summary'])
            elif post.get('praise_summary'):
                st.success(post['praise_summary'])
            elif post.get('pain_point'):
                st.info(post['pain_point'])

    # Title and tags row
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"[{post['title']}](<{post['url']}>)")
    with col2:
        tags_list = [tag.strip() for tag in post['tags'].split(',') if tag.strip()]
        tags_html = "".join(map(lambda tag: f"<span style='background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 8px; font-size: 11px; margin-right: 4px; display: inline-block; margin-bottom: 2px;'>{tag}</span>", tags_list))
        st.markdown(tags_html, unsafe_allow_html=True)

    # Product opportunity section
    if post.get('product_opportunity'):
        st.success(f"💡 **Market Opportunity:** {post['product_opportunity']}")

    # Add some white space
    st.markdown("")

    with st.expander("🔍 Details"):
        st.markdown("#### 📝 " + post['title'])
        # Truncate long posts
        body_text = post['body'][:500] + "..." if len(post['body']) > 500 else post['body']
        st.markdown(body_text)

        if post['justification']:
            st.markdown("**Justification:**")
            st.markdown(post['justification'])

        if post.get('product_mentioned'):
            st.markdown(f"**Product Mentioned:** {post['product_mentioned']}")

        if post.get('sentiment_aspects'):
            aspects = post['sentiment_aspects']
            if isinstance(aspects, str):
                try:
                    parsed = json.loads(aspects)
                    if isinstance(parsed, list):
                        aspects = parsed
                except Exception:
                    aspects = [item.strip() for item in aspects.split(',') if item.strip()]
            if isinstance(aspects, list) and aspects:
                st.markdown("**Aspects:** " + ", ".join(str(item) for item in aspects))

        # Show additional insight fields
        details_parts = []
        if post.get('affected_audience'):
            details_parts.append(f"**Affected Audience:** {post['affected_audience']}")
        if post.get('business_type'):
            details_parts.append(f"**Business Type:** {post['business_type']}")
        if post.get('existing_alternatives'):
            details_parts.append(f"**Existing Alternatives:** {post['existing_alternatives']}")
        if post.get('build_complexity'):
            details_parts.append(f"**Build Complexity:** {post['build_complexity']}")
        if post.get('technical_moat'):
            details_parts.append(f"**Technical Moat:** {post['technical_moat']}")
        if post.get('business_model'):
            details_parts.append(f"**Business Model:** {post['business_model']}")

        if details_parts:
            st.markdown("---")
            for part in details_parts:
                st.markdown(part)

def main():
    st.title("⚓ Marine Bilge Pump Sentiment Explorer")
    st.markdown("Browse Reddit posts and comments with AI-generated sentiment analysis for marine bilge pumps")

    # Configuration
    cfg = get_config()
    provider = cfg["ai"]["provider"]
    db_path = cfg["database"]["path"]
    insights_dir = cfg.get("paths", {}).get("batch_responses_dir", "data/batch_responses")

    # Check if files exist
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        return

    if not os.path.exists(insights_dir):
        st.error(f"Insights directory not found at {insights_dir}")
        return

    # Build cache-buster from current data file mtimes.
    insight_files = list(Path(insights_dir).glob("insight_result_*.jsonl"))
    latest_insight_mtime = max((f.stat().st_mtime for f in insight_files), default=0.0)
    data_version = max(os.path.getmtime(db_path), latest_insight_mtime)

    # Load data
    with st.spinner("Loading posts and insights..."):
        try:
            df = load_posts_with_insights(db_path, insights_dir, provider, data_version)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    if df.empty:
        st.warning("No posts with processed insights found.")
        return

    st.success(f"Loaded {len(df)} posts with insights (provider: {provider})")

    # Sidebar filters
    st.sidebar.header("🔧 Filters & Sorting")

    # Score range filters
    st.sidebar.subheader("Sentiment Filters")

    # Helper function to create safe sliders
    def create_safe_slider(label: str, values: pd.Series, key: str = None):
        min_val = float(values.min())
        max_val = float(values.max())

        # Handle case where all values are the same
        if min_val == max_val:
            st.sidebar.write(f"**{label}**: {min_val:.2f} (all posts have same value)")
            return (min_val, max_val)

        return st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=0.1,
            key=key
        )

    sentiment_range = create_safe_slider("Sentiment Magnitude Range", df['sentiment_magnitude'], "sentiment_magnitude")
    relevance_range = create_safe_slider("Relevance Score Range", df['relevance_score'], "relevance")
    confidence_range = create_safe_slider("Confidence Range", df['sentiment_confidence'], "confidence")
    pain_range = create_safe_slider("Pain Score Range", df['pain_score'], "pain")

    # Subreddit filter
    subreddits = df['subreddit'].unique().tolist()
    selected_subreddits = st.sidebar.multiselect(
        "Subreddits",
        options=subreddits,
        default=subreddits
    )

    # Sorting options
    st.sidebar.subheader("Sorting")
    sort_by = st.sidebar.selectbox(
        "Sort by",
        options=['sentiment_magnitude', 'sentiment_score', 'relevance_score', 'sentiment_confidence', 'created_utc'],
        index=0
    )

    sort_order = st.sidebar.radio(
        "Sort order",
        options=['Descending', 'Ascending'],
        index=0
    )

    # Apply filters
    filtered_df = df[
        (df['sentiment_magnitude'] >= sentiment_range[0]) &
        (df['sentiment_magnitude'] <= sentiment_range[1]) &
        (df['relevance_score'] >= relevance_range[0]) &
        (df['relevance_score'] <= relevance_range[1]) &
        (df['sentiment_confidence'] >= confidence_range[0]) &
        (df['sentiment_confidence'] <= confidence_range[1]) &
        (df['pain_score'] >= pain_range[0]) &
        (df['pain_score'] <= pain_range[1]) &
        (df['subreddit'].isin(selected_subreddits))
    ]

    # Apply sorting
    ascending = sort_order == 'Ascending'
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # Display results count
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} posts**")

    # Pagination
    posts_per_page = 10
    total_pages = (len(filtered_df) + posts_per_page - 1) // posts_per_page

    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1), index=0)
        start_idx = (page - 1) * posts_per_page
        end_idx = start_idx + posts_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_df

    # Display posts
    for idx, post in page_df.iterrows():
        display_post_card(post)

    # Summary statistics
    if len(filtered_df) > 0:
        st.sidebar.subheader("📈 Summary Stats")
        st.sidebar.metric("Total Posts", len(filtered_df))
        st.sidebar.metric("Avg Sentiment", f"{filtered_df['sentiment_score'].mean():.2f}")
        st.sidebar.metric("Avg Heat", f"{filtered_df['sentiment_magnitude'].mean():.2f}")
        st.sidebar.metric("Avg Relevance", f"{filtered_df['relevance_score'].mean():.2f}")
        st.sidebar.metric("Avg Confidence", f"{filtered_df['sentiment_confidence'].mean():.2f}")

if __name__ == "__main__":
    main()
