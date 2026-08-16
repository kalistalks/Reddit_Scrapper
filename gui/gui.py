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
    results_dir: str,
    provider: str,
    data_version: float
) -> pd.DataFrame:
    """Load scored posts and join with any batch result data available."""

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

        # Query posts that have at least filter-stage scores.
    query = """
        SELECT id, url, title, body, relevance_score, emotion_score, pain_score,
            implementability_score,
            COALESCE(technical_depth_score, 0) as technical_depth_score,
            COALESCE(tags, '') as tags,
            COALESCE(roi_weight, 0) as roi_weight,
            COALESCE(sentiment_label, '') as sentiment_label,
            COALESCE(sentiment_score, 0) as sentiment_score,
            COALESCE(sentiment_confidence, 0) as sentiment_confidence,
            COALESCE(sentiment_aspects, '') as sentiment_aspects,
            COALESCE(product_mentioned, '') as product_mentioned,
            COALESCE(complaint_summary, '') as complaint_summary,
            COALESCE(praise_summary, '') as praise_summary,
            subreddit, created_utc, processed_at,
            COALESCE(insight_processed, 0) as insight_processed
    FROM posts
        WHERE relevance_score IS NOT NULL OR sentiment_score IS NOT NULL
    """

    posts_df = pd.read_sql_query(query, conn)
    conn.close()

    # Load batch result data from JSONL files.
    results_data = {}
    results_path = Path(results_dir)

    result_files = list(results_path.glob("insight_result_*.jsonl"))
    if not result_files:
        result_files = list(results_path.glob("filter_result_*.jsonl"))

    for jsonl_file in result_files:
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
                        result_json = json.loads(_extract_json_from_text(content))
                        results_data[custom_id] = result_json
                    elif provider == "openai":
                        # OpenAI format
                        if (
                            data.get('response') and
                            data['response'].get('body') and
                            data['response']['body'].get('choices')
                        ):
                            content = data['response']['body']['choices'][0]['message']['content']
                            result_json = json.loads(_extract_json_from_text(content))
                            results_data[custom_id] = result_json

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # Add batch result data to the dataframe.
    posts_df['summary'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('summary', ''))
    posts_df['tags_from_result'] = posts_df['id'].map(lambda x: ', '.join(results_data.get(x, {}).get('tags', [])))
    posts_df['justification'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('justification', ''))
    posts_df['product_opportunity'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('product_opportunity', ''))
    posts_df['affected_audience'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('affected_audience', ''))
    posts_df['existing_alternatives'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('existing_alternatives', ''))
    posts_df['build_complexity'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('build_complexity', ''))
    posts_df['technical_moat'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('technical_moat', ''))
    posts_df['business_model'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('business_model', ''))
    posts_df['business_type'] = posts_df['id'].map(lambda x: results_data.get(x, {}).get('business_type', ''))
    posts_df['tags_from_db'] = posts_df['tags']
    posts_df['tags'] = posts_df['tags'].where(posts_df['tags'].astype(str).str.len() > 0, posts_df['tags_from_result'])
    posts_df['pain_point'] = posts_df['summary']
    posts_df['sentiment_magnitude'] = posts_df['sentiment_score'].abs()
    posts_df['has_deep_insights'] = posts_df['sentiment_label'].astype(str).str.len().gt(0) & (posts_df['sentiment_score'] != 0)

    return posts_df

def display_post_card(post: pd.Series):
    """Display a single post as a card."""
    with st.container():
        st.markdown("---")
        # Header with title and scores
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 9])

        with col1:
            if post.get('has_deep_insights'):
                st.metric("Sentiment", str(post.get('sentiment_label', '')).title() or "Unknown")
            else:
                st.metric("Relevance", f"{post['relevance_score']:.2f}")
        with col2:
            if post.get('has_deep_insights'):
                st.metric("Score", f"{post['sentiment_score']:.2f}")
            else:
                st.metric("Emotion", f"{post['emotion_score']:.2f}")
        with col3:
            if post.get('has_deep_insights'):
                st.metric("Confidence", f"{post['sentiment_confidence']:.2f}")
            else:
                st.metric("Pain", f"{post['pain_score']:.2f}")
        with col4:
            st.metric("Implement", f"{post['implementability_score']:.2f}")
        with col5:
            st.metric("Tech Depth", f"{post['technical_depth_score']:.2f}")
        with col6:
            if post.get('has_deep_insights'):
                if post.get('complaint_summary'):
                    st.error(post['complaint_summary'])
                elif post.get('praise_summary'):
                    st.success(post['praise_summary'])
                elif post.get('pain_point'):
                    st.info(post['pain_point'])
            else:
                st.info(post.get('summary', '') or post.get('pain_point', ''))

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

        if post.get('has_deep_insights') and post.get('sentiment_aspects'):
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

        if not post.get('has_deep_insights'):
            st.markdown("---")
            st.caption("Showing filter-stage results from the existing batch output. Run the deeper analysis later only if you want sentiment and product-opportunity fields.")

def main():
    st.title("⚓ Marine Bilge Pump Sentiment Explorer")
    st.markdown("Browse Reddit posts and comments with AI-generated sentiment analysis for marine bilge pumps")

    # Configuration
    cfg = get_config()
    provider = cfg["ai"]["provider"]
    db_path = cfg["database"]["path"]
    results_dir = cfg.get("paths", {}).get("batch_responses_dir", "data/batch_responses")

    # Check if files exist
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        return

    if not os.path.exists(results_dir):
        st.error(f"Results directory not found at {results_dir}")
        return

    # Build cache-buster from current data file mtimes.
    result_files = list(Path(results_dir).glob("insight_result_*.jsonl")) + list(Path(results_dir).glob("filter_result_*.jsonl"))
    latest_result_mtime = max((f.stat().st_mtime for f in result_files), default=0.0)
    data_version = max(os.path.getmtime(db_path), latest_result_mtime)

    # Load data
    with st.spinner("Loading posts and insights..."):
        try:
            df = load_posts_with_insights(db_path, results_dir, provider, data_version)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    if df.empty:
        st.warning("No scored posts found in the database or batch results.")
        return

    insight_count = int(df['has_deep_insights'].sum()) if 'has_deep_insights' in df.columns else 0
    if insight_count:
        st.success(f"Loaded {len(df)} scored posts, including {insight_count} posts with deep insights (provider: {provider})")
    else:
        st.info(f"Loaded {len(df)} scored posts from the existing filter results. Deep insights are not processed yet, so the dashboard is using the filter-stage data.")

    # Sidebar filters
    st.sidebar.header("🔧 Filters & Sorting")

    # Score range filters
    st.sidebar.subheader("Score Filters")

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

    relevance_range = create_safe_slider("Relevance Score Range", df['relevance_score'], "relevance")
    emotion_range = create_safe_slider("Emotion Score Range", df['emotion_score'], "emotion")
    pain_range = create_safe_slider("Pain Score Range", df['pain_score'], "pain")
    implementability_range = create_safe_slider("Implementability Range", df['implementability_score'], "implementability")
    tech_depth_range = create_safe_slider("Technical Depth Range", df['technical_depth_score'], "tech_depth")

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
        options=['relevance_score', 'emotion_score', 'pain_score', 'implementability_score', 'technical_depth_score', 'created_utc'],
        index=0
    )

    sort_order = st.sidebar.radio(
        "Sort order",
        options=['Descending', 'Ascending'],
        index=0
    )

    # Apply filters
    filtered_df = df[
        (df['relevance_score'] >= relevance_range[0]) &
        (df['relevance_score'] <= relevance_range[1]) &
        (df['emotion_score'] >= emotion_range[0]) &
        (df['emotion_score'] <= emotion_range[1]) &
        (df['pain_score'] >= pain_range[0]) &
        (df['pain_score'] <= pain_range[1]) &
        (df['implementability_score'] >= implementability_range[0]) &
        (df['implementability_score'] <= implementability_range[1]) &
        (df['technical_depth_score'] >= tech_depth_range[0]) &
        (df['technical_depth_score'] <= tech_depth_range[1]) &
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
        st.sidebar.metric("Avg Relevance", f"{filtered_df['relevance_score'].mean():.2f}")
        st.sidebar.metric("Avg Emotion", f"{filtered_df['emotion_score'].mean():.2f}")
        st.sidebar.metric("Avg Pain", f"{filtered_df['pain_score'].mean():.2f}")
        st.sidebar.metric("Avg Tech Depth", f"{filtered_df['technical_depth_score'].mean():.2f}")

if __name__ == "__main__":
    main()
