# app.py

import streamlit as st
import pandas as pd
import backend  # irs defined
import time
from scipy.sparse import hstack  # irs defined
import os

# ================================
# 1. Initialize Session State for History and Results
# ================================

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'results' not in st.session_state:
    st.session_state['results'] = {}

# ================================
# 2. Title and Description
# ================================

st.title("Music Information Retrieval System")
st.markdown("""
Welcome to the Music Information Retrieval System. Select a track and a retrieval system to discover similar songs based on your chosen relevance definition.
""")

# ================================
# 3. Sidebar: Track Selection and Retrieval System Selection
# ================================

st.sidebar.header("Selection Panel")

# Track Selection with Search Capability
st.sidebar.subheader("Select a Track")

# Create a mapping from display name to track ID
track_display_to_id = {
    f"{row['artist']} - {row['song']}": row['id']
    for idx, row in backend.catalog_df_filtered[['id', 'artist', 'song']].iterrows()
}

# Get list of display names
track_display_names = list(track_display_to_id.keys())

# Implement search functionality
search_query = st.sidebar.text_input("Search for a Track:")

if search_query:
    # Filter tracks based on search query (case-insensitive)
    filtered_tracks = [name for name in track_display_names if search_query.lower() in name.lower()]
else:
    filtered_tracks = track_display_names

# Selectbox for track selection
selected_track_display = st.sidebar.selectbox(
    "Choose a Track:",
    filtered_tracks
)

# Get the corresponding track ID
selected_track_id = track_display_to_id[selected_track_display]

# IR Systems Selection (Multi-select)
st.sidebar.subheader("Select Information Retrieval Systems")
retrieval_system_options = list(backend.retrieval_systems.keys())
selected_retrieval_systems = st.sidebar.multiselect(
    "Choose IR Systems:",
    retrieval_system_options
)

# ================================
# 4. Select Relevance Definition
# ================================

st.sidebar.subheader("Relevance Definition")
relevance_options = ['Top Genre Relevance', 'Tag Overlap Relevance']
selected_relevance = st.sidebar.radio(
    "Choose Relevance Type:",
    relevance_options
)

# Map selection to backend relevance definitions
relevance_mapping = {
    'Top Genre Relevance': 'top_genre',
    'Tag Overlap Relevance': 'tag_overlap'
}
selected_relevance_def = relevance_mapping[selected_relevance]

# ================================
# 5. Load YouTube URLs and Create Mapping
# ================================

# Load YouTube URLs from TSV file
URLS_FILE = os.path.join('data/', 'id_url_mmsr.tsv')  # irs defined


@st.cache_data  # irs defined
def load_urls(file_path):
    """
    Loads YouTube URLs from a TSV file and returns a dictionary mapping track IDs to URLs.
    """
    try:
        urls_df = pd.read_csv(file_path, sep='\t', header=0)
        url_mapping = pd.Series(urls_df.url.values, index=urls_df.id).to_dict()
        return url_mapping
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return {}


url_mapping = load_urls(URLS_FILE)

# ================================
# 6. Main Section: Retrieve Results
# ================================

st.header("Retrieval Results")

# Display Query Track's Genre
query_track_genre = \
backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == selected_track_id]['top_genre'].values[0]
st.subheader(f"Query Track: {selected_track_display}")
st.write(f"**Top Genre:** {query_track_genre.capitalize() if isinstance(query_track_genre, str) else 'N/A'}")

# Retrieve Results Button
if st.sidebar.button("Retrieve Results"):
    if not selected_retrieval_systems:
        st.sidebar.error("Please select at least one IR system.")
    else:
        with st.spinner('Retrieving results...'):
            results = {}
            for system in selected_retrieval_systems:
                retrieval_func = backend.retrieval_systems[system]
                try:
                    if system == 'Late Fusion MFCC+VGG19 Retrieval':
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            id_to_index=backend.id_to_index,
                            feature_matrices={
                                'MFCC Retrieval': backend.feature_matrices['MFCC Retrieval'],
                                'VGG19 Retrieval': backend.feature_matrices['VGG19 Retrieval']
                            },
                            track_ids=backend.track_ids,
                            catalog_df=backend.catalog_df_filtered,
                            N=10,
                            alpha=0.5,
                            relevance_definition=selected_relevance_def  # irs defined
                        )
                    elif system == 'Early Fusion BERT+MFCC Retrieval':
                        # Combine BERT and MFCC features using hstack
                        combined_feature_matrix = hstack([
                            backend.feature_matrices['BERT Retrieval'],
                            backend.feature_matrices['MFCC Retrieval']
                        ]).tocsr()
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            id_to_index=backend.id_to_index,
                            feature_matrix=combined_feature_matrix,
                            track_ids=backend.track_ids,
                            catalog_df=backend.catalog_df_filtered,
                            N=10,
                            relevance_definition=selected_relevance_def  # irs defined
                        )
                    elif system == 'Random Retrieval':
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            catalog_df=backend.catalog_df_filtered,
                            N=10
                        )
                    else:
                        feature_matrix = backend.feature_matrices.get(system)
                        if feature_matrix is not None:
                            retrieved = retrieval_func(
                                query_track_id=selected_track_id,
                                id_to_index=backend.id_to_index,
                                feature_matrix=feature_matrix,
                                track_ids=backend.track_ids,
                                catalog_df=backend.catalog_df_filtered,
                                N=10,
                                relevance_definition=selected_relevance_def  # irs defined
                            )
                        else:
                            st.error(f"Feature matrix for {system} not found.")
                            retrieved = pd.DataFrame()

                    # Store the retrieved DataFrame
                    results[system] = retrieved
                except Exception as e:
                    st.error(f"Error retrieving results for {system}: {e}")
                    results[system] = pd.DataFrame()

            # Store 'results' in session_state for Qualitative Analysis
            st.session_state['results'] = results

            # Display Results
            for system, df in results.items():
                st.subheader(f"Results from {system}")
                if df.empty:
                    st.write("No results found.")
                else:
                    # Determine which columns to display based on retrieval system
                    display_columns = ['artist', 'song', 'album_name', 'top_genre']
                    if 'similarity' in df.columns:
                        display_columns.append('similarity')
                    if 'distance' in df.columns:
                        display_columns.append('distance')
                    if 'cosine_similarity' in df.columns:
                        display_columns.append('cosine_similarity')
                    if 'aggregated_similarity' in df.columns:
                        display_columns.append('aggregated_similarity')

                    # Add YouTube URL column if available
                    df['YouTube URL'] = df['id'].map(url_mapping)

                    # Create a new DataFrame for display with embedded videos
                    display_df = df[display_columns + ['YouTube URL']].copy()


                    # Replace YouTube URL with embed code
                    def embed_video(url):
                        if pd.isnull(url):
                            return "No Video Available"
                        else:
                            video_id = url.split("v=")[-1]
                            return f'<iframe width="300" height="169" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'


                    display_df['Video'] = display_df['YouTube URL'].apply(embed_video)

                    # Drop the original URL column
                    display_df = display_df.drop(columns=['YouTube URL'])

                    # Reorder columns to place Video at the end
                    display_df = display_df[display_columns + ['Video']]

                    # Display each track with its video
                    for idx, row in display_df.iterrows():
                        st.markdown(f"**{row['artist']} - {row['song']}**")
                        st.write(f"**Album:** {row['album_name']}")
                        st.write(
                            f"**Top Genre:** {row['top_genre'].capitalize() if isinstance(row['top_genre'], str) else 'N/A'}")
                        if 'similarity' in row and not pd.isnull(row['similarity']):
                            st.write(f"**Similarity Score:** {row['similarity']:.4f}")
                        if 'cosine_similarity' in row and not pd.isnull(row['cosine_similarity']):
                            st.write(f"**Cosine Similarity:** {row['cosine_similarity']:.4f}")
                        if 'distance' in row and not pd.isnull(row['distance']):
                            st.write(f"**Distance Score:** {row['distance']:.4f}")
                        if 'aggregated_similarity' in row and not pd.isnull(row['aggregated_similarity']):
                            st.write(f"**Aggregated Similarity:** {row['aggregated_similarity']:.4f}")
                        # Embed the YouTube video
                        if row['Video'] != "No Video Available":
                            st.markdown(row['Video'], unsafe_allow_html=True)
                        else:
                            st.write("No Video Available")
                        st.markdown("---")

            # Update History
            retrieval_record = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'track': selected_track_display,
                'systems': selected_retrieval_systems,
                'relevance': selected_relevance_def,  # irs defined
                'results': {k: v.to_dict(orient='records') for k, v in results.items()}
            }
            st.session_state['history'].append(retrieval_record)

# ================================
# 7. Sidebar: History Management
# ================================

st.sidebar.subheader("History")

# View History Button
if st.sidebar.button("View History"):
    history = st.session_state['history']
    if not history:
        st.sidebar.info("No retrieval history available.")
    else:
        for record in reversed(history):
            with st.expander(f"{record['timestamp']} - {record['track']}"):
                st.write(f"**Selected IR Systems:** {', '.join(record['systems'])}")
                st.write(f"**Relevance Definition:** {record['relevance']}")
                for system, retrieved_list in record['results'].items():
                    st.markdown(f"### {system}")
                    if not retrieved_list:
                        st.write("No results found.")
                    else:
                        # Convert list of dicts back to DataFrame
                        df = pd.DataFrame(retrieved_list)
                        display_columns = ['artist', 'song', 'album_name', 'top_genre']
                        if 'similarity' in df.columns:
                            display_columns.append('similarity')
                        if 'distance' in df.columns:
                            display_columns.append('distance')
                        if 'cosine_similarity' in df.columns:
                            display_columns.append('cosine_similarity')
                        if 'aggregated_similarity' in df.columns:
                            display_columns.append('aggregated_similarity')

                        # Add YouTube URL column if available
                        df['YouTube URL'] = df['id'].map(url_mapping)

                        # Create a new DataFrame for display with embedded videos
                        display_df = df[display_columns + ['YouTube URL']].copy()


                        # Replace YouTube URL with embed code
                        def embed_video(url):
                            if pd.isnull(url):
                                return "No Video Available"
                            else:
                                video_id = url.split("v=")[-1]
                                return f'<iframe width="300" height="169" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'


                        display_df['Video'] = display_df['YouTube URL'].apply(embed_video)

                        # Drop the original URL column
                        display_df = display_df.drop(columns=['YouTube URL'])

                        # Reorder columns to place Video at the end
                        display_df = display_df[display_columns + ['Video']]

                        # Display each track with its video
                        for idx, row in display_df.iterrows():
                            st.markdown(f"**{row['artist']} - {row['song']}**")
                            st.write(f"**Album:** {row['album_name']}")
                            st.write(
                                f"**Top Genre:** {row['top_genre'].capitalize() if isinstance(row['top_genre'], str) else 'N/A'}")
                            if 'similarity' in row and not pd.isnull(row['similarity']):
                                st.write(f"**Similarity Score:** {row['similarity']:.4f}")
                            if 'cosine_similarity' in row and not pd.isnull(row['cosine_similarity']):
                                st.write(f"**Cosine Similarity:** {row['cosine_similarity']:.4f}")
                            if 'distance' in row and not pd.isnull(row['distance']):
                                st.write(f"**Distance Score:** {row['distance']:.4f}")
                            if 'aggregated_similarity' in row and not pd.isnull(row['aggregated_similarity']):
                                st.write(f"**Aggregated Similarity:** {row['aggregated_similarity']:.4f}")
                            # Embed the YouTube video
                            if row['Video'] != "No Video Available":
                                st.markdown(row['Video'], unsafe_allow_html=True)
                            else:
                                st.write("No Video Available")
                            st.markdown("---")

# Clear History Button
if st.sidebar.button("Clear History"):
    st.session_state['history'] = []
    st.sidebar.success("History cleared.")

# ================================
# 8. Main Section: Qualitative Analysis
# ================================

st.header("Qualitative Analysis")

# Analyze Retrieval Results Button
if selected_retrieval_systems and st.button("Analyze Retrieval Results"):
    if not st.session_state['results']:
        st.info("No retrieval results available for analysis.")
    else:
        with st.spinner('Conducting qualitative analysis...'):
            results = st.session_state['results']
            for system in selected_retrieval_systems:
                # Skip Random Retrieval in qualitative analysis
                if system == 'Random Retrieval':
                    st.subheader(f"Analysis for {system}")
                    st.write("Qualitative analysis is not applicable for Random Retrieval.")
                    continue

                df = results.get(system)
                if df is not None and not df.empty:
                    st.subheader(f"Analysis for {system}")

                    # Display Retrieved Tracks with Relevant Metrics
                    st.write("### Retrieved Tracks:")
                    display_columns = ['artist', 'song', 'album_name', 'top_genre']
                    if 'similarity' in df.columns:
                        display_columns.append('similarity')
                    if 'distance' in df.columns:
                        display_columns.append('distance')
                    if 'cosine_similarity' in df.columns:
                        display_columns.append('cosine_similarity')
                    if 'aggregated_similarity' in df.columns:
                        display_columns.append('aggregated_similarity')

                    st.dataframe(df[display_columns].fillna('').head(10))

                    # Example of how to analyze why tracks were retrieved
                    # You can expand this section based on available metadata and features
                    st.write("### Analysis:")
                    for idx, row in df.iterrows():
                        retrieved_track_id = row['id']
                        # Get query track's genre
                        query_genre = \
                        backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == selected_track_id][
                            'top_genre'].values[0]
                        # Get retrieved track's genre
                        retrieved_genre = \
                        backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == retrieved_track_id][
                            'top_genre'].values[0]

                        # Normalize genres for comparison
                        normalized_query_genre = query_genre.strip().lower() if isinstance(query_genre, str) else ''
                        normalized_retrieved_genre = retrieved_genre.strip().lower() if isinstance(retrieved_genre,
                                                                                                   str) else ''

                        # Get similarity score
                        if 'similarity' in row and not pd.isnull(row['similarity']):
                            similarity_score = row['similarity']
                            similarity_metric = 'Similarity Score'
                        elif 'cosine_similarity' in row and not pd.isnull(row['cosine_similarity']):
                            similarity_score = row['cosine_similarity']
                            similarity_metric = 'Cosine Similarity'
                        elif 'aggregated_similarity' in row and not pd.isnull(row['aggregated_similarity']):
                            similarity_score = row['aggregated_similarity']
                            similarity_metric = 'Aggregated Similarity'
                        elif 'distance' in row and not pd.isnull(row['distance']):
                            similarity_score = row['distance']
                            similarity_metric = 'Distance Score'
                        else:
                            similarity_score = None
                            similarity_metric = 'N/A'

                        # Determine genre match
                        genre_match = 'Yes' if normalized_retrieved_genre == normalized_query_genre else 'No'

                        # Formulate the analysis statement only if similarity_score exists
                        if similarity_score is not None:
                            analysis_statement = f"**{row['artist']} - {row['song']}** | **Genre Match:** {genre_match} | **{similarity_metric}:** {similarity_score:.4f}"
                        else:
                            analysis_statement = f"**{row['artist']} - {row['song']}** | **Genre Match:** {genre_match} | **Similarity Metric:** N/A"

                        st.write(analysis_statement)

                    # Optional: You can add more detailed analysis based on other metadata or features
                else:
                    st.write(f"No results to analyze for {system}.")
