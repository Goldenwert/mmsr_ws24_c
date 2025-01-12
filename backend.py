# backend.py

import pandas as pd
import numpy as np
import random
import ast
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, hstack
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
import string
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # irs defined

warnings.filterwarnings('ignore')

# Display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

DATA_DIR = 'data/'

# All file paths
INFORMATION_FILE = os.path.join(DATA_DIR, 'id_information_mmsr.tsv')
GENRES_FILE = os.path.join(DATA_DIR, 'id_genres_mmsr.tsv')
METADATA_FILE = os.path.join(DATA_DIR, 'id_metadata_mmsr.tsv')
TAGS_FILE = os.path.join(DATA_DIR, 'id_tags_dict.tsv')

LYRICS_TFIDF_FILE = os.path.join(DATA_DIR, 'id_lyrics_tf-idf_mmsr.tsv')
LYRICS_BERT_FILE = os.path.join(DATA_DIR, 'id_lyrics_bert_mmsr.tsv')
MFCC_BOW_FILE = os.path.join(DATA_DIR, 'id_mfcc_bow_mmsr.tsv')
SPECTRAL_CONTRAST_FILE = os.path.join(DATA_DIR, 'id_blf_spectralcontrast_mmsr.tsv')
VGG19_FILE = os.path.join(DATA_DIR, 'id_vgg19_mmsr.tsv')
RESNET_FILE = os.path.join(DATA_DIR, 'id_resnet_mmsr.tsv')

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# ================================
# 1. Data Loading Functions
# ================================

@st.cache_data  # irs defined
def load_dataframe(file_path, sep='\t', header='infer', names=None):
    """
    Utility function to load a TSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
        print(f"Loaded DataFrame from '{file_path}' with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)


# Load datasets
information_df = load_dataframe(INFORMATION_FILE)
genres_df = load_dataframe(GENRES_FILE)
metadata_df = load_dataframe(METADATA_FILE)
tags_df = load_dataframe(TAGS_FILE, header=None, names=['id', 'tags_str'])

lyrics_tfidf_df = load_dataframe(LYRICS_TFIDF_FILE)
tfidf_cols = [col for col in lyrics_tfidf_df.columns if col != 'id']
lyrics_tfidf_df.rename(columns={col: f"tfidf_{col}" for col in tfidf_cols}, inplace=True)

bert_df = load_dataframe(LYRICS_BERT_FILE)
bert_feature_columns = [col for col in bert_df.columns if col != 'id']
bert_df.rename(columns={col: f"bert_{col}" for col in bert_feature_columns}, inplace=True)

mfcc_bow_df = load_dataframe(MFCC_BOW_FILE)
mfcc_bow_columns = [col for col in mfcc_bow_df.columns if col != 'id']
mfcc_bow_df.rename(columns={col: f"mfcc_{col}" for col in mfcc_bow_columns}, inplace=True)

spectral_contrast_df = load_dataframe(SPECTRAL_CONTRAST_FILE)
spectral_contrast_columns = [col for col in spectral_contrast_df.columns if col != 'id']
spectral_contrast_df.rename(columns={col: f"spectral_{col}" for col in spectral_contrast_columns}, inplace=True)

vgg19_df = load_dataframe(VGG19_FILE)
vgg19_feature_columns = [col for col in vgg19_df.columns if col != 'id']
vgg19_df.rename(columns={col: f"vgg19_{col}" for col in vgg19_feature_columns}, inplace=True)

resnet_df = load_dataframe(RESNET_FILE)
resnet_feature_columns = [col for col in resnet_df.columns if col != 'id']
resnet_df.rename(columns={col: f"resnet_{col}" for col in resnet_feature_columns}, inplace=True)

# ================================
# 2. Data Preprocessing
# ================================

# Merge information and metadata
catalog_df = pd.merge(information_df, metadata_df[['id', 'popularity']], on='id', how='left')


def parse_genres(genre_str):
    if pd.isnull(genre_str):
        return []
    return [genre.strip().lower() for genre in genre_str.split(',')]


genres_df['genre'] = genres_df['genre'].apply(parse_genres)

# Update catalog_df with genres
catalog_df = pd.merge(catalog_df, genres_df, on='id', how='left')

# Assigning an empty list if genre is NaN
catalog_df['genre'] = catalog_df['genre'].apply(lambda x: x if isinstance(x, list) else [])


def get_top_genre(genres_list):
    if not genres_list:
        return None
    return genres_list[0]


# Determine the top genre
catalog_df['top_genre'] = catalog_df['genre'].apply(get_top_genre)

# Merge tags
catalog_df = pd.merge(catalog_df, tags_df[['id', 'tags_str']], on='id', how='left')

# Assigning an empty string if tags_str is NaN
catalog_df['tags_str'] = catalog_df['tags_str'].fillna('{}')


def parse_tags_and_weights(tag_weight_str):
    """
    Parses the 'tags_str' string into separate lists of tags and weights.
    """
    try:
        # String to a dictionary
        tag_weight_dict = ast.literal_eval(tag_weight_str)
        if isinstance(tag_weight_dict, dict):
            tags = list(tag_weight_dict.keys())
            weights = list(tag_weight_dict.values())
            return tags, weights
        else:
            print(f"Warning: Expected dict, got {type(tag_weight_dict)}")
            return [], []
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing tags: {e}")
        return [], []


# Create 'tags' and 'weights' columns
catalog_df[['tags', 'weights']] = catalog_df.apply(
    lambda row: pd.Series(parse_tags_and_weights(row['tags_str'])),
    axis=1
)


def preprocess_tag(tag):
    """
    Preprocesses a single tag: lowercases, removes punctuation, and lemmatizes.
    """
    tag = tag.lower()
    tag = tag.translate(str.maketrans('', '', string.punctuation))
    tag = lemmatizer.lemmatize(tag)
    return tag


# Apply preprocessing to all tags
catalog_df['processed_tags'] = catalog_df.apply(
    lambda row: [preprocess_tag(tag) for tag in row['tags']],
    axis=1
)

# Exclude genres from tags
genre_tags = set()
for genres in catalog_df['genre']:
    for genre in genres:
        genre_tags.add(preprocess_tag(genre))

# Add 'alternative' and 'indie' to genre_tags
genre_tags.update(['alternative', 'indie'])

print(f"\nGenre Tags to Exclude: {genre_tags}")


def exclude_genre_tags(tags, genre_tags):
    """
    Excludes any tag that matches any genre tag exactly.
    """
    return [tag for tag in tags if tag not in genre_tags]


# Exclusion of genre tags
catalog_df['filtered_processed_tags'] = catalog_df.apply(
    lambda row: exclude_genre_tags(row['processed_tags'], genre_tags),
    axis=1
)

# Filter tags using thresholds
# Weight threshold
min_weight_threshold = 50

# Column 'filtered_processed_tags_final' retains only tags with weight >= threshold
catalog_df['filtered_processed_tags_final'] = catalog_df.apply(
    lambda row: [tag for tag, weight in zip(row['filtered_processed_tags'], row['weights']) if
                 weight >= min_weight_threshold],
    axis=1
)

# Filter weights
catalog_df['filtered_weights_final'] = catalog_df.apply(
    lambda row: [weight for tag, weight in zip(row['filtered_processed_tags'], row['weights']) if
                 weight >= min_weight_threshold],
    axis=1
)

# Sample of the catalog_df to verify the filtering
sample_tracks = catalog_df[['id', 'artist', 'song', 'filtered_processed_tags_final', 'filtered_weights_final']].sample(
    5, random_state=42)

num_no_tags = catalog_df[catalog_df['filtered_processed_tags_final'].apply(len) == 0].shape[0]
total_tracks = catalog_df.shape[0]
percentage_no_tags = (num_no_tags / total_tracks) * 100
catalog_df_filtered = catalog_df[catalog_df['filtered_processed_tags_final'].apply(len) > 0].reset_index(drop=True)


# ================================
# 3. Vectorization
# ================================

def vectorize_tags_tfidf(catalog_df, min_df=1):
    """
    Vectorizes tags using TF-IDF.
    """
    # Convert tags to strings
    catalog_df['tags_str_final_tfidf'] = catalog_df['filtered_processed_tags_final'].apply(lambda tags: ' '.join(tags))

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(min_df=min_df)

    tag_matrix_tfidf = vectorizer.fit_transform(catalog_df['tags_str_final_tfidf'])
    print(f"TF-IDF Tag Matrix Shape: {tag_matrix_tfidf.shape}")

    return tag_matrix_tfidf, vectorizer


# TF-IDF Vectorization on Filtered Catalog
tag_matrix_tfidf, tfidf_vectorizer = vectorize_tags_tfidf(catalog_df_filtered, min_df=1)


def vectorize_tags_binary(catalog_df, min_df=1):
    """
    Vectorizes tags using binary encoding (presence/absence).
    """
    # Convert tags to strings
    catalog_df['tags_str_final_binary'] = catalog_df['filtered_processed_tags_final'].apply(lambda tags: ' '.join(tags))

    # CountVectorizer with binary=True
    vectorizer = CountVectorizer(binary=True, min_df=min_df)

    tag_matrix_binary = vectorizer.fit_transform(catalog_df['tags_str_final_binary'])
    print(f"Binary Tag Matrix Shape: {tag_matrix_binary.shape}")

    return tag_matrix_binary, vectorizer


# Binary Vectorization on Filtered Catalog
tag_matrix_binary, binary_vectorizer = vectorize_tags_binary(catalog_df_filtered, min_df=1)


# ================================
# 4. Feature Matrices for Retrieval Systems
# ================================

def merge_features(catalog_df_filtered, feature_df, prefix):
    """
    Merges a feature DataFrame with catalog_df_filtered on 'id'.
    Renames feature columns with the given prefix.
    """
    feature_df = pd.merge(catalog_df_filtered[['id']], feature_df, on='id', how='left')
    feature_cols = [col for col in feature_df.columns if col != 'id']
    feature_df.rename(columns={col: f"{prefix}_{col}" for col in feature_cols}, inplace=True)
    return feature_df


# Merge and align all feature matrices
merged_bert_df = merge_features(catalog_df_filtered, bert_df, 'bert')
merged_mfcc_bow_df = merge_features(catalog_df_filtered, mfcc_bow_df, 'mfcc')
merged_spectral_contrast_df = merge_features(catalog_df_filtered, spectral_contrast_df, 'spectral')
merged_vgg19_df = merge_features(catalog_df_filtered, vgg19_df, 'vgg19')
merged_resnet_df = merge_features(catalog_df_filtered, resnet_df, 'resnet')

# Convert merged feature DataFrames to numpy arrays
bert_matrix = merged_bert_df.drop('id', axis=1).values
mfcc_bow_matrix = merged_mfcc_bow_df.drop('id', axis=1).values
spectral_contrast_matrix = merged_spectral_contrast_df.drop('id', axis=1).values
vgg19_matrix = merged_vgg19_df.drop('id', axis=1).values
resnet_matrix = merged_resnet_df.drop('id', axis=1).values

# Convert to sparse matrices
bert_matrix = csr_matrix(bert_matrix)
mfcc_bow_matrix = csr_matrix(mfcc_bow_matrix)
spectral_contrast_matrix = csr_matrix(spectral_contrast_matrix)
vgg19_matrix = csr_matrix(vgg19_matrix)
resnet_matrix = csr_matrix(resnet_matrix)

print("\nFeature matrices have been merged and aligned with catalog_df_filtered.")

# Feature matrices for retrieval functions that require them
feature_matrices = {
    'TF-IDF Retrieval': tag_matrix_tfidf,
    'Tag-Based Retrieval': tag_matrix_binary,
    'BERT Retrieval': bert_matrix,
    'MFCC Retrieval': mfcc_bow_matrix,
    'Spectral Contrast Retrieval': spectral_contrast_matrix,
    'VGG19 Retrieval': vgg19_matrix,
    'ResNet Retrieval': resnet_matrix
}


# ================================
# 5. Retrieval Functions with Relevance Definition
# ================================

def random_retrieval(query_track_id, catalog_df, N=10):
    """
    Randomly selects N tracks from the catalog, excluding the query track.
    #irs defined
    """
    if catalog_df is None:
        raise ValueError("catalog_df must be provided for Random Retrieval.")

    # Exclude the query track
    candidates = catalog_df[catalog_df['id'] != query_track_id]

    # Determine the number of tracks to sample
    sample_size = min(N, len(candidates))

    # Randomly sample N tracks
    retrieved_tracks = candidates.sample(n=sample_size, replace=False, random_state=random.randint(0, 1000000))

    return retrieved_tracks.reset_index(drop=True)


def tfidf_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                    relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on TF-IDF cosine similarity.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def bert_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                   relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on BERT cosine similarity.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def mfcc_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                   relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on MFCC Euclidean distance.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index].toarray()

    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)

    distances[query_index] = np.inf  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        distances[~catalog_df.index.isin(relevant_indices)] = np.inf
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set distance to inf for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        distances[~overlap_mask] = np.inf

    top_indices = distances.argsort()[:N]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = [distances[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def spectral_contrast_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                                relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on Spectral Contrast Cosine similarity.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def vgg19_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                    relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on VGG19 Cosine similarity.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def resnet_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                     relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on ResNet Euclidean distance.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index].toarray()

    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)

    distances[query_index] = np.inf  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        distances[~catalog_df.index.isin(relevant_indices)] = np.inf
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set distance to inf for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        distances[~overlap_mask] = np.inf

    top_indices = distances.argsort()[:N]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = [distances[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def tag_based_retrieval_cosine(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                               relevance_definition='top_genre'):
    """
    Retrieves N tracks most similar to the query track based on Cosine similarity of binary tags.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'cosine_similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def early_fusion_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10,
                           relevance_definition='top_genre'):
    """
    Retrieves N tracks using Early Fusion by combining BERT and MFCC feature matrices.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1  # Exclude the query track

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        # Define tag_overlap logic here
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        # Example: Set similarity to -1 for tracks with less than 3 overlapping tags
        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask = catalog_df.apply(has_tag_overlap, axis=1)
        similarities[~overlap_mask] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'aggregated_similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def late_fusion_retrieval(query_track_id, id_to_index, feature_matrices, track_ids, catalog_df, N=10, alpha=0.5,
                          relevance_definition='top_genre'):
    """
    Retrieves N tracks using Late Fusion by combining similarities from MFCC and VGG19 retrieval systems.
    #irs defined
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]

    # MFCC Retrieval
    feature_matrix1 = feature_matrices.get('MFCC Retrieval')
    if feature_matrix1 is None:
        print("Error: 'MFCC Retrieval' feature matrix not found.")
        return pd.DataFrame()
    query_vector1 = feature_matrix1[query_index]
    similarities1 = cosine_similarity(query_vector1, feature_matrix1).flatten()
    similarities1[query_index] = -1

    # VGG19 Retrieval
    feature_matrix2 = feature_matrices.get('VGG19 Retrieval')
    if feature_matrix2 is None:
        print("Error: 'VGG19 Retrieval' feature matrix not found.")
        return pd.DataFrame()
    query_vector2 = feature_matrix2[query_index]
    similarities2 = cosine_similarity(query_vector2, feature_matrix2).flatten()
    similarities2[query_index] = -1

    # Apply relevance_definition filter
    if relevance_definition == 'top_genre':
        query_genre = catalog_df[catalog_df['id'] == query_track_id]['top_genre'].values[0]
        relevant_indices = catalog_df[catalog_df['top_genre'] == query_genre].index
        similarities1[~catalog_df.index.isin(relevant_indices)] = -1
        similarities2[~catalog_df.index.isin(relevant_indices)] = -1
    elif relevance_definition == 'tag_overlap':
        query_tags = set(catalog_df[catalog_df['id'] == query_track_id]['filtered_processed_tags_final'].values[0])

        def has_tag_overlap(row):
            retrieved_tags = set(row['filtered_processed_tags_final'])
            return len(query_tags.intersection(retrieved_tags)) >= 3

        overlap_mask1 = catalog_df.apply(has_tag_overlap, axis=1)
        overlap_mask2 = catalog_df.apply(has_tag_overlap, axis=1)
        similarities1[~overlap_mask1] = -1
        similarities2[~overlap_mask2] = -1

    # Weighted average of similarities
    aggregated_similarities = alpha * similarities1 + (1 - alpha) * similarities2

    top_indices = aggregated_similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [aggregated_similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'aggregated_similarity': retrieved_scores
    })

    # Include 'top_genre' in the merged DataFrame
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


# ================================
# 6. Retrieval Systems Dictionary
# ================================

retrieval_systems = {
    'Random Retrieval': random_retrieval,
    'Tag-Based Retrieval': tag_based_retrieval_cosine,  # Using Cosine Similarity
    'TF-IDF Retrieval': tfidf_retrieval,
    'BERT Retrieval': bert_retrieval,
    'MFCC Retrieval': mfcc_retrieval,
    'Spectral Contrast Retrieval': spectral_contrast_retrieval,
    'VGG19 Retrieval': vgg19_retrieval,
    'ResNet Retrieval': resnet_retrieval,
    'Early Fusion BERT+MFCC Retrieval': early_fusion_retrieval,
    'Late Fusion MFCC+VGG19 Retrieval': late_fusion_retrieval
}

# ================================
# 7. Track IDs and Index Mapping
# ================================

track_ids = catalog_df_filtered['id'].tolist()
id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}

# ================================
# 8. Evaluation Metrics (Optional)
# ================================

# [Keep your existing evaluation metrics and functions here]
# Ensure they handle relevance_definition correctly

# [Your existing evaluation functions here...]

# ================================
# 9. Summary and Additional Functions
# ================================

# [Include any additional utility functions if necessary]
