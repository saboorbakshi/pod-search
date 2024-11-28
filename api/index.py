from fastapi import FastAPI
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from typing import List
import torch

app = FastAPI(docs_url="/api/py/docs", openapi_url="/api/py/openapi.json")

######################################################
# Get chunks and timestamps
######################################################
def get_chunks(transcript_list, max_chunk_length=500):
    """
    Processes the transcript into chunks of text and their corresponding timestamps.

    Args:
        transcript_list (list): List of transcript segments.
        max_chunk_length (int): Maximum number of characters per chunk.

    Returns:
        chunks (list): List of text chunks.
        chunk_timestamps (list): List of lists containing timestamps for each chunk.
    """
    texts = []
    timestamps = []

    for segment in transcript_list:
        texts.append(segment['text'])
        timestamps.append(segment['start'])

    chunks = []
    chunk_timestamps = []
    current_chunk = ''
    current_timestamps = []

    for text, timestamp in zip(texts, timestamps):
        if len(current_chunk) + len(text) + 1 <= max_chunk_length:
            current_chunk += ' ' + text
            current_timestamps.append(timestamp)
        else:
            chunks.append(current_chunk.strip())
            chunk_timestamps.append(current_timestamps)
            current_chunk = text
            current_timestamps = [timestamp]

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        chunk_timestamps.append(current_timestamps)

    return chunks, chunk_timestamps

######################################################
# Generate embeddings
######################################################
def generate_embeddings(texts: List[str], model):
    """
    Generates embeddings for each text chunk using the specified SentenceTransformer model.

    Args:
        texts (list): List of text chunks.

    Returns:
        embeddings (Tensor): Tensor of embedding vectors for each chunk.
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

# Load the model once when the server starts
dimensions = 1024
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

######################################################
# API endpoint to get query embedding
######################################################
@app.get("/api/py/get_query_embedding")
def api_get_query_embedding(query: str):
    """
    API endpoint to get the embedding of a single sentence.

    Args:
        query (str): The query string.

    Returns:
        embedding (list): The embedding vector of the query string.
    """
    embedding = generate_embeddings([query], model)
    embedding_list = embedding.cpu().numpy().tolist()
    return {"query": query, "embedding": embedding_list[0]}

######################################################
# API endpoint to get YouTube embeddings
######################################################
@app.get("/api/py/get_video_embeddings")
def api_get_video_embeddings(video_id: str):
    """
    API endpoint to get embeddings for a YouTube video transcript.

    Args:
        video_id (str): The YouTube video id.

    Returns:
        response (dict): Contains chunks, embeddings, and start_timestamps.
    """
    # Get the transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    # Process the transcript into chunks
    chunks, chunk_timestamps = get_chunks(transcript_list)
    # Generate embeddings for the chunks
    corpus_embeddings = generate_embeddings(chunks, model)
    # Since tensors are not JSON serializable, convert embeddings to list
    embeddings_list = corpus_embeddings.cpu().numpy().tolist()
    return {
        "chunks": chunks,
        "embeddings": embeddings_list,
        "start_timestamps": [timestamps[0] for timestamps in chunk_timestamps]
    }