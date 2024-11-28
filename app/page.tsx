'use client'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useState, useRef, useEffect } from 'react'
import { cosineSimilarity } from 'ai'
import YouTube from 'react-youtube'

interface SearchResult {
  chunk: string
  timestamp: number
  similarity: number
}

const semanticSearch = (
  queryEmbedding: number[],
  corpusEmbeddings: number[][],
  chunks: string[],
  timestamps: number[],
  topK: number = 5
): SearchResult[] => {
  console.log(timestamps.length)
  // Calculate similarity scores for all chunks
  const similarities = corpusEmbeddings.map(embedding =>
    cosineSimilarity(queryEmbedding, embedding)
  )

  // Create array of indices and sort by similarity score
  const indices = similarities.map((_, i) => i)
  indices.sort((a, b) => similarities[b] - similarities[a])

  // Return top K results
  return indices.slice(0, topK).map(index => ({
    chunk: chunks[index],
    timestamp: timestamps[index],
    similarity: similarities[index],
  }))
}

const getYouTubeVideoId = (url: string): string | null => {
  try {
    const urlObj = new URL(url)
    if (urlObj.hostname.includes('youtube.com')) {
      return urlObj.searchParams.get('v')
    } else if (urlObj.hostname === 'youtu.be') {
      return urlObj.pathname.slice(1)
    }
    return null
  } catch {
    return null
  }
}

export default function Home() {
  const [url, setUrl] = useState('')
  const [query, setQuery] = useState('')
  const [videoId, setVideoId] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])

  const [chunks, setChunks] = useState<string[]>([])
  const [timestamps, setTimestamps] = useState<number[]>([])
  const [corpusEmbeddings, setCorpusEmbeddings] = useState<number[][]>([])

  // Create a ref to store the player instance
  const playerRef = useRef(null)

  // Handler called when the YouTube player is ready
  const onPlayerReady = (event: { target: null }) => {
    // Store the player instance in the ref
    playerRef.current = event.target
  }

  // Function to seek to a particular timestamp
  const seekToTimestamp = (seconds: number) => {
    if (playerRef.current && 'seekTo' in playerRef.current) {
      ;(playerRef.current as any).seekTo(seconds, true) // 'true' for allowing seeking ahead of buffered data
    }
  }

  const performSemanticSearch = async () => {
    if (!query.trim()) return

    try {
      const response = await fetch(`/api/py/get_query_embedding?query=${encodeURIComponent(query)}`)
      if (!response.ok) {
        throw new Error('Failed to fetch query embedding')
      }
      const data = await response.json()
      const results = semanticSearch(data.embedding, corpusEmbeddings, chunks, timestamps)
      setSearchResults(results)
    } catch (error) {
      console.error('Error performing semantic search:', error)
    }
  }

  // Fetch video data when videoId changes
  useEffect(() => {
    setSearchResults([])
    setIsLoading(true)
    if (videoId) {
      const fetchVideoData = async () => {
        try {
          const response = await fetch(`/api/py/get_video_embeddings?video_id=${videoId}`)
          if (!response.ok) {
            throw new Error('Failed to fetch video data')
          }
          const data = await response.json()
          setChunks(data.chunks)
          setCorpusEmbeddings(data.embeddings)
          setTimestamps(data.start_timestamps)
        } catch (error) {
          console.error('Error fetching video data:', error)
        }
        setIsLoading(false)
      }
      fetchVideoData()
    }
  }, [videoId])

  return (
    <div className="max-w-[1600px] mx-auto">
      <main className="flex flex-col min-h-screen items-center p-8 py-12 gap-y-12">
        <div className="text-4xl">PodSearch</div>
        <div className="flex items-center space-x-2">
          <Input
            className="w-[600px]"
            placeholder="Paste YouTube url"
            value={url}
            onChange={e => setUrl(e.target.value)}
          />
          <Button
            onClick={() => {
              const videoId = getYouTubeVideoId(url)
              if (videoId) {
                console.log(videoId)
                setVideoId(videoId)
              }
            }}
          >
            Generate Embeddings
          </Button>
        </div>

        {videoId && (
          <div className="flex w-full justify-between items-start gap-10">
            <div className="w-3/5 aspect-video rounded-lg overflow-hidden">
              <YouTube
                videoId={videoId}
                opts={{
                  width: '100%',
                  height: '100%',
                  playerVars: {
                    autoplay: 0,
                  },
                }}
                onReady={onPlayerReady}
                className="h-full"
              />
            </div>

            {isLoading ? (
              <div className="w-2/5 aspect-video flex flex-col justify-center items-center gap-4">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-800" />
                <p className="text-gray-600">Generating embeddings...</p>
              </div>
            ) : (
              <div className="flex flex-col w-2/5">
                <div className="flex w-full items-center space-x-2">
                  <Input
                    placeholder="Ask a question"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                  />
                  <Button onClick={performSemanticSearch}>Go</Button>
                </div>

                {searchResults.map((result, index) => (
                  <div
                    key={index}
                    className="flex flex-col mt-4 p-4 border rounded-lg hover:bg-gray-50 hover:border-gray-600 cursor-pointer"
                    onClick={() => seekToTimestamp(Math.floor(result.timestamp))}
                  >
                    <div className="w-fit text-sm text-white text-left p-1 px-2 rounded-md bg-blue-500">
                      {Math.floor(result.timestamp / 3600)}h{' '}
                      {Math.floor((result.timestamp % 3600) / 60)}m{' '}
                      {Math.floor(result.timestamp % 60)}s
                    </div>
                    <div className="mt-1">{result.chunk}</div>
                    <div
                      className={`w-fit text-sm text-white mt-1 p-1 px-2 rounded-md ${
                        result.similarity > 0.7
                          ? 'bg-green-500'
                          : result.similarity > 0.5
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                    >
                      Similarity Score: {result.similarity.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
