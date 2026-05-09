CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE transcript_category AS ENUM (
  'frontend',
  'databases',
  'ai-ml',
  'architecture',
  'devops',
  'general'
);

CREATE TABLE transcripts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename      TEXT NOT NULL UNIQUE,
  title         TEXT NOT NULL,
  category      transcript_category NOT NULL,
  speakers      TEXT[] NOT NULL,
  recorded_date DATE NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE transcript_chunks (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  transcript_id     UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  chunk_index       INTEGER NOT NULL,
  chunk_text        TEXT NOT NULL,
  speakers_in_chunk TEXT[] NOT NULL,
  embed_model       TEXT NOT NULL,
  embedding         vector(384),
  embedded_at       TIMESTAMPTZ,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (transcript_id, chunk_index)
);

CREATE INDEX idx_chunks_transcript    ON transcript_chunks(transcript_id);
CREATE INDEX idx_chunks_embedding     ON transcript_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_transcripts_category ON transcripts(category);
CREATE INDEX idx_transcripts_recorded ON transcripts(recorded_date);
