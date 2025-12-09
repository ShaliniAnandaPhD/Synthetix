-- Migration for shared_debates table
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS shared_debates (
  share_code TEXT PRIMARY KEY,
  debate_data JSONB NOT NULL,
  topic TEXT NOT NULL,
  speakers TEXT[] NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  view_count INTEGER DEFAULT 0,
  expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '30 days'
);

-- Add RLS policies for public read access
ALTER TABLE shared_debates ENABLE ROW LEVEL SECURITY;

-- Anyone can read shared debates (public links)
CREATE POLICY "Public can read shared debates"
  ON shared_debates FOR SELECT
  USING (true);

-- Only authenticated users can create shares
CREATE POLICY "Authenticated users can create shares"
  ON shared_debates FOR INSERT
  WITH CHECK (auth.role() = 'authenticated');

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_shared_debates_share_code ON shared_debates(share_code);
CREATE INDEX IF NOT EXISTS idx_shared_debates_expires_at ON shared_debates(expires_at);

-- Optional: Function to increment view count
CREATE OR REPLACE FUNCTION increment_debate_views(code TEXT)
RETURNS void AS $$
BEGIN
  UPDATE shared_debates 
  SET view_count = view_count + 1 
  WHERE share_code = code;
END;
$$ LANGUAGE plpgsql;
