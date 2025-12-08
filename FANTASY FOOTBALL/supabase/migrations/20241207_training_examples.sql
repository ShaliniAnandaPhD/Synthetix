-- Training Examples Table for AI Fine-Tuning
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS training_examples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  personality_id UUID REFERENCES personalities(id) ON DELETE CASCADE,
  city VARCHAR(100) NOT NULL,
  input_prompt TEXT NOT NULL,
  context JSONB DEFAULT '{}',
  original_output TEXT NOT NULL,
  edited_output TEXT,
  rating VARCHAR(20) NOT NULL CHECK (rating IN ('positive', 'negative', 'edited', 'saved')),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL
);

-- Index for fast queries by personality
CREATE INDEX IF NOT EXISTS idx_training_examples_personality 
ON training_examples(personality_id);

-- Index for filtering by rating type
CREATE INDEX IF NOT EXISTS idx_training_examples_rating 
ON training_examples(rating);

-- Index for export queries (positive/edited/saved only)
CREATE INDEX IF NOT EXISTS idx_training_examples_export 
ON training_examples(personality_id, rating) 
WHERE rating IN ('positive', 'edited', 'saved');

-- RLS Policies
ALTER TABLE training_examples ENABLE ROW LEVEL SECURITY;

-- Allow users to insert their own training examples
CREATE POLICY "Users can insert training examples"
ON training_examples FOR INSERT
TO authenticated
WITH CHECK (true);

-- Allow users to read all training examples (for stats)
CREATE POLICY "Users can read training examples"
ON training_examples FOR SELECT
TO authenticated
USING (true);

-- Comments for documentation
COMMENT ON TABLE training_examples IS 'Stores user feedback on AI responses for fine-tuning';
COMMENT ON COLUMN training_examples.rating IS 'positive=good, negative=bad, edited=user corrected, saved=exceptional example';
COMMENT ON COLUMN training_examples.edited_output IS 'User-corrected version of AI output (only when rating=edited)';
COMMENT ON COLUMN training_examples.metadata IS 'Additional context: turn_number, event_type, latency_ms, etc.';
