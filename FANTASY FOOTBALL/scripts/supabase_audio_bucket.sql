-- ============================================================================
-- SUPABASE AUDIO SAMPLES STORAGE BUCKET
-- ============================================================================
-- Run this script in the Supabase SQL Editor to create the audio storage bucket
-- for the Style Capture audio transcription feature.

-- Create audio samples bucket (public read for transcription service access)
INSERT INTO storage.buckets (id, name, public)
VALUES ('audio-samples', 'audio-samples', true)
ON CONFLICT (id) DO NOTHING;


-- ============================================================================
-- STORAGE POLICIES
-- ============================================================================

-- Allow authenticated users to upload audio samples
CREATE POLICY "Users can upload audio samples"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'audio-samples');

-- Allow public read for transcription service
-- (Modal needs to download the audio file for Whisper transcription)
CREATE POLICY "Public read for audio samples"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'audio-samples');

-- Allow users to delete their own audio samples
CREATE POLICY "Users can delete own audio samples"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'audio-samples');


-- ============================================================================
-- OPTIONAL: AUTO-DELETE OLD SAMPLES (7 DAYS)
-- ============================================================================
-- Uncomment and run these to automatically clean up old audio files
-- This helps control storage costs for temporary transcription files.

/*
-- Function to delete old audio samples
CREATE OR REPLACE FUNCTION delete_old_audio_samples()
RETURNS void AS $$
BEGIN
    DELETE FROM storage.objects
    WHERE bucket_id = 'audio-samples'
    AND created_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Schedule the cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-audio-samples', '0 3 * * *', 'SELECT delete_old_audio_samples()');
*/


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Run these after creating the bucket to verify setup:

-- Check bucket exists:
-- SELECT * FROM storage.buckets WHERE id = 'audio-samples';

-- Check policies:
-- SELECT * FROM pg_policies WHERE tablename = 'objects' AND policyname LIKE '%audio%';
