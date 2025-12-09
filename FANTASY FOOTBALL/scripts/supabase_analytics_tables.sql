-- ============================================================================
-- CREATOR ANALYTICS TABLES
-- ============================================================================
-- Run this script in the Supabase SQL Editor to create analytics tracking tables.

-- ============================================================================
-- 1. ANALYTICS EVENTS TABLE - Raw event log
-- ============================================================================

CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    event_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_analytics_events_user ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_events_created ON analytics_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_user_type ON analytics_events(user_id, event_type);

-- Enable RLS
ALTER TABLE analytics_events ENABLE ROW LEVEL SECURITY;

-- Users can only see their own events
CREATE POLICY "Users can view own analytics events"
ON analytics_events FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Users can insert their own events
CREATE POLICY "Users can insert own analytics events"
ON analytics_events FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

-- Users can delete their own events (for privacy)
CREATE POLICY "Users can delete own analytics events"
ON analytics_events FOR DELETE
TO authenticated
USING (auth.uid() = user_id);

COMMENT ON TABLE analytics_events IS 'Raw event log for creator analytics tracking';
COMMENT ON COLUMN analytics_events.event_type IS 'Event types: debate_started, debate_completed, segment_regenerated, audio_exported, transcript_exported, panel_preset_saved, panel_preset_loaded, style_capture_completed';


-- ============================================================================
-- 2. CREATOR STATS TABLE - Aggregated metrics per user
-- ============================================================================

CREATE TABLE IF NOT EXISTS creator_stats (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    total_debates INTEGER DEFAULT 0,
    total_exports INTEGER DEFAULT 0,
    total_regenerations INTEGER DEFAULT 0,
    favorite_panel JSONB,  -- { panel_ids, panel_names, use_count }
    favorite_topic_words TEXT[],  -- Most common topic keywords
    avg_debate_duration_ms INTEGER,
    avg_segments_per_debate FLOAT,
    completion_rate FLOAT,  -- debates completed / debates started
    last_active_at TIMESTAMPTZ,
    stats_updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE creator_stats ENABLE ROW LEVEL SECURITY;

-- Users can only see their own stats
CREATE POLICY "Users can view own creator stats"
ON creator_stats FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Users can update their own stats
CREATE POLICY "Users can update own creator stats"
ON creator_stats FOR UPDATE
TO authenticated
USING (auth.uid() = user_id);

-- Allow insert for new users
CREATE POLICY "Users can insert own creator stats"
ON creator_stats FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

COMMENT ON TABLE creator_stats IS 'Aggregated analytics metrics per creator, updated periodically';


-- ============================================================================
-- 3. PANEL PERFORMANCE TABLE - Track panel combination performance
-- ============================================================================

CREATE TABLE IF NOT EXISTS panel_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    panel_hash TEXT NOT NULL,  -- Hash of sorted panel_ids for grouping
    panel_ids TEXT[] NOT NULL,
    panel_names TEXT[],
    use_count INTEGER DEFAULT 1,
    completion_count INTEGER DEFAULT 0,
    export_count INTEGER DEFAULT 0,
    avg_regenerations FLOAT DEFAULT 0,
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, panel_hash)
);

CREATE INDEX IF NOT EXISTS idx_panel_performance_user ON panel_performance(user_id);
CREATE INDEX IF NOT EXISTS idx_panel_performance_use ON panel_performance(user_id, use_count DESC);

-- Enable RLS
ALTER TABLE panel_performance ENABLE ROW LEVEL SECURITY;

-- Users can only see their own panel performance
CREATE POLICY "Users can view own panel performance"
ON panel_performance FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Users can insert their own panel performance
CREATE POLICY "Users can insert own panel performance"
ON panel_performance FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

-- Users can update their own panel performance
CREATE POLICY "Users can update own panel performance"
ON panel_performance FOR UPDATE
TO authenticated
USING (auth.uid() = user_id);

COMMENT ON TABLE panel_performance IS 'Tracks which panel combinations perform best for each creator';


-- ============================================================================
-- 4. HELPER FUNCTION: Generate panel hash
-- ============================================================================

CREATE OR REPLACE FUNCTION generate_panel_hash(panel_ids TEXT[])
RETURNS TEXT AS $$
BEGIN
    -- Sort the array and concatenate with delimiter
    RETURN array_to_string(ARRAY(SELECT unnest(panel_ids) ORDER BY 1), '|');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION generate_panel_hash IS 'Generates a consistent hash for panel combinations regardless of order';


-- ============================================================================
-- 5. TRIGGER: Auto-update last_active_at on new events
-- ============================================================================

CREATE OR REPLACE FUNCTION update_last_active()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO creator_stats (user_id, last_active_at, stats_updated_at)
    VALUES (NEW.user_id, NOW(), NOW())
    ON CONFLICT (user_id) 
    DO UPDATE SET last_active_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_last_active ON analytics_events;
CREATE TRIGGER trg_update_last_active
AFTER INSERT ON analytics_events
FOR EACH ROW
EXECUTE FUNCTION update_last_active();


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Run these after creating the tables to verify setup:

-- Check tables exist:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%analytics%' OR table_name LIKE '%creator%' OR table_name LIKE '%panel_performance%';

-- Check RLS policies:
-- SELECT tablename, policyname FROM pg_policies WHERE tablename IN ('analytics_events', 'creator_stats', 'panel_performance');
