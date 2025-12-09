#!/bin/bash
# Fantasy Football Neuron - API Endpoint Tests
# Run: bash scripts/test_api_endpoints.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_BASE="${API_BASE:-http://localhost:3000}"

echo "============================================================"
echo "🧪 API Endpoint Tests"
echo "============================================================"
echo "Base URL: $API_BASE"
echo ""

# Track results
PASSED=0
FAILED=0

test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    local expected=$5
    
    echo -n "Testing: $name... "
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_BASE$endpoint" 2>/dev/null || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null || echo "000")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" == "$expected" ]; then
        echo -e "${GREEN}✓ (HTTP $http_code)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ (Expected $expected, got $http_code)${NC}"
        ((FAILED++))
    fi
}

# Health check
echo ""
echo "--- Health Checks ---"
test_endpoint "API Health" "GET" "/api" "" "200"

# Debate endpoints
echo ""
echo "--- Debate Endpoints ---"
test_endpoint "Create Debate" "POST" "/api/debate" '{"players":["Mahomes","Allen"]}' "200"
test_endpoint "Invalid Debate" "POST" "/api/debate" '{}' "400"

# TTS endpoints
echo ""
echo "--- TTS Endpoints ---"
test_endpoint "Generate TTS" "POST" "/api/tts" '{"text":"Hello world","voice":"dallas_homer"}' "200"

# Content extraction (may fail if external services not available)
echo ""
echo "--- Content Extraction (Optional) ---"
test_endpoint "YouTube Extract" "POST" "/api/content/extract-youtube" '{"url":"https://youtube.com/watch?v=test"}' "200"
test_endpoint "Article Extract" "POST" "/api/content/extract-article" '{"url":"https://example.com/article"}' "200"

# Summary
echo ""
echo "============================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All API tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed (may need local server running)${NC}"
    exit 1
fi
