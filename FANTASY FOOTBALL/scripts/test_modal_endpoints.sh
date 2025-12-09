#!/bin/bash
# Fantasy Football Neuron - Modal Endpoint Tests
# Run: bash scripts/test_modal_endpoints.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get Modal base URL from env or use default
MODAL_BASE="${MODAL_BASE:-https://shalini--neuron-orchestrator}"

echo "============================================================"
echo "🧪 Modal Endpoint Tests"
echo "============================================================"
echo "Base URL: $MODAL_BASE"
echo ""

PASSED=0
FAILED=0

test_modal_endpoint() {
    local name=$1
    local endpoint=$2
    local data=$3
    
    echo -n "Testing: $name... "
    
    url="${MODAL_BASE}${endpoint}.modal.run"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data" \
        --max-time 30 2>/dev/null || echo "000")
    
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}✓ (HTTP 200)${NC}"
        ((PASSED++))
    elif [ "$http_code" == "000" ]; then
        echo -e "${YELLOW}⚠ Timeout/Unreachable${NC}"
        ((FAILED++))
    else
        echo -e "${RED}✗ (HTTP $http_code)${NC}"
        ((FAILED++))
    fi
}

# Test endpoints
echo ""
echo "--- Core Debate Endpoints ---"
test_modal_endpoint "Run Debate" "-run-debate" '{"players":["Mahomes","Allen"],"format":"quick"}'
test_modal_endpoint "Generate TTS" "-generate-tts" '{"text":"Touchdown Kansas City!","voice":"kc_homer"}'
test_modal_endpoint "Regenerate Segment" "-regenerate-segment" '{"segment_id":"test","text":"New text"}'

echo ""
echo "--- Content Extraction ---"
test_modal_endpoint "Content Extract" "-content-extract" '{"action":"youtube","url":"https://youtube.com/watch?v=test"}'

echo ""
echo "--- Multi-City Commentary ---"
test_modal_endpoint "Multi-City Commentary" "-generate-multi-city-commentary" '{"event":"touchdown","regions":["dallas","kansas_city"]}'

# Summary
echo ""
echo "============================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All Modal tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed (check Modal deployment status)${NC}"
    exit 1
fi
