#!/bin/bash
# Quick deployment verification and setup script

echo "=================================================="
echo "P0 Cultural Cognition - Deployment Verification"
echo "=================================================="
echo ""

# Check Modal authentication
echo "1. Checking Modal authentication..."
if modal token show > /dev/null 2>&1; then
    echo "   ✅ Modal authenticated"
else
    echo "   ❌ Not authenticated - run: modal token set"
    exit 1
fi

echo ""
echo "2. Current Modal apps:"
modal app list
echo ""

echo "3. Next steps:"
echo "   a) If 'neuron-orchestrator' exists:"
echo "      - Check logs: modal app logs neuron-orchestrator"
echo "      - Get URL: modal app show neuron-orchestrator"
echo ""
echo "   b) If app DOES NOT exist:"
echo "      - Deploy: modal deploy infra/modal_orchestrator.py"
echo ""

read -p "Do you want to deploy/redeploy Modal app now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deploying Modal app with P0 enhancements..."
    modal deploy infra/modal_orchestrator.py
    
    echo ""
    echo "=================================================="
    echo "Deployment complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Copy the endpoint URLs shown above"
    echo "2. Set Vercel environment variables:"
    echo "   vercel env add MODAL_CULTURAL_URL production"
    echo "3. Deploy Vercel:"
    echo "   vercel deploy --prod"
    echo "4. Test:"
    echo "   curl -X POST https://your-app.vercel.app/api/cultural \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"city\":\"Philadelphia\",\"user_input\":\"test\"}'"
    echo ""
else
    echo "Skipping deployment."
fi
