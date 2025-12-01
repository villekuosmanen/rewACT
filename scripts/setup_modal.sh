#!/bin/bash
# Setup script for Modal training pipeline

set -e

echo "=========================================="
echo "RewACT Modal Pipeline Setup"
echo "=========================================="
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal is not installed."
    echo "   Install with: pip install modal"
    exit 1
fi

echo "✅ Modal is installed"

# Check if modal is authenticated
if ! modal profile list &> /dev/null; then
    echo "❌ Modal is not authenticated."
    echo "   Run: modal setup"
    exit 1
fi

echo "✅ Modal is authenticated"

# Check for secrets
echo ""
echo "Checking Modal secrets..."

HF_SECRET=$(modal secret list 2>/dev/null | grep "huggingface-secret" || echo "")
WANDB_SECRET=$(modal secret list 2>/dev/null | grep "wandb-secret" || echo "")
DISCORD_SECRET=$(modal secret list 2>/dev/null | grep "discord-webhook" || echo "")

if [ -z "$HF_SECRET" ]; then
    echo "❌ HuggingFace secret not found"
    echo ""
    read -p "   Enter your HuggingFace token (hf_...): " HF_TOKEN
    modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
    echo "   ✅ Created huggingface-secret"
else
    echo "✅ HuggingFace secret configured"
fi

if [ -z "$WANDB_SECRET" ]; then
    echo "❌ WandB secret not found"
    echo ""
    read -p "   Enter your WandB API key: " WANDB_KEY
    modal secret create wandb-secret WANDB_API_KEY="$WANDB_KEY"
    echo "   ✅ Created wandb-secret"
else
    echo "✅ WandB secret configured"
fi

if [ -z "$DISCORD_SECRET" ]; then
    echo "⚠️  Discord webhook not found (optional)"
    echo ""
    read -p "   Enter Discord webhook URL (or press Enter to skip): " DISCORD_URL
    if [ -n "$DISCORD_URL" ]; then
        modal secret create discord-webhook DISCORD_WEBHOOK_URL="$DISCORD_URL"
        echo "   ✅ Created discord-webhook"
    else
        echo "   ⏭️  Skipped (you can add it later with: modal secret create discord-webhook DISCORD_WEBHOOK_URL=...)"
    fi
else
    echo "✅ Discord webhook configured"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review MODAL_PIPELINE.md for usage instructions"
echo "2. Run the pipeline with:"
echo "   modal run scripts/modal_pipeline.py"
echo ""
echo "To test your Discord webhook (if configured):"
echo "   python scripts/test_discord.py"
echo ""





