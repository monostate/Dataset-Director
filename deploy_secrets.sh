#!/bin/bash

# Deploy secrets to Fly.io from .env file
# This script reads the .env file and sets secrets in Fly
#
# Usage:
#   FLY_APP_NAME=your-app-name ./deploy_secrets.sh
#   # or set FLY_APP_NAME in your .env file

echo "Setting Fly secrets from .env file..."

# Read .env file and set secrets
if [ -f .env ]; then
    # Export the .env variables
    export $(cat .env | grep -v '^#' | xargs)
    
    # Set secrets in Fly (with production CORS)
    fly secrets set \
        API_KEY="$API_KEY" \
        KUMO_API_KEY="$KUMO_API_KEY" \
        KUMO_URL="$KUMO_URL" \
        HF_TOKEN="$HF_TOKEN" \
        CORS_ORIGINS="${CORS_ORIGINS:-https://yourdomain.com}" \
        REDIS_URL="$REDIS_URL" \
        SESSION_TTL="$SESSION_TTL" \
        ENCRYPTION_KEY="$ENCRYPTION_KEY" \
        LOG_LEVEL="$LOG_LEVEL" \
        --app ${FLY_APP_NAME:-your-app-name}
    
    echo "Secrets set successfully!"
    echo "Note: Remember to update CORS_ORIGINS for your domain"
else
    echo "Error: .env file not found"
    exit 1
fi
