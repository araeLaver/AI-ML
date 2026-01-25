#!/bin/bash
# Finance RAG - Health Check Script
# Checks the health of all services

set -e

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
STREAMLIT_URL="${STREAMLIT_URL:-http://localhost:8501}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check_api() {
    echo -n "Checking API... "
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

check_streamlit() {
    echo -n "Checking Streamlit... "
    if curl -sf "${STREAMLIT_URL}/_stcore/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

check_redis() {
    echo -n "Checking Redis... "
    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        fi
    fi
    echo -e "${RED}FAILED${NC}"
    return 1
}

echo "=========================================="
echo "Finance RAG Health Check"
echo "=========================================="
echo ""

FAILED=0

check_api || FAILED=1
check_streamlit || FAILED=1
check_redis || FAILED=1

echo ""
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All services healthy!${NC}"
    exit 0
else
    echo -e "${RED}Some services are unhealthy!${NC}"
    exit 1
fi
