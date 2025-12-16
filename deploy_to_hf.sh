#!/bin/bash
# 🚀 Hugging Face Spaces Quick Deploy Script
# This script automates the deployment process
# Usage: bash deploy_to_hf.sh YOUR_HF_USERNAME

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if username provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: bash deploy_to_hf.sh YOUR_HF_USERNAME${NC}"
    echo ""
    echo "Steps before running this script:"
    echo "1. Create HF account: https://huggingface.co/join"
    echo "2. Create HF Space:"
    echo "   - Go to https://huggingface.co/spaces"
    echo "   - Click 'Create new Space'"
    echo "   - Name: chat-with-your-data"
    echo "   - SDK: Docker"
    echo "   - License: gpl-3.0"
    echo "   - Visibility: Public"
    echo "3. Install HF CLI: pip install huggingface-hub"
    echo "4. Authenticate: huggingface-cli login"
    echo ""
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME="chat-with-your-data"
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
TEMP_DIR="/tmp/rag-hf-deploy-$$"

# Get the source directory BEFORE changing directories
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🚀 Deploying Chat-with-your-datato Hugging Face Spaces${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git first."
    exit 1
fi

if ! python3 -m pip show huggingface-hub &> /dev/null; then
    echo "⚠️  HF CLI not installed. Installing..."
    pip install huggingface-hub
fi

if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not authenticated with Hugging Face. Run: huggingface-cli login"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites met${NC}"
echo ""

# Step 2: Get HF token and configure git
echo -e "${YELLOW}[2/6] Configuring git authentication...${NC}"

# Get HF token from huggingface_hub config
HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())" 2>/dev/null)

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Could not retrieve HF token. Make sure you're authenticated with 'huggingface-cli login'"
    exit 1
fi

# Configure git to use token-based auth
git config --global credential.helper store
git config --global credential."https://huggingface.co".username "$HF_USERNAME"

echo -e "${GREEN}✓ Git configured${NC}"
echo ""

# Step 3: Create temporary directory and clone
echo -e "${YELLOW}[3/6] Cloning HF Space repository...${NC}"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Use token-based HTTPS URL for clone
git clone "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}" . 2>/dev/null || {
    echo "❌ Failed to clone space. Verify space was created at:"
    echo "   https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
    exit 1
}

# Configure git to use token for future pushes
git config user.email "neetikasaxena@huggingface.co"
git config user.name "$HF_USERNAME"
git remote set-url origin "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo -e "${GREEN}✓ Repository cloned${NC}"
echo ""

# Step 4: Copy project files
echo -e "${YELLOW}[4/6] Copying project files...${NC}"

if [ ! -f "${SOURCE_DIR}/Dockerfile" ]; then
    echo "❌ Source directory not found. Expected Dockerfile at: ${SOURCE_DIR}"
    exit 1
fi

# Copy all files except virtual environments and cache
# Note: Need to copy both regular and hidden files (like .streamlit/)
cp -r "${SOURCE_DIR}"/* . 2>/dev/null || true
cp -r "${SOURCE_DIR}"/.streamlit . 2>/dev/null || true

# Clean up local-only files (but keep .git which is needed for HF)
echo "  Cleaning up local files..."
rm -rf venv venv\ 2 __pycache__ .env dump.rdb .github
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Ensure README.md is from HF_README.md
if [ -f "HF_README.md" ]; then
    rm -f README.md
    mv HF_README.md README.md
fi

echo -e "${GREEN}✓ Files copied and cleaned${NC}"
echo ""

# Step 5: Verify essential files
echo -e "${YELLOW}[5/6] Verifying essential files...${NC}"
REQUIRED_FILES=("Dockerfile" "docker-compose.yml" "requirements.txt" "app.py" "README.md" "LICENSE")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo -e "${GREEN}✓ All essential files present${NC}"
echo ""

# Step 5.5: Configure Git LFS for large files
echo -e "${YELLOW}[5.5/6] Configuring Git LFS for binary files...${NC}"

# Install Git LFS if not already installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y git-lfs
    fi
fi

# Initialize Git LFS and track binary files
git lfs install --skip-repo
git lfs track "*.png" "*.jpg" "*.jpeg" "*.gif" "*.pdf" "*.db"

echo -e "${GREEN}✓ Git LFS configured${NC}"
echo ""

# Step 6: Git commit and push
echo -e "${YELLOW}[6/6] Committing and pushing to Hugging Face...${NC}"

git add -A

# Check if there are changes
if ! git diff --cached --quiet; then
    git commit -m "Initial deployment of RAG with Gemma-3 to Hugging Face Spaces

- Added production-grade RAG system with Gemma-3 LLM
- Intelligent query caching (700x performance improvement)
- Multi-document support with semantic search
- User authentication and session management
- FastAPI backend with async streaming responses
- Streamlit web interface
- Qdrant vector database for embeddings
- PostgreSQL for user data
- Redis for cache and history
- Comprehensive documentation and troubleshooting guides"

    # Push to HF Spaces
    if git push; then
        echo -e "${GREEN}✓ Changes pushed to Hugging Face${NC}"
    else
        echo -e "${YELLOW}⚠️  First push with Git LFS may take a moment...${NC}"
        git push
        echo -e "${GREEN}✓ Changes pushed to Hugging Face${NC}"
    fi
else
    echo "ℹ️  No changes to push (up to date)"
fi

echo ""

# Step 6: Summary and next steps
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Deployment initiated successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "📍 Your Space URL:"
echo -e "   ${GREEN}${SPACE_URL}${NC}"
echo ""
echo "⏱️  Build Status:"
echo "   Docker build in progress (10-15 minutes typical)"
echo ""
echo "📋 Next Steps:"
echo "   1. Go to: ${SPACE_URL}"
echo "   2. Click 'Build Logs' to monitor Docker build"
echo "   3. Wait for 'Successfully built' message"
echo "   4. Click 'App' tab when ready"
echo "   5. Test the application:"
echo "      - Register a test account"
echo "      - Upload a sample document"
echo "      - Ask questions about it"
echo "      - Try same question twice to see cache speedup"
echo ""
echo "💡 Tips:"
echo "   - Free tier uses CPU (slow, but works)"
echo "   - Upgrade to HF Pro for GPU (2-3x faster)"
echo "   - First response takes 60-90 seconds (LLM startup)"
echo "   - Repeated queries <100ms (cache hit)"
echo ""
echo "📚 Documentation:"
echo "   - Full guide: HUGGING_FACE_DEPLOYMENT.md"
echo "   - Quick checklist: HF_DEPLOYMENT_CHECKLIST.md"
echo "   - Project info: README.md"
echo ""
echo "🔗 Share your project:"
echo "   Add to GitHub README:"
echo "   [![HF Space](badge.svg)](${SPACE_URL})"
echo ""
echo -e "${GREEN}Deployment directory: ${TEMP_DIR}${NC}"
echo ""
