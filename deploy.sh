#!/bin/bash

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

#-----------------------------------------------------------------------------
# Global config.

# Build mode: up, image or build (default: up)
BUILD_MODE="up"

# Theta registry base URL
THETA_REGISTRY_BASE="theta-public-registry.cn-hangzhou.cr.aliyuncs.com"

# Cloud images (only for backend)
CLOUD_BACKEND_IMAGE="${THETA_REGISTRY_BASE}/theta/mirobody-backend"

# Local images
BACKEND_IMAGE="mirobody-backend"
DOCKER_COMPOSE_FILE="docker/docker-compose.yaml"

#-----------------------------------------------------------------------------
# Parse command line arguments.

for arg in "$@"; do
    case $arg in
        --mode=*)
            BUILD_MODE="${arg#*=}"
            shift
            ;;
        --help)
            echo "Usage: ./deploy.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode=image    Pull pre-built backend image from theta registry"
            echo "  --mode=build    Build backend from scratch using official base images"
            echo "  --mode=up       Skip build, directly compose up (default)"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./deploy.sh                  # Skip build and start (default)"
            echo "  ./deploy.sh --mode=image     # Use theta registry images"
            echo "  ./deploy.sh --mode=build     # Build from scratch using official Docker Hub images"
            echo "  ./deploy.sh --mode=up        # Skip build and start"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate build mode
if [[ "$BUILD_MODE" != "up" && "$BUILD_MODE" != "image" && "$BUILD_MODE" != "build" ]]; then
    echo "Error: BUILD_MODE must be 'up', 'image' or 'build', got: $BUILD_MODE"
    exit 1
fi

echo "=========================================="
echo "Build Mode: $BUILD_MODE"
echo "=========================================="

#-----------------------------------------------------------------------------
# Functions.

# Build backend image based on mode
build_backend() {
    local mode=$1
    if [[ "$mode" == "image" ]]; then
        echo "Pulling backend base image from cloud..."
        docker pull $CLOUD_BACKEND_IMAGE:latest
        echo "Building backend image with local requirements..."
        docker build -f docker/Dockerfile.backend.cloud -t $BACKEND_IMAGE:latest --build-arg THETA_REGISTRY="${THETA_REGISTRY}" .
    else
        echo "Building backend image from Ubuntu base..."
        docker build -f docker/Dockerfile.backend -t $BACKEND_IMAGE:latest --build-arg BASE_REGISTRY="${BASE_REGISTRY}" .
    fi
}

#-----------------------------------------------------------------------------
# Build or prepare images based on mode.
 
# Set images and registry prefixes based on mode
if [[ "$BUILD_MODE" == "image" ]]; then
    export REDIS_IMAGE="${THETA_REGISTRY_BASE}/docker.io/redis:7.0"
    export POSTGRES_IMAGE="${THETA_REGISTRY_BASE}/docker.io/pgvector:pg17"
    export BASE_REGISTRY="${THETA_REGISTRY_BASE}/"
    export THETA_REGISTRY="${THETA_REGISTRY_BASE}/"
else
    # Build and up mode use official images
    export REDIS_IMAGE="redis:7.0-alpine"
    export POSTGRES_IMAGE="pgvector/pgvector:pg17"
    export BASE_REGISTRY=""
    export THETA_REGISTRY=""
fi

if [[ "$BUILD_MODE" == "up" ]]; then
    echo "=========================================="
    echo "Up Mode: Skip build, use existing images"
    echo "=========================================="
    
    echo "Using existing images:"
    echo "  - $BACKEND_IMAGE:latest"
    
    # Check if backend image exists
    if ! docker image inspect $BACKEND_IMAGE:latest > /dev/null 2>&1; then
        echo "Error: Backend image $BACKEND_IMAGE:latest not found."
        echo "Please run with --mode=image or --mode=local first."
        exit 1
    fi

elif [[ "$BUILD_MODE" == "image" ]]; then
    echo "=========================================="
    echo "Image Mode: Pulling pre-built backend"
    echo "=========================================="
    
    build_backend "image"

else
    echo "=========================================="
    echo "Build Mode: Building images from scratch"
    echo "=========================================="
    
    build_backend "build"
fi

#-----------------------------------------------------------------------------
# Docker compose.

docker compose -f $DOCKER_COMPOSE_FILE down
docker compose -f $DOCKER_COMPOSE_FILE up -d --remove-orphans
docker compose -f $DOCKER_COMPOSE_FILE logs -f


