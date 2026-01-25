#!/bin/bash
# Finance RAG - Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Examples:
#   ./deploy.sh dev up          # Start development environment
#   ./deploy.sh prod build      # Build production images
#   ./deploy.sh k8s deploy      # Deploy to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="finance-rag"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
K8S_NAMESPACE="finance-rag"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed."; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log_error "Docker Compose is required but not installed."; exit 1; }

    if [ "$ENV" == "k8s" ]; then
        command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required for Kubernetes deployment."; exit 1; }
    fi

    log_info "All requirements met."
}

build_images() {
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"

    # Build API image
    docker build -t ${PROJECT_NAME}-api:latest --target production .
    log_info "Built ${PROJECT_NAME}-api:latest"

    # Build Streamlit image
    docker build -t ${PROJECT_NAME}-streamlit:latest --target streamlit .
    log_info "Built ${PROJECT_NAME}-streamlit:latest"

    # Build Worker image
    docker build -t ${PROJECT_NAME}-worker:latest --target worker .
    log_info "Built ${PROJECT_NAME}-worker:latest"

    if [ -n "$DOCKER_REGISTRY" ]; then
        log_info "Tagging images for registry: $DOCKER_REGISTRY"
        docker tag ${PROJECT_NAME}-api:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:latest
        docker tag ${PROJECT_NAME}-streamlit:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}-streamlit:latest
        docker tag ${PROJECT_NAME}-worker:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}-worker:latest
    fi
}

push_images() {
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_error "DOCKER_REGISTRY environment variable is not set."
        exit 1
    fi

    log_info "Pushing images to registry: $DOCKER_REGISTRY"
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:latest
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-streamlit:latest
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-worker:latest
    log_info "Images pushed successfully."
}

deploy_docker() {
    local action=$1
    cd "$PROJECT_ROOT"

    case $action in
        up)
            log_info "Starting services with Docker Compose..."
            docker-compose up -d
            log_info "Services started. Checking health..."
            sleep 10
            docker-compose ps
            ;;
        down)
            log_info "Stopping services..."
            docker-compose down
            ;;
        logs)
            docker-compose logs -f
            ;;
        restart)
            log_info "Restarting services..."
            docker-compose restart
            ;;
        *)
            log_error "Unknown action: $action"
            exit 1
            ;;
    esac
}

deploy_k8s() {
    local action=$1
    cd "$PROJECT_ROOT/deploy/k8s"

    case $action in
        deploy)
            log_info "Deploying to Kubernetes..."
            kubectl apply -k .
            log_info "Waiting for rollout..."
            kubectl -n $K8S_NAMESPACE rollout status deployment/finance-rag-api
            kubectl -n $K8S_NAMESPACE rollout status deployment/finance-rag-streamlit
            log_info "Deployment complete."
            ;;
        delete)
            log_warn "Deleting Kubernetes resources..."
            kubectl delete -k .
            ;;
        status)
            kubectl -n $K8S_NAMESPACE get all
            ;;
        logs)
            local pod=$2
            kubectl -n $K8S_NAMESPACE logs -f deployment/$pod
            ;;
        *)
            log_error "Unknown action: $action"
            exit 1
            ;;
    esac
}

show_help() {
    echo "Finance RAG Deployment Script"
    echo ""
    echo "Usage: $0 [environment] [action]"
    echo ""
    echo "Environments:"
    echo "  dev       Local development with Docker Compose"
    echo "  prod      Production Docker deployment"
    echo "  k8s       Kubernetes deployment"
    echo ""
    echo "Actions:"
    echo "  build     Build Docker images"
    echo "  push      Push images to registry"
    echo "  up        Start services"
    echo "  down      Stop services"
    echo "  restart   Restart services"
    echo "  logs      View logs"
    echo "  deploy    Deploy to K8s"
    echo "  delete    Delete K8s resources"
    echo "  status    Show K8s status"
    echo ""
    echo "Examples:"
    echo "  $0 dev up          Start development environment"
    echo "  $0 prod build      Build production images"
    echo "  $0 k8s deploy      Deploy to Kubernetes"
}

# Main
ENV=${1:-dev}
ACTION=${2:-help}

if [ "$ACTION" == "help" ] || [ "$ACTION" == "-h" ] || [ "$ACTION" == "--help" ]; then
    show_help
    exit 0
fi

check_requirements

case $ENV in
    dev)
        export DOCKER_BUILDKIT=1
        case $ACTION in
            build)
                build_images
                ;;
            *)
                deploy_docker $ACTION
                ;;
        esac
        ;;
    prod)
        export DOCKER_BUILDKIT=1
        case $ACTION in
            build)
                build_images
                ;;
            push)
                push_images
                ;;
            *)
                deploy_docker $ACTION
                ;;
        esac
        ;;
    k8s)
        case $ACTION in
            build)
                build_images
                ;;
            push)
                push_images
                ;;
            *)
                deploy_k8s $ACTION
                ;;
        esac
        ;;
    *)
        log_error "Unknown environment: $ENV"
        show_help
        exit 1
        ;;
esac

log_info "Done!"
