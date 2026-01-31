#!/usr/bin/env bash
#
# Release script for local-transcribe
#
# This script creates a new release by:
# 1. Bumping the version (patch by default)
# 2. Creating a git tag
# 3. Updating web_ui package.json
# 4. Building the package
#
# Usage:
#   ./release.sh                    # Patch version bump (0.1.0 -> 0.1.1)
#   ./release.sh minor              # Minor version bump (0.1.0 -> 0.2.0)
#   ./release.sh major              # Major version bump (0.1.0 -> 1.0.0)
#   ./release.sh 1.2.3              # Specific version
#

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_msg() {
    echo -e "${BLUE}[RELEASE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    print_error "Working directory is not clean. Please commit or stash changes first."
    git status --short
    exit 1
fi

# Check if we're on a branch
if ! git rev-parse --abbrev-ref HEAD &>/dev/null; then
    print_error "Not on a git branch. Cannot create release."
    exit 1
fi

# Get current version
CURRENT_VERSION=$(uv run python -c "from local_transcribe import __version__; print(__version__)" | sed 's/\.dev.*//')
print_msg "Current version: $CURRENT_VERSION"

# Validate current version format
if ! [[ $CURRENT_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && ! [[ $CURRENT_VERSION =~ ^[0-9]+\.[0-9]+$ ]]; then
    print_warning "Current version '$CURRENT_VERSION' may not be a clean semantic version."
    # Try to extract base version
    CURRENT_VERSION=$(echo "$CURRENT_VERSION" | sed -E 's/^([0-9]+\.[0-9]+)(\.[0-9]+)?.*$/\1\2/' | sed 's/\.$//')
    if [[ -z $CURRENT_VERSION ]]; then
        CURRENT_VERSION="0.1.0"
        print_warning "Using default version: $CURRENT_VERSION"
    fi
fi

# Determine new version
if [[ $# -eq 0 ]]; then
    # Default: patch bump
    IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
    # Handle case where patch might be empty
    patch=${patch:-0}
    NEW_VERSION="$major.$minor.$((patch + 1))"
    BUMP_TYPE="patch"
elif [[ $# -eq 1 ]]; then
    case $1 in
        major)
            IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
            NEW_VERSION="$((major + 1)).0.0"
            BUMP_TYPE="major"
            ;;
        minor)
            IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
            NEW_VERSION="$major.$((minor + 1)).0"
            BUMP_TYPE="minor"
            ;;
        patch)
            IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
            NEW_VERSION="$major.$minor.$((patch + 1))"
            BUMP_TYPE="patch"
            ;;
        *)
            # Specific version - validate format
            if ! [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                print_error "Invalid version format: $1 (expected: X.Y.Z)"
                exit 1
            fi
            NEW_VERSION=$1
            BUMP_TYPE="specific ($1)"
            ;;
    esac
else
    print_error "Usage: $0 [major|minor|patch|<version>]"
    exit 1
fi

print_msg "Creating $BUMP_TYPE release: $CURRENT_VERSION -> $NEW_VERSION"

# Check if tag already exists
if git rev-parse "v$NEW_VERSION" &>/dev/null; then
    print_error "Tag v$NEW_VERSION already exists!"
    exit 1
fi

# Confirm
read -p "Continue with release $NEW_VERSION? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "Release cancelled."
    exit 0
fi

# Update web_ui package.json version
print_msg "Updating web_ui/package.json version..."
if [[ -f web_ui/package.json ]]; then
    # Use sed to update version in package.json
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" web_ui/package.json
    else
        # Linux
        sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" web_ui/package.json
    fi
    print_success "Updated web_ui version to $NEW_VERSION"
    
    # Stage the change
    git add web_ui/package.json
    git commit -m "chore: bump web_ui version to $NEW_VERSION"
else
    print_warning "web_ui/package.json not found, skipping web_ui version update"
fi

# Create annotated tag
print_msg "Creating git tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Build package to verify everything works
print_msg "Building package..."
if ! uv build; then
    print_error "Build failed! Rolling back tag..."
    git tag -d "v$NEW_VERSION"
    # Also rollback the web_ui commit if it was made
    if git log -1 --pretty=%B | grep -q "chore: bump web_ui version to $NEW_VERSION"; then
        git reset --hard HEAD~1
    fi
    exit 1
fi

# Show what was created
print_msg "Release created successfully!"
print_success "Tag: v$NEW_VERSION"
print_success "Package built in dist/"

# Optional: push tag and commit
read -p "Push tag and commits to remote? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_msg "Pushing commits and tag to remote..."
    if git push origin "$CURRENT_BRANCH" && git push origin "v$NEW_VERSION"; then
        print_success "Tag and commits pushed!"
    else
        print_error "Push failed. You may need to push manually."
        print_warning "  git push origin $CURRENT_BRANCH"
        print_warning "  git push origin v$NEW_VERSION"
    fi
else
    print_warning "Remember to push:"
    print_warning "  git push origin $(git rev-parse --abbrev-ref HEAD)"
    print_warning "  git push origin v$NEW_VERSION"
fi

print_msg "Release process complete!"
print_msg "The version will now be $NEW_VERSION for tagged commits."