#!/usr/bin/env bash
set -euo pipefail

# Build the project for AMD GPUs (ROCm/HIP) inside a Docker container.
# The container used is rocm/dev-ubuntu-24.04:7.1.1-complete. We mount a host
# build directory so artifacts persist and are owned by the host user (not root).

IMAGE_DEFAULT="rocm/dev-ubuntu-24.04:7.1.1-complete"
IMAGE="${IMAGE:-$IMAGE_DEFAULT}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR_HOST="${BUILD_DIR:-$REPO_ROOT/build-amd}"
TARGET="${TARGET:-train}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--image IMAGE] [--build-dir DIR] [--target NAME] [--] [extra cmake args]

Environment variables (preferred over flags):
  IMAGE        Docker image to use (default: ${IMAGE_DEFAULT})
  BUILD_DIR    Host build directory (default: <repo>/build-amd)
  TARGET       Build target (default: train)

Examples:
  BUILD_DIR=./build-amd ./build_amd.sh
  ./build_amd.sh -- -DCMAKE_BUILD_TYPE=Release

Notes:
  - Runs CMake with -DUSE_ROCM=ON and disables CUDA-specific optional deps (NVML, cuFile, MPI).
  - Chooses Ninja if available in the container, otherwise Unix Makefiles.
  - Files in the build directory are created with your UID/GID (not root).
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --build-dir) BUILD_DIR_HOST="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$BUILD_DIR_HOST"

REPO_MNT="/src"
BUILD_MNT="/build"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "Using Docker image: $IMAGE"
echo "Repository root: $REPO_ROOT -> $REPO_MNT (ro)"
echo "Build directory: $BUILD_DIR_HOST -> $BUILD_MNT"
echo "Build target:    $TARGET"
echo "Extra CMake args:${EXTRA_ARGS[*]:+ ${EXTRA_ARGS[*]}}"

# Prepare the build script to be run as the host user
BUILD_SCRIPT="set -euo pipefail
GEN='Unix Makefiles'
if command -v ninja >/dev/null 2>&1; then GEN='Ninja'; fi
echo \"CMake generator: \${GEN}\"
cmake -S '$REPO_MNT' -B '$BUILD_MNT' -G \"\${GEN}\" \
  -DUSE_ROCM=ON -DUSE_CUFILE=OFF -DUSE_NVML=OFF -DUSE_MPI=OFF ${EXTRA_ARGS[*]}
cmake --build '$BUILD_MNT' --parallel --target '$TARGET'"

# Prepare the inner command.
# 1. Install dependencies as root.
# 2. Create a user matching the host UID/GID.
# 3. Switch to that user to run the build script.
INNER_CMD="set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y cmake ninja-build git

# Ensure group exists
if ! getent group $HOST_GID >/dev/null 2>&1; then
  groupadd -g $HOST_GID builder_group
fi

# Ensure user exists
if ! getent passwd $HOST_UID >/dev/null 2>&1; then
  useradd -u $HOST_UID -g $HOST_GID -m builder_user
fi

BUILD_USER=\$(getent passwd $HOST_UID | cut -d: -f1)

# Write build script to file. Quoted EOF prevents variable expansion by the container shell.
cat << 'EOF' > /tmp/build.sh
$BUILD_SCRIPT
EOF
chmod +x /tmp/build.sh

# Run the build as the user
su -p \"\$BUILD_USER\" -c /tmp/build.sh
"

echo "..."

docker run --rm \
  -v "$REPO_ROOT:$REPO_MNT:ro" \
  -v "$BUILD_DIR_HOST:$BUILD_MNT" \
  -w "$BUILD_MNT" \
  "$IMAGE" \
  bash -lc "$INNER_CMD"

echo "Build finished. Artifacts are in: $BUILD_DIR_HOST"
