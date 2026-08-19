#!/usr/bin/env bash
set -euo pipefail

if [[ ! -e /dev/kfd ]]; then
  echo "ROCm device /dev/kfd is missing. Install a supported amdgpu/ROCm host driver." >&2
  exit 1
fi

if [[ ! -d /dev/dri ]]; then
  echo "DRI device directory /dev/dri is missing." >&2
  exit 1
fi

docker compose down
docker compose -f docker-compose.rocm.yaml up --build "$@"
