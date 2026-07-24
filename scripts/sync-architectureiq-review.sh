#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SITE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE="${ARCHITECTUREIQ_REVIEW_SOURCE:-$SITE_DIR/../ArchitectureIQ/docs/0718_project_review.html}"
DESTINATION="$SITE_DIR/public/blogs/architectureiq-project-review/index.html"

if [ ! -f "$SOURCE" ]; then
  printf 'Source document not found: %s\n' "$SOURCE" >&2
  exit 1
fi

cp "$SOURCE" "$DESTINATION"
printf 'Synced %s -> %s\n' "$SOURCE" "$DESTINATION"
