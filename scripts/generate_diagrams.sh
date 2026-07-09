#!/bin/bash
# Script to generate SVG and PDF diagrams from PlantUML files

# Check if PlantUML is installed
if ! command -v plantuml &> /dev/null
then
    echo "PlantUML could not be found. Please install it first."
    exit 1
fi

# Target file to process (first argument)
TARGET_FILE=$1

if [ -z "$TARGET_FILE" ]; then
    echo "Usage: ./generate_diagrams.sh <path/to/file.puml>"
    exit 1
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: File '$TARGET_FILE' does not exist."
    exit 1
fi

echo "Processing $TARGET_FILE..."

# Generate SVG securely
if plantuml -tsvg "$TARGET_FILE"; then
    echo "✓ SVG generated successfully."
else
    echo "❌ Error: Failed to generate SVG from $TARGET_FILE"
    exit 1
fi

# Get filenames
DIR_NAME=$(dirname "$TARGET_FILE")
BASE_NAME=$(basename "$TARGET_FILE" .puml)
SVG_FILE="$DIR_NAME/$BASE_NAME.svg"
PDF_FILE="$DIR_NAME/$BASE_NAME.pdf"

# Generate PDF from the valid SVG using rsvg-convert to bypass Java/Batik missing dependencies
if command -v rsvg-convert &> /dev/null; then
    if rsvg-convert -f pdf -o "$PDF_FILE" "$SVG_FILE"; then
        echo "✓ PDF generated successfully: $PDF_FILE"
    else
        echo "❌ Error: Failed to convert SVG to PDF."
        exit 1
    fi
else
    echo "⚠️ Warning: rsvg-convert not found. Cannot generate PDF. Please install it (e.g. brew install librsvg)."
fi

echo "✅ Diagram generation finished for $TARGET_FILE"
