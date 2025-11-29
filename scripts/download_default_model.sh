echo "Download default model if not present..."

MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/default_model.npz"
DEFAULT_MODEL_URL="https://nextcloud.fit.vutbr.cz/s/93fwg3bGGKGCzZP/download"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Default model not found. Downloading..."
    echo "Model URL: $DEFAULT_MODEL_URL"
    mkdir -p "$MODEL_DIR"
    wget -O "$MODEL_FILE" "$DEFAULT_MODEL_URL"
    echo "Default model downloaded to $MODEL_FILE"
else
    echo "Default model already exists at $MODEL_FILE"
fi