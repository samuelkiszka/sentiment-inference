echo "Download default model if not present..."
echo "Default parameters:"
echo "N_LAYERS: 2, N_HEADS: 4, D_FF: 64"

MODEL_DIR="models/sample_250"
MODEL_FILE="$MODEL_DIR/best_model_l2h4ff64.npz"
DEFAULT_MODEL_URL="https://nextcloud.fit.vutbr.cz/s/inwnzKp6A39rbcD/download"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Default model not found. Downloading..."
    echo "Model URL: $DEFAULT_MODEL_URL"
    mkdir -p "$MODEL_DIR"
    wget -O "$MODEL_FILE" "$DEFAULT_MODEL_URL"
    echo "Default model downloaded to $MODEL_FILE"
else
    echo "Default model already exists at $MODEL_FILE"
fi