echo "Setting up Python virtual environment and installing dependencies..."
sudo apt install python3-pip python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt