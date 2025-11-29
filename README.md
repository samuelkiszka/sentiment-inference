# Sentiment inference from IMDB movie reviews dataset
Author: Samuel Kiszka (xkiszk00@vutbr.cz)  
Date: November 2025

This repository contains code for training and evaluating a sentiment analysis model on the IMDB movie reviews dataset. 
The model classifies movie reviews as either positive or negative based on their text content.

This model was created as project for the course "Soft Computing" at Brno University of Technology.

The model itself is build purely in NumPy, 
but the data input pipeline uses PyTorch and the Hugging Face Transformers library to create embeddings from the text data.


## Requirements
- Python 3.12
- matplotlib
- PyTorch
- Transformers
- datasets


## Project structure
- `train.py`: Main training script, which uses other modules in `src/` to load data, create the model, train it, and evaluate it.
- `src/`: Contains all the source code for data loading, model definition, training, and evaluation.
  - `data_loader.py`: Code for loading and preprocessing the IMDB dataset.
  - `transformer_encoder.py`: Definition of the sentiment analysis model.
  - `trainer.py`: Code for training and evaluating the model.
- `review_classification.py`: Script for classifying new movie reviews using a trained model.
- `scripts/setup_env.sh`: Script to set up a virtual environment and install required packages.
- `scripts/download_default_model.sh`: Script to download a pre-trained model for inference.
- `requirements.txt`: List of required Python packages.
- `src/train_torch.ipynb`: PyTorch implementation of the training process as a reference.  
In the process of training, further folders are created:
  - `models/`: Parameters of the trained models.
  - `data/`: Cached datasets.
  - `results/`: Plots of training and validation metrics.

## Project setup
(For SFC environment)

### Clone the repository
  ```bash
  git clone https://github.com/samuelkiszka/sentiment-inference.git
  cd sentiment-inference
  ```
  You may need to install git first.


### Prepare SFC environment
It takes some time.
  ```bash
  sudo bash ./scripts/setup_sfc_env.sh
  ```


### Setup the environment and install the required packages
This is also slow, depending on the internet connection.
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
   This will create a virtual environment and install all required packages from `requirements.txt`. 


### Train the model 
   First startup may take longer as the IMDB dataset and pre-trained transformer model for text embedding will be downloaded and cached.
  ```bash
  python train.py
  ```
   This will start the training process, which will take some time depending on your hardware and chosen parameters.
   Possible parameters and default values can be checked by running:
  ```bash
  python train.py --help
  ```
  Alternatively, you can download a pre-trained model for inference using:
  ```bash
  . ./scripts/download_default_model.sh
  ```


### Classify new reviews:  
  ```bash
  python review_classification.py --mode [test | interactive]
  ```
You can test the model on new reviews [interactive], or check how it performs on test dataset [test].  
You can change model parameters and path in the script or by adding command line arguments.  
!!! But be aware that the model must be already trained and saved in the specified path with specified parameters.
