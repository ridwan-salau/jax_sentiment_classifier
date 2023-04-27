# JAX/FLAX SENTIMENT CLASSIFIER WITH BERT
This repository implements sentiment classifier with Google Jax/Flax using HuggingFace Flax Bert interface as backbone. It also demonstrates the checkpointing of fine-tuned transformer models and loading the saved model for inference.

## Setup

### Environment setup and Installation

Setup your environment by running: `conda env create -f env.yml`.

Activate the environment by running `conda activate jax`

## How to Run training

The training file is [jax_sentiment.py](jax_sentiment.py). To fine-tune the model, run `python3 jax_sentiment.py`

The inference file is [jax_inference.py](jax_inference.py). To run inference, run the command `python3 jax_inference.py`

