# Coding My Own Large Language Model (LLM)

## Overview
I'm fascinated by AI, and indeed, I have a hard time understanding how anybody can't be fascinated by it. The first time I used ChatGPT felt like magic, and I wanted to better understand how a large language model (LLM) works. There's no better way to learn something than by going back to first principles and building it from the ground up. This repo is the result of that effort, including all the code I've written to implement and pretrain a LLM. I've also included Python notebooks for fine-tuning the model for a classification task (determining whether a text message is spam/not spam) and to act as a personal assistant.

I want to credit Sebasstian Raschka's [Build a Large Language Model from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) for teaching me the process and how a LLM works under the hood.

## Code

### Preparing the data
LLMs require text to be first converted into vector embeddings. This is accomplished with the create_embeddings method in the embeddings module. That module, in turn, uses the dataset and dataloader modules to prepare and load the training data.

### Attention Mechanism
The LLM's attention mechanism is implemented in the attention module. This is a multiheaded attention mechanism that is used within the transformer block (see below).

### LLM Architecture
The LLM architecture is implemented in the gpt_model module. The basic building block of the model is the transformer block, which consists of a multiheaded attention mechanism (implemented in the attention module) and a feed forward neural network. The feed forward neural network uses a GELU (Gaussian error linear unit) activation function. Additionally, layer normalization is used to improve training.

### Pretraining the LLM
The code for pretraining and evaluating the model and for generating text is found in the pretrain_model module. The default text for used to pretrain the model is an Edith Wharton short story, "The Verdict". A model trained on a such a small dataset will be pretty limited, but it has the benefit of being able to run in minutes on a M3 Mac. The same code can be used to train more powerful models given access to more data and GPUs.

OpenAI has released the weights for its GPT2 model. Code for loading these pretrained weights into the model is included in the load_pretrained_weights module (and its dependency, the gpt_download module).

### Finetuning
Also included are two Python notebooks that finetune the model for specific tasks:

1. spam_classifier: This notebook includes code to finetune the model on a classification task--classifying text messages as spam or not spam.

2. supervised_instruction: This notebook includes code to finetune the model to act as a personal assistant, training it on an instruction dataset.