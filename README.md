# LLM finetuning (text complation)
This repository guides through the process of fine-tuning a text completion (not instruct) dense base LLM (like Llama 2/3 or Mistral) from content.

## Forethought
This is a thought experiment and only for practice/educational purpose. The code in this repository in its current form is not production-ready.

## Requirements
- **Hugging Face account**: We will upload the training dataset to Hugging Face, because the `SFTTrainer` can very easily load datasets from there.
- **Example dataset**: Have some txt/epub files ready for acquiring training data. If you do not have one, you can download books from https://www.gutenberg.org/ - Just to be on the safe side of things, only download books where the copyright status is **Public domain**.
- **A workstation for data preparation**: This can be any computer, preferably its something local as for best results we'll need to do a bit of data cleaning manually.
- **A few euros**: For renting a GPU powered container instance for training - I will list alternatives later in the section - or a workstation with a modern NVidia GPU with at least 16GB of VRAM. Currently the training library do not support AMD/Intel GPUs.
- **Patience and stable internet connection**: The fine-tuning process takes a few hours at least and will involve transferring a few gigabytes of models.


# Step 1: Preparation of training data
At this point you hopefully selected a few books you like, and downloaded them in txt format.

For the sake of this training, I will use the Gutendex API (https://gutendex.com/) to query The Gutenberg projects for horror topic books:

`python .\pipeline\step0-acquire.py --output_dir "./training-data/0_raw/" --topic horror --num_records 200`

This will download a bunch of text files to the library that we can work with.

# Step 2: Preprocessing the training data
This is basic cleaning of the training data. Book contents here are stripped of Project Gutenberg prefix and postfix metadata. 

`python .\pipeline\step1-preprocess.py --output_dir "./training-data/1_preprocessed/" --input_dir "./training-data/0_raw/"`

If you have time, it is **highly recommended** to go through the preprocessed text files one by one, manually (or with more sophisticated automation) and remove even more filler content at the beginning and at the end stuff such as contents, acknowledgement, chapter headers, etc.

# Step 3: Chunking

# Step 4: Hugging Face dataset upload

# Step 5: Setting up training environment

# Step 6: Executing the training itself & Quantizing the model

# Step 7: Trying out locally


# Step 8: Publishing your model
## On Hugging Face

## On the Ollama registry 