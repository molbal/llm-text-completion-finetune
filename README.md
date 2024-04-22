# LLM finetuning (text complation)
This repository guides through the process of fine-tuning a text completion (not instruct) dense base LLM (like Llama 2/3 or Mistral) from content.

## Forethought
This is a thought experiment and only for practice/educational purpose. The code in this repository in its current form is not production-ready.

## Requirements
- **Hugging Face account**: We will upload the training dataset to Hugging Face, because the `SFTTrainer` can very easily load datasets from there.
- **A workstation for data preparation**: This can be any computer with Python installed, preferably its something local as for best results we'll need to do a bit of data cleaning manually. (I used a Windows 11 laptop with Python 3.10)
- **5 euros or incredible patience**: For renting a GPU powered container instance for training - I will list alternatives later in the section, even free ones - or a workstation with a modern Nvidia GPU with at least 16GB of VRAM. _Currently_ the training library do not support AMD/Intel GPUs. (See: https://github.com/unslothai/unsloth/issues/37)


## Step 1: Acquiring training data
**What you will have at the end of this step: A directory with txt files in it**

Project Gutenberg is a digital library that provides free access to 70k+ e-books in the public domain. Since the books are in the public domain, there are no copyright issues or restrictions on using them for LLM training. 

I will use the Gutendex API (https://gutendex.com/) to query The Gutenberg projects for horror topic books:

`python .\pipeline\step0-acquire.py --output_dir "./training-data/0_raw/" --topic horror --num_records 200`

This will download a bunch of text files to the library that we can work with. 

## Step 2: Preprocessing the training data
**What you will have at the end of this step: A directory with txt files in it, which are cleared of artifacts, and errors**

This is basic cleaning of the training data. Book contents here are stripped of Project Gutenberg prefix and postfix metadata. 

`python .\pipeline\step1-preprocess.py --output_dir "./training-data/1_preprocessed/" --input_dir "./training-data/0_raw/"`

If you have time, it is **highly recommended** to go through the preprocessed text files one by one, manually or with more sophisticated automation and remove even more filler content at the beginning and at the end stuff such as contents, acknowledgement, chapter headers, etc.

**For production fine-tunes, we need squeaky clean training data.**

## Step 3: Chunking
**What you will have at the end of this step: A single parquet file containing the preprocessed training data in chunks.**

In this step, we chunk the training data into pieces. For chunk size, I do not have a golden rule. Smaller chunk size will result to faster learning, larger chunk size might learn more patterns from the training data. For me, 7000 characters yielded good results.

To run this snippet, you will need some dependencies:

`pip install pyarrow pandas nltk`

Installed depepndencies

- `pyarrow`: A library for working with the Apache Arrow data format, which is used for writing the Parquet file in this script.
- `pandas`: A popular data analysis and manipulation library for Python, which is used to create a DataFrame from the processed data.
- `nltk`: The Natural Language Toolkit library, which is used for sentence tokenization in this script.

Then once you have the necessary libraries installed, execute the script:

`python .\pipeline\step3-chunking.py --source_dir "./training-data/1_preprocessed/" --output_file "./training-data/data.parquet"`

## Step 4: Hugging Face dataset upload

Once we have the data.parquet file, we can upload it to Hugging Face ( https://huggingface.co/new-dataset ). I name it molbal/horror-novel-chunks as it is the topic I used for this example. Since this dataset is based on public domain works, I select the _unlicense_ license. I set public visibility, as I would like to share this with the community.


Once we created the dataset, we are greeted by an empty skeleton. I first edit the dataset card to list the novels I parsed.

![Empty dataset](images/step3-empty-dataset.png)

I will directly edit the readme markdown on the Hugging Face editor, but you can check the dataset and commit it with git if you prefer that. It suggests me to import a dataset card template, but it is way too detailed for our plain and simple use case now. For good measure, I listed the source files and linked the GitHub repository with the example scripts. (https://github.com/molbal/llm-text-completion-finetune)

I upload the data.parquet file as it is, commit it directly to the main branch, because brancing is unnecessary here, and here is the result: 
https://huggingface.co/datasets/molbal/horror-novel-chunks


## Step 5: Setting up training environment

## Step 6: Executing the training itself & Quantizing the model

## Step 7: Trying out locally


## Step 8: Publishing your model
### On Hugging Face

### On the Ollama registry 