# A Fine-grained Sentiment Analysis of App Reviews using LLMs: An Evaluation Study 

<!--The code in this repository evaluates the performance of GPT-4, ChatGPT, LLama-2-Chat-7b, LLama-2-Chat-13b, and LLama-2-Chat-70b for identifying app features with their associated sentiments from app reviews.-->

This repository contains the implementation of the paper "A Fine-grained Sentiment Analysis of App Reviews using Large Language Models: An Evaluation Study", which evaluates the performance of LLMs for identifying app features with their associated sentiments in app reviews. LLM performance (_i.e._ GPT-4, ChatGPT, LLaMA-2-Chat-7b, LLaMA-2-Chat-13b, and LLaMA-2-Chat-70b) is evaluated using zero-shot and few-shot (_i.e._ 1-shot, 5-shot) in-context learning strategies.

<!--[![arXiv](https://img.shields.io/badge/arXiv-2209.08163-b31b1b.svg)](https://arxiv.org) --->


![https://arxiv.org/abs/2409.07162](https://img.shields.io/badge/arXiv-2409.07162-b31b1b)


[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1vi9NeZgUpHY7MOu0rCefBIb9ztwPw255?usp=sharing) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://ahmed.jp/project_page/App_LLMs_2024/app_llms.html)

<!--

[![Static Badge](https://img.shields.io/badge/paper-demo-blue)
](./python_scripts/paper_demo.ipynb)

-->
## Project Structure

The project is organized into the following folders:

### 1. `dataset`
- `Ground_truth.xlsx`: It contains review data with labeled features and sentiments. The data is publicly available at [https://github.com/jsdabrowski/IS-22/](https://github.com/jsdabrowski/IS-22/)
### 2. `prompt_data`
- `long_prompt.txt`: Stores the longer version of the prompt text to identify features with their associated sentiments in a review text.
- `short_prompt.txt`: Stores the shorter version of the prompt text to identify features with their corresponding sentiments in a review text.
- `few_shot_examples.json`: Stores 10 review sentences with labeled features and associated sentiments being used for few-shot evaluation.
### 3. `python_scripts`
- `evaluation.py`: The class evaluating the extracted features and sentiments by comparing them to true features and sentiments using exact and partial matching.
- `feature_level_sentiment_analysis.py`: The main class prompts LLMs to extract app features and corresponding sentiments, and logs the feature extraction and sentiment prediction performances at the app level and overall dataset.
- `utility.py`: Python script containing useful utility functions used in Feature_Level_Sentiment_Analysis.py
### 4. `results`
Running the main script "feature_level_sentiment_analysis.py" generates detailed performance analysis of LLM (i.e., feature extraction performance, sentiment prediction performance) at the app level, per iteration, and dataset level and saves them in this folder.

To run the experiments, perform the following steps:
1. Create a conda environment with Python 3.8.13
2. Run `pip install -r requirements.txt`
3. Set the secret key to the environment variable "OPENAI_API_KEY" for evaluating OpenAI chat models
4. Go to the project directory
4. Enter the command "cd python_scripts"
### Evaluating the performance of zero-shot with short or long prompts
Use the following commands to evaluate the zero-shot performance of GPT family models (i.e., ChatGPT, GPT-4)
```bash
# Evaluating ChatGPT (default model) with the "short" prompt under zero-shot settings
python feature_level_sentiment_analysis.py --prompt_type "short"

# Evaluating ChatGPT (default model) with the "long" prompt under zero-shot settings
python feature_level_sentiment_analysis.py --prompt_type "long"

# Evaluating gpt-4 with the "short" prompt under zero-shot settings
python feature_level_sentiment_analysis.py --llm_model "gpt-4" --prompt_type "short"

# To evaluate GPT-4 with the "long" prompt under zero-shot settings 
python feature_level_sentiment_analysis.py --llm_model "gpt-4" --prompt_type "long"

# If an environment variable "OPENAI_API_KEY" is not set, assign the OpenAI secret key to parameter "secret_key" as shown in the following command
python feature_level_sentiment_analysis.py --llm_model "gpt-4" --secret_key "PUT_SECRET_KEY_HERE" --prompt_type "long"
```
Use the following commands to evaluate the zero-shot performance of LLama chat models (i.e., llama-2-7b-chat-hf, llama-2-13b-chat-hf,llama-2-70b-chat-hf) with the short prompt or long prompts. Ensure the secret key from huggingface has been set to variable "hf-token".
```bash
# Evaluating LLama 7b model with the "short" prompt under zero-shot settings. Similarly, pass the bigger llama-2-chat model for its evaluation
python feature_level_sentiment_analysis.py --llm_model "meta-llama/Llama-2-7b-chat-hf" --secret_key %hf-token% --prompt_type "short" 

# To evaluate LLama 7b with the "long" prompt under zero-shot settings. Similarly, pass the bigger llama-2-chat model for its evaluation 
python feature_level_sentiment_analysis.py --llm_model "meta-llama/Llama-2-7b-chat-hf" --secret_key %hf-token% --prompt_type "long"
```
### Evaluating 1-shot and 5-shot performance of LLMs with the short prompt
Use the following commands to evaluate the 1-shot and 5-shot performance of GPT family models 
```bash
# To evaluate the 1-shot performance of ChatGPT with the "short" prompt
python feature_level_sentiment_analysis.py --few_shot_examples 1

# To evaluate 1-shot performance of gpt-4 with the "short" prompt
python feature_level_sentiment_snalysis.py --llm_model "gpt-4" --few_shot_examples 1

# To evaluate the 5-shot performance of ChatGPT with the "short" prompt
python feature_level_sentiment_snalysis.py  --few_shot_examples 5

# To evaluate 5-shot performance of gpt-4 with the "short" prompt
python feature_level_sentiment_analysis.py --llm_model="gpt-4" --prompt_type "short" --few_shot_examples 5
```
Use the following commands to evaluate the 1-shot and 5-shot performances of Llama-2-chat models
```bash
# To evaluate 1-shot performance of LLama-2-7b-chat-hf with the "short" prompt
python feature_level_sentiment_analysis.py --llm_model "meta-llama/Llama-2-7b-chat-hf" --secret_key %hf-token% --few_shot_examples 1

# To evaluate 5-shot performance of LLama-2-7b-chat-hf with the "short" prompt
python feature_level_sentiment_snalysis.py  --llm_model "meta-llama/Llama-2-7b-chat-hf" --secret_key %hf-token% --few_shot_examples 5
```
Running the script "feature_level_sentiment_analysis.py" will create a folder with the name of the LLM model inside the folder "results". All the detailed performance analyses of the LLM at the feature level, sentiment level for each app, and individual iteration will be put in this folder. This folder also contains overall summary reports of model's performances for predicting feature and sentiment tasks. 
