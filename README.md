# Zero-Shot Text Classification by Natural Language Inference Fine-Tuning of a Large Language Model
This repository contains scripts and notebooks for fine-tuning an NLI language model for zero-shot text classification. The root folder contains notebooks I regularly use for categorizing chatbot prompts in my website project [Talking to Chatbots](https://talkingtochatbots.com). All the scripts, classes, and notebooks may be reused in any project involving natural language inference datasets and zero-shot text classification with large language models.

- Dataset available on Hugging Face: [reddgr/nli-chatbot-prompt-categorization](https://huggingface.co/datasets/reddgr/nli-chatbot-prompt-categorization))
- Model available on Hugging Face: [reddgr/zero-shot-prompt-classifier-bart-ft](https://huggingface.co/reddgr/zero-shot-prompt-classifier-bart-ft))

## Model: zero-shot-prompt-classifier-bart-ft

This model is a fine-tuned version of [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) on the [reddgr/nli-chatbot-prompt-categorization](https://huggingface.co/datasets/reddgr/nli-chatbot-prompt-categorization) dataset.

The purpose of the model is to help classify chatbot prompts into categories that are relevant in the context of working with LLM conversational tools: 
coding assistance, language assistance, role play, creative writing, general knowledge questions... 

The model is fine-tuned and tested on the natural language inference (NLI) dataset [reddgr/nli-chatbot-prompt-categorization](https://huggingface.co/datasets/reddgr/nli-chatbot-prompt-categorization)

Below is a confusion matrix calculated on zero-shot inferences for the 10 most popular categories in the Test split of [reddgr/nli-chatbot-prompt-categorization](https://huggingface.co/datasets/reddgr/nli-chatbot-prompt-categorization) at the time of the first model upload. The classification with the base model on the same small test dataset is shown for comparison:

![Zero-shot prompt classification confusion matrix for reddgr/zero-shot-prompt-classifier-bart-ft](https://talkingtochatbots.com/wp-content/uploads/2025/01/zero-shot-prompt-classification-comparison-10-classes-60-accuracy.png)

The current version of the fine-tuned model outperforms the base model [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) by 24 percentage points (60% accuracy vs 36% accuracy) in a test set with 10 candidate zero-shot classes (the most frequent categories in the test split of [reddgr/nli-chatbot-prompt-categorization](https://huggingface.co/datasets/reddgr/nli-chatbot-prompt-categorization)).

The chart below compares the results for the 12 most popular candidate classes in the Test split, where the base model's zero-shot accuracy is outperformed by 25 percentage points:

![Zero-shot prompt classification confusion matrix for reddgr/zero-shot-prompt-classifier-bart-ft](https://talkingtochatbots.com/wp-content/uploads/2025/01/zero-shot-prompt-classification-comparison-12-classes-57-accuracy.png)

The dataset and the model are continously updated as they assist with content publishing on my website [Talking to Chatbots](https://talkingtochatbots) 

## Dataset: reddgr/nli-chatbot-prompt-categorization

The dataset contains chatbot prompts annotated with natural language inference (NLI) category hypotheses and labels ({0: "contradiction", 1: "neutral", 2: "entailment"}).

The primary purpose is to perform natural language inference categorization of chatbot conversations, such as those shared by the author on Talking to Chatbots.

Category hypotheses (language, coding, role play, science...) are chosen as the most relevant in the context of chatbot conversations, whose language context and main use cases typically differ heavily from the text patterns and categories frequently found in the most popular NLI datasets sourced from news, scientific articles, news publications, etc.
