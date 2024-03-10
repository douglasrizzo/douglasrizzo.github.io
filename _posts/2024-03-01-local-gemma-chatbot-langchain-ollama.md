---
layout: post
title: Running a Gemma-powered question-answering chatbot locally with LangChain + Ollama
categories: llms python langchain ollama google-gemma mistral
---

Last week, I started my quest to build my first LLM-powered chatbot that runs locally and executes RAG on my [Obsidian](https://obsidian.md/) document base. You can see it [here]({% post_url 2024-02-19-llm-qa-obsidian-rag %}).

This post is a follow-up to that one. The highlights of this one are:

1. I will test the brand new [Google Gemma](https://ai.google.dev/gemma) family of models and qualitatively compare it to a Mistral model of same size.
2. The chatbot will run 100% locally in my computer with a GTX 1070, 7h gen i7 and 32 GB of RAM, using Chroma for the vector store and [Ollama](https://ollama.com/) for the models.
3. Unlike in the [previous post]({% post_url 2024-02-19-llm-qa-obsidian-rag %}), we'll configure the correct prompt templates for our instruction-tuned models.

Also, since [my last post]({% post_url 2024-02-19-llm-qa-obsidian-rag %}), I've refactored the code base in its current state as both a Python package and a command-line tool, which you can find in [this link](https://github.com/douglasrizzo/langsidian). This is what I used to run the experiments shown in this page. A sample notebook is provided [here](https://github.com/douglasrizzo/langsidian/blob/master/2%20-%20Langsidian%20as%20a%20package.ipynb) and [the underlying code](https://github.com/douglasrizzo/langsidian/tree/master/langsidian) is < 200 lines long!

## Test-driving Gemma-7b

A few days ago, Google released [Gemma](https://ai.google.dev/gemma), a family of lightweight models the same size as Mistral-7b, the one I was already using.

My first attempt at loading Gemma-7b was through [Hugging Face](https://huggingface.co/google/gemma-7b-it). However, it needs extra configuration to fit in my memory, i.e. float16/bfloat16 quantization with PyTorch, or 8bit/4bit quantization with bitsandbytes. Also, the model files are huge, totalling 20 GB.

For now, let's familiarize ourselves with Ollama, which is simpler, and leave the Hugging Face ecosystem for later.

## Enter Ollama

It turns out a very widespread way of running models locally is through [Ollama](https://ollama.com/), which is very simple to install on Linux.

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama has a Docker-like interface (it was made by [an ex-Docker employee](https://twitter.com/jmorgan)) and models can be downloaded using their names and some basic tags. Downloaded models can be executed via python using their [official library](https://pypi.org/project/ollama/).

With that, I decided to stop using models from multiple libraries and frameworks and run all of them in Ollama. The commands below downloaded Mistral-7b, Gemma-7b, and a text embedding model from [Nomic.AI](https://home.nomic.ai/).

```sh
ollama pull mistral:7b-instruct
ollama pull gemma:7b-instruct
ollama pull nomic-embed-text
```

The model files downloaded by Ollama are considerably smaller (each 7b model fit in a single 5GB file). They also loaded into my GPU memory, alongside the embedding model.

## Configuring the correct prompt templates

It turns out the prompt templates, especially for Instruct models, need to use the correct tokens and that is not done automatically by LangChain or any other library.

I went after the correct templates on the Ollama website. They are shown below. Contrast with [my last post]({% post_url 2024-02-19-llm-qa-obsidian-rag %}#retrieval-qa-with-custom-prompt-template), in which I copied an internal prompt from LangChain, which was not suitable for any specific model.

While tweaking the contents of the prompt template, I realized how much variability in the responses I got from only minor tweaks. For example, if I told Gemma something like *"if you don't know the answer, just say you don't know and not make anything up"*, the model would just refuse to answer any questions, even if the information was present in the retrieved text chunks. To make the comparison fair, I also removed the same section from the mistral prompt template, which I used in the [previous post]({% post_url 2024-02-19-llm-qa-obsidian-rag %}#retrieval-qa-with-custom-prompt-template).

### Mistral prompt template

Sources: [Ollama](https://ollama.com/library/mistral:7b-instruct) and [Hugging Face Hub](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

```text
<s>[INST] Use the following pieces of context to answer the question at the end. Present a well-formatted answer, using Markdown if possible. Don't go over three paragraphs when answering.
---
{context}
---
Question: {question} [/INST]
```

### Gemma prompt template

Sources: [Ollama](https://huggingface.co/google/gemma-7b-it) and [Hugging Face Hub](https://ollama.com/library/gemma:7b-instruct).

```text
<start_of_turn>user
Use the following pieces of context to answer the question at the end. Present a well-formatted answer, using Markdown if possible. Don't go over three paragraphs when answering.
---
{context}
---
Question: {question}<end_of_turn>
<start_of_turn>model
```

## Tweaking RAG parameters

One problem I was having was that performing maximal marginal similarity on a small number of chunks (e.g. 3) tended to result only a single helpful text chunk. To fix that I

- reduced the chunk sizes
- retrieved more chunks

Some final tweaking that made the output of both models better was using the following parameters:

- Chunk size: 400 characters
- Chunk overlap: 50 characters
- Chunks returned by similarity for [MMR](https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types/mmr) search: 10
- Final number of chunks returned by [MMR](https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types/mmr): 6
- [MMR](https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types/mmr) $\lambda$ (0 = maximum diversity among chunks; 1 = no diversity): 0.7

## Running the models

Since my last post, I've refactored the code base in its current state as both a Python package and a command-line tool, which you can find in [this link](https://github.com/douglasrizzo/langsidian). This is what I used in the construction of this post.

The snippet below (which is also available as a [Jupyter Notebook](https://github.com/douglasrizzo/langsidian/blob/master/2%20-%20Langsidian%20as%20a%20package.ipynb)) showcases how to instantiate and talk to the LLM-powered Q&A bot:

```py
from pathlib import Path
from langsidian import ChatBot, DocumentBase

bot = ChatBot(
  docs_path=(Path.home("Documents", "Obsidian")),
  vectorstore_db_path=Path("docs/chroma"),
  document_type=DocumentBase.OBSIDIAN,
  model_type="mistral:7b-instruct",
  embeddings="nomic",
)

q = "What is zero-shot learning?"
print(f"**Question: {q}**\n\n Answer: {bot.answer(q)}", end="\n\n---\n\n")
```

It can also be executed as a standalone module. The document path and model name can be selected with command-line arguments:

```sh
python -m langsidian
```

```text
[Prompt]: Who were the authors of the Deep Q-Network paper?
[Assistant]:  The authors of the Deep Q-Network paper were Mnih, Vaswani, Graves, et al. (2015). You can find their paper here: <https://arxiv.org/abs/1312.5602>.
```

## Results

Below, I asked the same questions to both Mistral-7b-it and Gemma-7b-it, giving them access to my Obsidian document base, whose chunks are now embedded using [Nomic Embed](https://blog.nomic.ai/posts/nomic-embed-text-v1).

Analyzing the output qualitatively, I found that Mistral tends to give more informative and helpful answers. It also makes better use of the information in the chunks provided by the RAG procedure, creating richer answers, whereas Gemma tends to repeat itself or cut answers short when it hits a dead-end such as in the answer for [this question](#gemma-7b-1).

Also, in [the first question](#gemma-7b), Gemma explained terms that may have been in the context chunks recovered for RAG (such as $a'$ and $\pi_*$), but they did not appear in its final answer, which would definitely confuse users in a conversation.

---

### Question: Give me the equation for the action value function update and explain its terms.

#### Mistral-7b

The equation for the action-value function update using the n-step return is as follows:

$$Q_{t+n}(S_t, A_t) \leftarrow Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - \gamma^n Q_{t+n-1}(S_t, A_t)]$$

This equation is a form of the Bellman equation for the action-value function. Let's break down its terms:

- $Q_{t+n}$: The action-value function at time step $t+n$ for state-action pair $(S_t, A_t)$.
- $\alpha$: The learning rate or step size, which determines how much new information is incorporated in each update.
- $G_{t:t+n}$: The n-step return from time step t to time step $t+n$. It's the total reward obtained by following policy $\pi$ from state $S_t$ and taking action $A_t$, plus all the discounted future rewards, i.e., $G_{t:t+n} = R_{t+1:t+n}+ \gamma V_{\pi}(S_{t+n})$
- $\gamma$: The discount factor, which is used to discount future rewards by a certain percentage to make them present values.
- $Q_{t+n-1}$: The old estimate of the action-value function at time step $t+n-1$ for state-action pair $(S_t, A_t)$.

This update rule aims to minimize the difference between the current and new estimates of $Q(S_t,A_t)$, gradually improving the estimation as more samples are collected.

#### Gemma-7b

**Equation for Action Value Function Update:**

$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - \gamma^n Q_{t+n-1}(S_t, A_t)]$$

**Explanation of Terms:**

- **$Q_{t+n}(S_t, A_t)$:** The action-value function at time $t+n$ for state $S_t$ and action $A_t$.
- **$Q_{t+n-1}(S_t, A_t)$:** The action-value function at time $t+n-1$ for state $S_t$ and action $A_t$.
- **$\alpha$:** The learning rate, which controls the rate of learning.
- **$G_{t:t+n}$:** The n-step return, which is the total reward from time $t$ to time $t+n$.
- **$\gamma$:** The discount factor, which accounts for the value of future rewards.
- **$a'$:** The action that is not taken at time $t$.
- **$\pi_*$:** The optimal policy.

**Note:** This equation is for the action-value function update in Q-learning, not the state-value function update.

---

### Question: What is the architecture of the Deep Q-Networks?

#### Mistral-7b

The Deep Q-Network (DQN) architecture uses a deep neural network to approximate the Q* function in reinforce learning. The input to the network are 4 frames of 84x84 pixel screenshots from Atari 2600 games, only using the luminance values of the pixels. The network then uses three convolutional layers to learn visual representations of the game states. The output of the neural network is the Q values of all possible actions for each state, ranging from 4 to 18.

This architecture builds upon the Q-learning algorithm, which aims to find the optimal policy by iteratively improving the Q values based on observed state-action pairs and their corresponding rewards. However, directly applying this algorithm to deep neural networks with large input spaces is infeasible due to the computational demands.

To overcome this challenge, DQN uses experiences from multiple frames and applies a target network for more stable learning. The target network is updated periodically to match the weights of the online network, ensuring that the Q values in the target network do not change too frequently during training. Additionally, the authors use experience replay, where they store transitions in a buffer and sample them randomly to train the network, which helps in reducing correlation between consecutive samples and improving stability.

These architectural enhancements allowed DQN to learn complex visual representations of Atari games, achieve human-level performance on several benchmark tasks, and set a new standard for deep reinforce learning research.

#### Gemma-7b

The text describes the architecture of Deep Q-Networks, which uses a deep neural network to approximate the Q* function. The neural network has three convolution layers to learn visual representations of the game states.

The text does not describe the architecture of the convolutional layers, therefore I cannot answer the question.

---

### Question: What is the difference between causal language modeling and masked language modeling?

#### Mistral-7b

Causal and masked language modeling are two different methods used for pre-training language models.

In **causal language modeling**, also known as autoregressive language modeling, the language model is trained to predict the next token in a sequence given the context of the previous tokens. The model has access to the entire sequence during training and prediction, and its task is to generate the next token based on the context provided. However, since there is no ground truth available during inference, this can lead to a discrepancy between training and inference, known as Exposure Bias. This issue can be addressed through techniques like Curriculum learning.

In **masked language modeling**, also known as Masked Language Modeling (MLM), tokens are randomly masked at training time and the model is trained to reconstruct the original text by predicting the masked tokens. For instance, in the sentence "the teacher teaches the student", the word "teaches" might be masked as "<MASK>". During training, the model is presented with the masked token and its context, and its task is to learn to fill in the masked token based on the context provided. BERT is an example of a masked language model. The main difference between causal and masked language modeling lies in how the models are trained and what they are optimized for: causal models predict the next token given the context, while masked models learn to fill in missing tokens based on their context.

Sources: DeepLearning.AI Natural Language Processing Specialization, Generative AI with Large Language Models.

#### Gemma-7b

Causal language modeling and masked language modeling are two methods for pre-training language models.

Causal language modeling allows the model to see the first tokens of the sequence and its task is to predict the next token. Masked language modeling, on the other hand, randomly masks tokens at training time and the model is trained to reconstruct the original text by predicting the masked tokens.

The main difference between causal language modeling and masked language modeling is that in causal language modeling, the model has access to the first tokens of the sequence and its task is to predict the next token, while in masked language modeling, the model is trained to reconstruct the original text by predicting the masked tokens.

---

### Question: What is zero-shot learning?

#### Mistral-7b

Zero-shot learning is a concept used in machine learning, specifically in the context of Generative AI and Large Language Models (LLMs), to describe the capability of an LLM to predict classes it was not explicitly trained on without being given any explicit examples during training or at inference time. In other words, the model makes predictions based on its existing knowledge, rather than learning from new examples provided in real-time. This is in contrast to few-shot learning, where the model is given a few examples of the task it's expected to perform before being presented with the actual task. Zero-shot learning requires a large and diverse dataset for the model to generalize effectively to unseen classes or concepts.

One limitation of zero-shot learning is that it may not be as effective as few-shot learning, especially when dealing with complex tasks or rare classes. Another limitation is that it relies heavily on the quality and representativeness of the training data for making accurate predictions. However, since zero-shot learning allows models to execute new tasks without being given any explicit examples, it can be beneficial in scenarios where acquiring labeled data is difficult, costly, or time-consuming.

Regarding your additional question: In Q-learning, importance sampling is not used because it is an off-policy reinforce learning algorithm that learns from a sequence of state-action pairs (trajectories) generated by a policy π, which may not be the optimal one. The goal is to estimate the action-value function Q(s,a), and importance sampling would introduce additional complexity without significant benefits in this context.

#### Gemma-7b

Zero-shot learning is the capability of a model to predict classes it was not explicitly trained on. In the context of Generative AI and LLMs, zero-shot learning refers to the capability of an LLM to execute new tasks without being given any explicit examples.

In zero-shot learning, the model is given no examples of the task it is expected to perform, and it must generalize from its previous experiences to complete the task. This contrasts with few-shot learning, which involves providing the model with a few examples of the task it is expected to perform.

For example, if you ask a model to classify a review as positive or negative, and it has never seen a review with that sentiment before, zero-shot learning would involve the model making a generalization based on its previous experiences to determine whether the review is positive or negative.

---

### Question: Explain to me the concept of bucketing in RNNs

#### Mistral-7b

Bucketing in Recurrent Neural Networks (RNNs) is a technique used for batching sequences while minimizing padding. The main goal is to process sequences of varying lengths efficiently and maintain a consistent batch size during training. Here's a step-by-step explanation:

1. **Divide data into buckets:** Divide the input data into `m` buckets based on sequence lengths. Each bucket represents a specific range of sequence lengths.
2. **Sample sequences from a single bucket:** For each batch, sample all sequences from only one of the buckets. This ensures that most sequences in a batch have similar lengths, reducing the need for extensive padding.
3. **Pad sequences to the largest one:** Pad all sequences in the batch to match the size of the longest sequence. This guarantees that every sequence receives the same amount of processing during each time step.

Bucketing is an essential technique when dealing with sequence data, as it helps to minimize padding and improve training efficiency. However, it might require more computational resources due to the need to process multiple buckets for generating batches over time.

It's important to note that this method assumes that sequences within a bucket are independent of one another. If there is any dependence between sequences in a bucket, this assumption would not hold true, and other data processing techniques might be required.

#### Gemma-7b

Bucketing in RNNs is a technique for batching sequences in a way that minimizes padding. The idea is to group sequences of similar length into buckets and then process the buckets separately.

The process of bucketing involves dividing the data into buckets, sampling sequences from a single bucket, and padding all sequences in a batch to the size of the largest sequence in the batch. This technique is useful for RNNs because it allows for the efficient processing of sequences of varying lengths.

The technique is illustrated in the provided text with two images. The first image shows the process of dividing the data into buckets, and the second image shows the process of padding the sequences in a batch.

---

### Question: What is a named entity?

#### Mistral-7b

A named entity is a real-world object that can be denoted with a proper name, examples include persons, locations, organizations, products, geographical entities, geopolitical entities, time indicators, artifacts, events, and natural phenomena. Named Entity Recognition (NER) systems are used to automatically locate named entities from unstructured text. This process is important for various applications such as improving search engine efficiency by scanning websites and storing named entities for later matching with user queries, and enhancing recommendation systems by extracting named entities from a user's search history to identify their interests.

For instance, in the sentence "Sharon flew to Miami on Friday," the named entities are "Sharon" (person), "Miami" (location), and "Friday" (time indicator). NER systems can accurately identify and tag these named entities within the text. The remaining tokens that do not represent named entities are typically tagged as filler words or `O`.

#### Gemma-7b

A named entity is a real-world object that can be denoted with a proper name. It can be abstract or have a physical existence. Examples of named entities include people, locations, organizations, and products.

Named entities are extracted from text and are used to identify and classify named objects in a sentence. They are used in various applications, such as search engine efficiency, recommendation systems, and information extraction.

In the text provided, the sentence "Sharon flew to Miami on Friday" would have the named entities "Sharon," "Miami," and "Friday."

## Conclusions

On this post, I show you how I locally executed an LLM-powered Q&A chatbot that performs RAG on an Obsidian, using Mistral-7b and Gemma-7b instruction fine-tuned models as well as Nomic-Embed, all provided by Ollama.

I showed you how I did and you can:

- Install Ollama;
- Download Ollama models;
- Configure prompt templates for the instruction-tuned versions of Gemma and Mistral, and;
- Configure some search parameters to get better chunks and RAG results.
