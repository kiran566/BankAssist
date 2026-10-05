# Banking AI Assistant — LLM Fine-Tuning using QLoRA

A domain-specific banking AI assistant built by fine-tuning **Qwen2.5-3B-Instruct** with **QLoRA** on banking query-response data and evaluating the generated responses using **LangSmith**.

## Overview

The project focuses on adapting a general-purpose instruction-tuned LLM for banking-related conversations while keeping GPU memory usage low through 4-bit quantization and LoRA adapters.

The fine-tuned model is evaluated on a separate set of banking questions using **LangSmith** and an **LLM-as-a-Judge** approach.

## Features

* Fine-tuned **Qwen2.5-3B-Instruct** on custom banking query-response data.
* Used the **Qwen chat template** to format conversational training examples.
* Applied **QLoRA with 4-bit NF4 quantization** to reduce GPU memory usage.
* Used **LoRA adapters** so that only a small portion of the model parameters are trained.
* Built the training pipeline using **Transformers, TRL SFTTrainer, PEFT, BitsAndBytes, and PyTorch**.
* Saved the trained **LoRA adapter and tokenizer** for later inference.
* Created a separate evaluation dataset containing banking questions and expected answers.
* Used **LangSmith** to run and track model evaluations.
* Used a **Groq-hosted LLM as an evaluator** to judge the correctness of generated responses.

## Project Flow

```text
Banking Dataset
      ↓
Chat Template Formatting
      ↓
Qwen2.5-3B-Instruct
      ↓
4-bit Quantization + LoRA
      ↓
QLoRA Fine-Tuning
      ↓
LoRA Adapter
      ↓
Banking AI Assistant
      ↓
Evaluation Dataset
      ↓
LangSmith
      ↓
LLM-as-a-Judge
      ↓
Evaluation Score
```

## Fine-Tuning

The model was fine-tuned using **QLoRA**, which combines:

* 4-bit quantization of the base model
* Frozen base model weights
* Trainable LoRA adapters

Instead of updating all parameters of the 3B parameter model, LoRA trains a much smaller set of additional parameters.

### Training Stack

* Python
* Qwen2.5-3B-Instruct
* Hugging Face Transformers
* PEFT
* QLoRA
* LoRA
* BitsAndBytes
* TRL
* SFTTrainer
* PyTorch
* CUDA
* Google Colab

## Dataset

The training data contains banking-related conversational examples covering topics such as:

* Savings accounts
* KYC
* NEFT
* RTGS
* Loans
* Digital banking
* Account-related queries

The conversations are formatted using the Qwen chat template before supervised fine-tuning.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What documents are required to open a savings account?"
    },
    {
      "role": "assistant",
      "content": "Banks generally require identity proof, address proof, PAN or Form 60, and other documents may be required depending on the bank."
    }
  ]
}
```

## Evaluation with LangSmith

After fine-tuning, the model is loaded with the saved LoRA adapter and evaluated separately from the training process.

The evaluation flow is:

```text
Test Question
     ↓
Qwen2.5-3B + LoRA
     ↓
Generated Answer
     ↓
Compare with Expected Answer
     ↓
Groq LLM Judge
     ↓
Correctness Score
     ↓
LangSmith Experiment
```

LangSmith stores the evaluation runs so that the generated answer, expected answer, evaluator result, and score can be inspected.

### LLM-as-a-Judge

A separate LLM is used as an evaluator rather than manually checking every generated response.

For example:

```text
Question:
What is NEFT?

Expected Answer:
NEFT is a banking payment system used to electronically
transfer money between bank accounts.

Model Answer:
NEFT allows electronic transfer of funds between bank accounts.

        ↓

LLM Judge

        ↓

Correctness Score + Reason
```

The **Groq model is only used as the evaluation judge**. The actual banking assistant remains the fine-tuned **Qwen2.5-3B-Instruct + LoRA** model.

## Model Saving

The fine-tuning process saves the LoRA adapter and tokenizer:

```python
trainer.save_model("./banking-chatbot-lora")
tokenizer.save_pretrained("./banking-chatbot-lora")
```

The adapter can later be loaded on top of the original Qwen2.5-3B-Instruct base model for inference.

## Technologies

**Python | Qwen2.5-3B-Instruct | Hugging Face Transformers | PEFT | LoRA | QLoRA | BitsAndBytes | TRL | PyTorch | CUDA | LangSmith | Groq | Google Colab**
