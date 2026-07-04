# 🏦 Banking AI Assistant – Domain-Specific LLM Fine-Tuning

## Overview

Banking AI Assistant is a domain-specific conversational chatbot built by fine-tuning **Qwen2.5-3B-Instruct** using **QLoRA (4-bit Quantization + LoRA)**. The assistant is designed to answer banking-related queries such as account opening, KYC, NEFT, RTGS, loans, debit cards, internet banking, and other customer support questions.

Instead of training a Large Language Model (LLM) from scratch, this project applies **Parameter-Efficient Fine-Tuning (PEFT)** to adapt a pretrained model using a custom banking question–answer dataset.

---

## Features

* Domain-specific Banking AI Assistant
* QLoRA-based parameter-efficient fine-tuning
* 4-bit model quantization for memory-efficient training
* Banking conversational dataset in chat format
* Supports banking queries such as:

  * Savings & Current Accounts
  * KYC
  * NEFT
  * RTGS
  * IMPS
  * Loans
  * Debit/Credit Cards
  * Internet & Mobile Banking
* Interactive chatbot inference

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* TRL (SFTTrainer)
* PEFT
* BitsAndBytes
* QLoRA
* Qwen2.5-3B-Instruct

---

## Project Architecture

```text
Banking Dataset
        │
        ▼
Data Preprocessing
(Query → Messages Format)
        │
        ▼
Qwen Chat Template
        │
        ▼
Tokenizer
        │
        ▼
QLoRA Fine-Tuning
(PEFT + TRL + Transformers)
        │
        ▼
LoRA Adapter
        │
        ▼
Inference Pipeline
        │
        ▼
Banking AI Assistant
```

---

## Dataset Format

The dataset follows the conversational format expected by Qwen.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What documents are required to open a savings account?"
    },
    {
      "role": "assistant",
      "content": "Banks generally require identity proof, address proof, PAN or Form 60, and passport-size photographs."
    }
  ]
}
```

---

## Model

* Base Model: **Qwen2.5-3B-Instruct**
* Fine-Tuning Method: **QLoRA**
* Quantization: **4-bit (NF4)**
* Framework: **Hugging Face Transformers**
* Trainer: **TRL SFTTrainer**

---

## Fine-Tuning Pipeline

1. Prepare banking question–answer dataset.
2. Convert data into Qwen chat message format.
3. Load Qwen2.5-3B-Instruct in 4-bit precision.
4. Prepare the model for k-bit training.
5. Apply LoRA adapters using PEFT.
6. Fine-tune with TRL SFTTrainer.
7. Save LoRA adapters.
8. Load the model for inference.

---

## Example

**User**

```
What is NEFT?
```

**Assistant**

```
NEFT (National Electronic Funds Transfer) is an electronic payment system that enables secure fund transfers between bank accounts across India.
```

---

## Project Structure

```text
banking-ai-assistant/
│
├── dataset/
│   └── banking_dataset.json
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── qlora_training.ipynb
│   └── inference.ipynb
│
├── banking-chatbot-lora/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Integrate Retrieval-Augmented Generation (RAG) with official banking documents.
* Add multilingual support for regional languages.
* Develop a FastAPI backend with a React or Streamlit frontend.
* Evaluate the model using response accuracy and hallucination metrics.
* Deploy the chatbot on a cloud platform for public access.

---

## Skills Demonstrated

* Large Language Models (LLMs)
* Domain-Specific Fine-Tuning
* QLoRA
* LoRA
* Parameter-Efficient Fine-Tuning (PEFT)
* Hugging Face Transformers
* TRL SFTTrainer
* PyTorch
* Model Quantization
* Conversational AI
* Prompt Formatting
* Inference Pipeline

---

## License

This project is intended for educational and research purposes.
