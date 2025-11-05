# 📚 Bilingual RAG Course Chatbot (English + Arabic)

An intelligent chatbot that helps students learn from any textbook by answering questions in both English and Arabic using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-121212)

## 🌟 Overview

This project implements a RAG-powered chatbot that transforms any textbook into an interactive learning companion. Upload a PDF in English or Arabic, ask questions naturally, and get accurate answers backed by the source material. The system automatically detects the language and responds appropriately.

## ✨ Key Features

- **PDF Upload Interface** — Upload any English or Arabic textbook or course material
- **Intelligent Document Processing** — Automatic chunking and vector storage using FAISS
- **Natural Conversation** — Chat-based Q&A interface for intuitive interaction
- **True Bilingual Support** — Seamlessly handles Arabic and English questions and documents
- **Optimized Performance** — 4-bit quantized models for efficient inference on 12GB GPUs
- **Context-Aware Answers** — Retrieves relevant chunks before generating responses

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Document Loading** | PyPDFLoader (LangChain) | PDF text extraction |
| **Text Processing** | RecursiveCharacterTextSplitter | Document chunking |
| **Vector Database** | FAISS | Embedding storage and retrieval |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Text vectorization |
| **LLM** | Qwen2.5-1.5B-Instruct (4-bit) | Answer generation |
| **Quantization** | bitsandbytes | Memory optimization |
| **Language Detection** | langdetect | Arabic/English detection |
| **Framework** | LangChain | RAG pipeline orchestration |

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (12GB+ recommended)
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/hamdi-404/Bilingual-RAG-Course-Chatbot.git
cd bilingual-rag-chatbot
```

## 📖 Usage

1. **Upload a PDF**: Click the upload button and select your textbook or course material
2. **Wait for Processing**: The system will chunk and embed the document
3. **Ask Questions**: Type your question in English or Arabic
4. **Get Answers**: Receive context-aware responses based on the uploaded material

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Storage
                                                            ↓
User Question → Language Detection → Query Embedding → Retrieval
                                                            ↓
Retrieved Chunks + Question → LLM (Qwen2.5) → Final Answer
```

## 🎯 Future Enhancements

- [ ] Support for additional languages
- [ ] Multi-document support
- [ ] Citation of specific page numbers
- [ ] Export conversation history
- [ ] Fine-tuning on educational datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Qwen2.5 model
- LangChain for the RAG framework
- Streamlit for the intuitive UI framework

## 📧 Contact

 Hamdi - (hamdi404.cs@gmail.com)
 LinkedIn: https://www.linkedin.com/in/hamdi-mohammed-a0314b213/
