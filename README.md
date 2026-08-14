<div align="center">

# Muhammet Dikmetaş

**AI Engineer · NLP / LLM / RAG**

*I don't stop at calling the API — I derive the math behind it.*

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00?style=flat-square&logo=graphql&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white)

</div>

---

Computer Engineering student building **RAG pipelines, semantic search and agentic LLM systems**.
Background in C/C++ and algorithms — so I treat a model as differentiable math, not a black box.

---

### 🧠 NLP & Deep Learning

- **Neural nets** — backpropagation derived by hand, softmax + cross-entropy gradient (`ŷ − y`), Adam, LayerNorm, vanishing gradients
- **Text → vector** — Word2Vec (CBOW / Skip-gram, negative sampling), GloVe co-occurrence objective, and why static vectors can't disambiguate *"bank"*
- **Attention** — `softmax(QKᵀ/√d_k)V`; the `√d_k` prevents softmax saturation. Multi-head = parallel relational subspaces
- **Q, K, V** — self-attention takes all three from one sequence; in **cross-attention Q comes from the decoder, K and V from the encoder output** — that's the entire bridge
- **Masking** — the decoder masks causally because teacher forcing would otherwise let token `t` copy `t+1`. The **encoder has no causal mask**: understanding needs both directions
- **Positional encoding** — sinusoidal (relative offsets as linear transforms), learned, **RoPE**, ALiBi. Attention is permutation-equivariant without it
- **Encoder-only vs decoder-only** — BERT is bidirectional and built for *representation* (classification, embeddings, retrieval); GPT is causal and built for *generation*. Decoder-only scaled better because every token is a training signal and the KV cache makes inference cheap

### 🧭 Embeddings & Semantic Search

- **Sentence-Transformers** — siamese bi-encoder + **mean pooling** (beats `[CLS]`), trained with **MultipleNegativesRankingLoss / InfoNCE** on in-batch negatives
- **Bi-encoder vs cross-encoder** — retrieve top-50 fast, rerank top-5 accurately
- **Cosine vs dot product** — identical on L2-normalized vectors; unnormalized, dot rewards magnitude. Cosine measures direction only, which is why it's the default for text
- **Model choice** — MiniLM for speed, mpnet / bge / e5 for quality, multilingual for Turkish

### 🔍 RAG & Vector Databases

- **Ingestion** — chunking strategy, overlap, metadata filtering
- **Indexing** — Flat vs **HNSW** vs IVF-PQ; the recall / latency / memory triangle
- **Hybrid search** — dense + BM25 fused with RRF, MMR for diversity, cross-encoder reranking
- **Adaptive RAG** — retrieval grading → hallucination grading → answer grading, with query rewrite and web-search fallback, built as a **LangGraph** state machine on Pydantic-typed decisions

### 🤖 Orchestration & Local LLMs

- **LangChain** — LCEL chains, prompt templates, output parsers, memory, streaming, LangServe, LangSmith tracing
- **LangGraph** — state schemas, conditional edges, self-correcting cycles
- **Agents** — ReAct loop, tool calling, Tavily search, agent memory
- **Self-hosted** — Ollama / llama.cpp, GGUF quantization and VRAM budgeting, fully offline RAG, LoRA/QLoRA vs prompting: knowing when to stop climbing

### 🛠️ Also

Computer Vision (CNN, OpenCV, ViT) · C/C++ · SQL · pandas / scikit-learn · RPA automation · Git, Docker

---

### 🚀 Projects

| Project | Focus |
|---|---|
| **Adaptive RAG Agent** | Self-correcting retrieval graph — LangGraph, Pydantic, Tavily |
| **Semantic Search Engine** | Hybrid dense + BM25 with cross-encoder reranking |
| **Local RAG Assistant** | Fully offline document Q&A — Ollama + Chroma |
| **Credit Risk & Price Prediction** | End-to-end ML pipeline |

**Currently:** Hugging Face internals · fine-tuning embedding models on Turkish data · RAG evaluation

---

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammet-dikmeta%C5%9F-3b55b7252/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MuhammetDikmetas)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muhammetdikmetas756@gmail.com)

Open to **AI / NLP engineering** roles and collaboration.

</div>
