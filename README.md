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

</div>

---

Computer Engineering student building **RAG pipelines, semantic search and agentic LLM systems**.
Background in C/C++ and algorithms — I treat a model as differentiable math, not a black box.

### 🧠 Transformers & NLP

- **Neural nets** — backprop by hand, softmax + cross-entropy gradient (`ŷ − y`), Adam, LayerNorm
- **Text → vector** — Word2Vec (Skip-gram, negative sampling), GloVe co-occurrence objective, and why static vectors can't disambiguate *"bank"*
- **Attention** — `softmax(QKᵀ/√d_k)V`; the `√d_k` prevents softmax saturation. Multi-head = parallel relational subspaces
- **Q, K, V** — self-attention draws all three from one sequence; in **cross-attention Q comes from the decoder, K and V from the encoder**
- **Masking** — the decoder masks causally so teacher forcing can't leak token `t+1`. The **encoder has no causal mask** — understanding needs both directions
- **Positional encoding** — sinusoidal, learned, **RoPE**, ALiBi; attention is permutation-equivariant without it
- **Encoder-only vs decoder-only** — BERT is bidirectional, built for *representation*; GPT is causal, built for *generation*. Decoder-only scaled better: every token is a training signal and the KV cache makes inference cheap

### 🧭 Embeddings & Retrieval

- **Sentence-Transformers** — siamese bi-encoder + **mean pooling** (beats `[CLS]`), trained with **MultipleNegativesRankingLoss / InfoNCE**
- **Bi-encoder vs cross-encoder** — retrieve fast, rerank accurately
- **Cosine vs dot product** — identical on L2-normalized vectors; unnormalized, dot rewards magnitude
- **Vector DBs** — HNSW vs IVF-PQ, chunking strategy, hybrid dense + BM25 with RRF, MMR
- **Adaptive RAG** — retrieval / hallucination / answer grading with query rewrite and web fallback, as a **LangGraph** state machine

### 🤖 Orchestration & Local LLMs

- **LangChain / LangGraph** — LCEL chains, memory, streaming, LangSmith tracing, self-correcting graphs
- **Agents** — ReAct loop, tool calling, Tavily search
- **Self-hosted** — Ollama, GGUF quantization, VRAM budgeting, fully offline RAG, LoRA/QLoRA vs prompting

### 🛠️ Also

Computer Vision (CNN, OpenCV, ViT) · C/C++ · SQL · pandas / scikit-learn · RPA · Git, Docker

---

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammet-dikmeta%C5%9F-3b55b7252/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MuhammetDikmetas)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muhammetdikmetas756@gmail.com)

Open to **AI / NLP engineering** roles and collaboration.

</div>
