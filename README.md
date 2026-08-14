<div align="center">

# Muhammet Dikmetaş

### AI Engineer · NLP / LLM / RAG Systems

**I don't stop at calling an API — I derive the math behind it.**
Retrieval-augmented and agentic LLM systems, from the attention equations up to a production LangGraph.

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00?style=for-the-badge&logo=graphql&logoColor=white)

![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=databricks&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammet-dikmetas/)
[![Mail](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muhammetdikmetas756@gmail.com)

</div>

---

## 🎯 About

Computer Engineering student, focused on **Natural Language Processing and Large Language Model systems**.

My starting point was low-level engineering — C and C++, data structures, algorithm design — and that shaped how I approach AI: **a model is not a black box, it's a composition of differentiable operations.** So when I learn a method, I learn the loss function, the gradient flow, and the reason the architecture is shaped the way it is — not just the library call.

Today I build **RAG pipelines, semantic search systems, and agentic LLM workflows**, and I run them both on hosted APIs and on **self-hosted local models**.

> **Ask me about:** why attention is scaled by `√d_k` · why the encoder has no causal mask · why cosine similarity and dot product are the same thing on normalized vectors · why a bi-encoder retrieves and a cross-encoder reranks.

---

## 🧠 Core Competencies

<details open>
<summary><b>🔢 Neural Network Foundations — derived, not memorized</b></summary>

<br/>

I can write these on a blank sheet of paper, which is the only test that matters:

- **Forward & backward propagation** — the chain rule walked layer by layer through affine transforms and non-linearities
- **Softmax + cross-entropy** — why the gradient collapses to the elegant `ŷ − y`, and why that makes the pair numerically stable when fused
- **Gradient pathologies** — vanishing/exploding gradients, saturating activations, gradient clipping, residual connections as gradient highways
- **Weight initialization** — Xavier/Glorot vs. He, and why the choice follows the activation function's variance behaviour
- **Optimizers** — SGD → Momentum → RMSProp → **Adam** (first/second moment estimates + bias correction), AdamW's decoupled weight decay
- **Normalization** — BatchNorm vs. **LayerNorm**, and why sequence models with variable-length batches demand LayerNorm; Pre-LN vs. Post-LN stability
- **Regularization** — dropout's train/inference asymmetry (inverted dropout scaling), L2 / weight decay, early stopping

</details>

<details>
<summary><b>📐 Text → Vector: Word2Vec & GloVe</b></summary>

<br/>

The algorithms that turned text into geometry. These are the foundation everything else stands on.

**Word2Vec**

- **CBOW** (context → center) vs. **Skip-gram** (center → context), and why Skip-gram wins on rare words while CBOW is faster
- The softmax bottleneck over `|V|` and the two escapes: **negative sampling** (binary logistic objective with `k` noise samples drawn from the `U(w)^{3/4}` distribution) and **hierarchical softmax** (Huffman tree, `O(log|V|)`)
- Two separate matrices — input embeddings `W` and output embeddings `W'` — and exactly which gradients update which
- Subsampling of frequent words, and the dynamic context window

**GloVe**

- Global **co-occurrence matrix** `X` instead of local windows — a count-based method wearing a prediction-based coat
- Weighted least-squares objective: `J = Σ f(X_ij)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log X_ij)²`
- The weighting function `f(x)` and why it must be capped — otherwise frequent pairs dominate the loss

**The shared limitation:** both produce *static* vectors. One vector for "bank," no matter whether the sentence is about rivers or money. That limitation is precisely what contextual models were invented to fix.

</details>

<details open>
<summary><b>⚡ Transformers & Attention — where I go deepest</b></summary>

<br/>

**Scaled dot-product attention**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

- The `√d_k` is not decoration. With `d_k`-dimensional unit-variance vectors, the dot product has variance `d_k`; without scaling the softmax saturates into a near one-hot distribution and gradients vanish. Scaling restores unit variance.
- **Multi-head attention** — `h` heads at `d_model/h` dimensions each. Same parameter budget, but each head learns a different relational subspace (syntactic dependency, coreference, positional locality). One wide head can only average them.
- **Complexity** — `O(n²·d)` in sequence length. This single fact drives KV caching, FlashAttention's IO-aware tiling, sliding-window and sparse attention, and every long-context trick in the field.

**Attention variants — who provides Q, K and V**

| Variant | Q from | K, V from | Purpose |
|---|---|---|---|
| **Encoder self-attention** | encoder states | encoder states | Bidirectional context — every token sees every token |
| **Masked (causal) self-attention** | decoder states | decoder states | Autoregressive generation — token `t` sees only `≤ t` |
| **Cross-attention** | **decoder** states | **encoder** output | The bridge: the decoder queries the source sequence |
| **Grouped-Query / Multi-Query** | all heads | shared K/V heads | Shrinks the KV cache for fast inference |

The cross-attention row is the whole encoder–decoder story in one line: **the decoder asks the question (Q), the encoder holds the answer (K, V).**

**Masking — the question people get wrong**

- The **decoder** masks because it is trained with teacher forcing on the full target sequence in parallel. Without a causal mask, position `t` would attend to position `t+1` and simply copy the answer — the model would learn nothing and collapse at inference time. Implementation: add `−∞` to the pre-softmax scores in the upper triangle, so `softmax` sends them to exactly zero.
- The **encoder** has *no* causal mask, because its job is **understanding**, not generation. To classify a token or embed a sentence you want the full left *and* right context — that bidirectionality is the entire point of BERT. The encoder only uses a **padding mask**, which is a batching detail, not an architectural one.
- So: causal masking is a consequence of the *training objective*, not of the architecture.

**Positional encoding**

Attention is permutation-equivariant — it is a set operation. Without positional information "the dog bit the man" and "the man bit the dog" are identical inputs. The remedies:

- **Sinusoidal** (original) — fixed `sin/cos` at geometrically spaced frequencies. Its elegance: `PE(pos+k)` is a *linear transform* of `PE(pos)`, so relative offsets are learnable, and it extrapolates beyond training length
- **Learned absolute** (BERT, GPT-2) — simple, effective, hard-capped at the trained context length
- **RoPE** — rotates Q and K in 2D subspaces by an angle proportional to position, so the dot product depends only on *relative* distance. The modern default (LLaMA, Qwen, Mistral)
- **ALiBi** — a linear distance penalty added directly to attention scores; strong length extrapolation, no embedding at all

**Architecture choice — why encoder-only vs. decoder-only**

| | **Encoder-only** (BERT, RoBERTa, E5) | **Decoder-only** (GPT, LLaMA, Mistral) |
|---|---|---|
| **Attention** | Bidirectional, no causal mask | Causal, strictly left-to-right |
| **Objective** | Masked LM — denoising ~15% of tokens | Next-token prediction |
| **Optimized for** | Understanding & representation | Generation & continuation |
| **Native output** | A vector per token / per sequence | A probability distribution over the vocabulary |
| **Use it for** | Classification, NER, retrieval, **embeddings**, reranking | Chat, RAG generation, agents, code, summarization |
| **Can't do well** | Generate fluent free text | Produce a symmetric sentence embedding without tricks |

Why decoder-only won the scaling race: every token is a training signal (MLM wastes 85% of positions per step), the objective is uniform and self-supervised on raw text, and the **KV cache** makes autoregressive inference cheap — reuse past keys/values, compute one new column per step. Encoder-decoder (T5, BART) still holds ground where input and output are structurally different sequences — translation, summarization, seq2seq.

But the encoder never left. **Every RAG system I build has a decoder-only model writing the answer and an encoder-only model finding the evidence.**

</details>

<details open>
<summary><b>🧭 Embeddings & Semantic Search</b></summary>

<br/>

**From static to contextual**

BERT gives "bank" a different vector in "river bank" and "investment bank" — context flows through every layer. But raw BERT is a *poor* sentence encoder out of the box: its `[CLS]` token is trained for NSP, not similarity, and untuned cosine scores between BERT sentence vectors are close to worthless.

**Sentence-Transformers — how the embedding is actually produced**

1. A transformer encoder produces token vectors
2. A **pooling layer** collapses them into one sentence vector — **mean pooling** (attention-mask-weighted average) beats `[CLS]` pooling in almost every benchmark, and max pooling is rarely competitive
3. The network is trained **siamese** — the same weights encode both texts, so the two vectors land in one shared space
4. The loss shapes the geometry directly:
   - **MultipleNegativesRankingLoss / InfoNCE** — for every anchor, the paired positive must beat all other in-batch items as negatives. Larger batch → harder negatives → better embeddings. This is the workhorse.
   - **CosineSimilarityLoss** — MSE against a labelled similarity score, when you have graded pairs
   - **TripletLoss** — `max(0, d(a,p) − d(a,n) + margin)`, when you have explicit hard negatives

**Bi-encoder vs. cross-encoder — the architectural trade-off that defines retrieval**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | Two texts encoded **separately** | The pair encoded **together** |
| Attention across texts | ❌ None | ✅ Full |
| Corpus vectors | Precomputable, indexable | Must recompute per query |
| Cost for `N` docs | `1` encode + `N` cheap similarities | `N` full forward passes |
| Accuracy | Good | Higher |

Hence the standard two-stage design: **bi-encoder retrieves top-50 fast, cross-encoder reranks to top-5 accurately.**

**Cosine similarity vs. dot product**

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)
```

The formula is trivial; the *decision* is not.

- On **L2-normalized** vectors, cosine and dot product are **identical** — and `‖a − b‖² = 2 − 2·cos(a,b)`, so Euclidean distance ranks identically too. Most modern embedding models normalize by default, which makes the choice moot for them.
- On **unnormalized** vectors they diverge: dot product rewards magnitude. That is a *feature* when the norm carries meaning (term importance, document popularity, confidence) and a *bug* when it doesn't — long documents get an unearned boost purely for being long.
- Cosine measures **direction only** — pure semantic orientation, length-invariant. That is why it is the default for text similarity.
- Practical note: many ANN indexes only implement inner product, so you normalize at write time and let dot product *become* cosine.

**Choosing a model**

`all-MiniLM-L6-v2` for speed and prototypes · `all-mpnet-base-v2` for quality/English · `bge` and `e5` families for state-of-the-art retrieval (with their `query:` / `passage:` prefixes) · **multilingual models for Turkish** · always check MTEB before committing, and always check dimensionality against your index budget.

</details>

<details open>
<summary><b>🔍 RAG & Vector Databases</b></summary>

<br/>

**Ingestion**

- **Chunking** — the most underrated variable in the entire pipeline. Fixed-size, recursive-character, and semantic (embedding-drift) splitting; overlap to protect cross-boundary meaning; the tension between chunks small enough for precise retrieval and large enough to stay self-contained
- Metadata attached at write time — source, section, date — because metadata filtering is what turns retrieval into *scoped* retrieval

**Indexing & search**

- **Index structures** — Flat (exact, `O(N)`), **HNSW** (navigable small-world graph; `M` and `ef_search` trade recall against latency), IVF-PQ (coarse partitioning + product quantization for memory-bound scale)
- Approximate nearest neighbour is a **recall/latency/memory triangle** — you pick two
- **Hybrid search** — dense embeddings capture meaning, **BM25** captures exact tokens (product codes, names, acronyms). Fused with **Reciprocal Rank Fusion**, they cover each other's blind spots
- **MMR** for diversity when top-k returns five near-duplicate chunks
- **Reranking** with a cross-encoder before anything reaches the context window

**Advanced / agentic RAG** — built as an explicit **LangGraph** state machine rather than a linear chain:

```
Query → Route → Retrieve → Grade documents ─┬─ relevant ──→ Generate → Grade for
                    ↑                       │                          hallucination
                    │                       └─ irrelevant ─→ Web search      ↓
                    └───────────── Rewrite query ←──────── fails ←── Grade answer
```

Every grader is a structured-output LLM call validated by **Pydantic**, so the graph routes on typed decisions instead of parsed strings. The result is a system that notices when its own retrieval failed and does something about it.

**Evaluation** — retrieval and generation are graded separately: context precision/recall for the retriever, faithfulness and answer relevance for the generator. A RAG system without evaluation is a demo, not a product.

</details>

<details open>
<summary><b>🤖 LLM Orchestration & Agents</b></summary>

<br/>

**LangChain** — LCEL composition, prompt templates, structured output parsers, message history and memory management, token-level streaming, **LangServe** deployment, **LangSmith** tracing and evaluation

**LangGraph** — typed state schemas, conditional edges, cycles and self-correction loops, checkpointing. The mental shift: an LLM application is a **state machine**, not a pipeline

**Agents** — the **ReAct** loop (reason → act → observe → repeat), tool calling and schema design, web search integration via **Tavily**, agent memory, and the failure modes that matter: infinite loops, tool-selection errors, and context overflow

**Production concerns** — prompt versioning, cost/latency tracing, output validation, graceful degradation

</details>

<details open>
<summary><b>🖥️ Local & Self-Hosted LLMs</b></summary>

<br/>

Not every problem should leave the building. I run models locally when **data privacy, cost, latency, or vendor independence** outweigh the last few points of benchmark score.

- **Runtimes** — Ollama for fast iteration, llama.cpp underneath, vLLM's continuous batching + PagedAttention for serving throughput
- **Quantization** — GGUF `Q4_K_M` / `Q5_K_M` / `Q8_0`, AWQ and GPTQ; the practical question is always *how much quality am I trading for how much VRAM* — and how to compute that budget before downloading 40 GB
- **KV cache math** — why context length costs memory linearly and why GQA/MQA exist
- **Fully offline RAG** — local embedding model + local vector store + local generator, with no token ever crossing the network boundary
- **Adaptation strategy** — prompt engineering → few-shot → RAG → **LoRA/QLoRA** → full fine-tuning, in that order of cost. Most problems are solved before step four, and knowing *when to stop climbing* is the actual skill

</details>

<details>
<summary><b>👁️ Computer Vision</b></summary>

<br/>

- CNN architectures — convolution/pooling arithmetic, receptive fields, residual blocks, transfer learning
- **OpenCV** pipelines — preprocessing, classical detection, image transforms
- Vision Transformers — patch embeddings, and how the same attention machinery transfers from tokens to image patches
- Multimodal embeddings (CLIP-style) — where vision and NLP meet in a shared vector space

</details>

<details>
<summary><b>🛠️ Engineering Foundations</b></summary>

<br/>

- **C / C++** — structured programming, data structures, algorithm design, memory model
- **Python** — application development, clean project structure, `dotenv`-based configuration, virtual environments
- **Data Science** — pandas / NumPy / scikit-learn, feature engineering, statistical analysis, model evaluation
- **Databases & SQL** — querying, schema design, data management
- **RPA** — business process automation, applied on real production workflows
- **Tooling** — Git, Docker, REST APIs, deployment

</details>

---

## 🚀 Selected Work

| Project | What it demonstrates | Stack |
|---|---|---|
| **Adaptive RAG Agent** | Self-correcting retrieval graph with document grading, hallucination detection, answer grading and web-search fallback | LangGraph · Pydantic · Vector DB · Tavily |
| **Semantic Search Engine** | Hybrid dense + BM25 retrieval with cross-encoder reranking over a custom corpus | Sentence-Transformers · FAISS · RRF |
| **Local RAG Assistant** | Fully offline document Q&A — no data leaves the machine | Ollama · local embeddings · Chroma |
| **Credit Risk & Price Prediction** | End-to-end ML: data cleaning → feature engineering → modelling → evaluation | scikit-learn · pandas |
| **Computer Vision Projects** | CNN architectures and classical CV pipelines for image analysis | PyTorch · OpenCV |
| **RPA Automation (Internship)** | Real-world business process automation with ML components | RPA · Python |

> 📌 *Pinned repositories below — each with a README covering the approach, the trade-offs, and the results.*

---

## 🌱 Currently Working On

- **Hugging Face ecosystem in depth** — `transformers` internals, `datasets`, `PEFT`, `sentence-transformers` training loops
- **Fine-tuning embedding models on Turkish data** — because multilingual off-the-shelf models leave real performance on the table for Turkish
- **RAG evaluation frameworks** — moving from "it looks right" to measured faithfulness and context recall
- **Inference optimization** — quantization trade-off curves and serving throughput

---

## 📊 GitHub

<div align="center">

![Stats](https://github-readme-stats.vercel.app/api?username=MuhammetDikmetas&show_icons=true&theme=tokyonight&hide_border=true&count_private=true)
![Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=MuhammetDikmetas&layout=compact&theme=tokyonight&hide_border=true&langs_count=8)

</div>

---

<div align="center">

## 📫 Get in Touch

Open to **AI / NLP engineering** roles, internships, and collaboration on LLM and retrieval systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammet%20Dikmeta%C5%9F-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammet-dikmetas/)
[![Email](https://img.shields.io/badge/Email-muhammetdikmetas756@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:muhammetdikmetas756@gmail.com)

<br/>

<sub><i>"Understand the gradient, and the architecture explains itself."</i></sub>

</div>
