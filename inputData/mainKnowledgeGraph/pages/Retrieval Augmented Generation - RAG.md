- ## RAG Frameworks and Tools

  **RAGFlow**
  RAGFlow is an open-source RAG engine based on deep document understanding, offering a streamlined workflow for businesses[2]. It provides scalable architecture and integration with various business data sources and LLMs[5].

  **Verba (The Golden RAGtriever)**
  Verba is an end-to-end, user-friendly RAG application designed for seamless data exploration and insight extraction[3]. It supports multiple data formats and integrates with Weaviate for vector storage[5].

  **FastGPT**
  FastGPT is a knowledge-based platform built on LLMs, offering out-of-the-box data processing and model invocation capabilities. It allows for workflow orchestration through Flow visualization[4].

  **Quivr**
  Quivr is a personal productivity assistant that uses RAG to chat with various document types (PDF, CSV, etc.) and apps. It leverages Langchain and supports multiple LLM providers[4].

  **Langchain-Chatchat**
  This project (formerly Langchain-ChatGLM) is based on Langchain and ChatGLM, providing local knowledge base question-answering capabilities[4].

  **AnythingLLM**
  AnythingLLM is an open-source multi-user ChatGPT alternative for various LLMs, embedders, and vector databases. It offers unlimited documents, messages, and users in a privacy-focused app[4].

  **QAnything**
  QAnything is a tool for question-answering based on any type of data[4].

  **Danswer**
  Danswer allows users to ask questions in natural language and get answers backed by private sources. It connects to tools like Slack, GitHub, and Confluence[4].

  **Rags**
  Rags enables building ChatGPT-like interfaces over your data, all using natural language[4].

  **Khoj**
  Khoj acts as a copilot to search and chat with your knowledge base (PDF, markdown, org) using RAG. It supports both powerful online LLMs (e.g., GPT-4) and private, offline models (e.g., Mistral)[4].
-
- ## Comparison Table
	- Stacks and frameworks available on GitHub.
		- | Feature | RAG Stack Lambda | RAGFlow | Verba | BERGEN | Korvus | fastRAG |
		  |---------|------------------|---------|-------|--------|--------|---------|
		  | Language | Go, React | Python | Python | Python | Python, JavaScript, Rust | Python |
		  | Deployment | AWS Lambda, API Gateway, CloudFront | Local and cloud | Local and cloud | Local | Local and cloud | Local |
		  | Vector DB | DynamoDB | Not specified | Weaviate | Not specified | PostgreSQL (pgvector) | Not specified |
		  | LLM Integration | Not specified | Multiple LLMs | Ollama, Huggingface, Anthropic, Cohere, OpenAI | Multiple models | Not specified | State-of-the-art LLMs |
		  | User Authentication | ✅ | Not specified | Not specified | Not specified | Not specified | Not specified |
		  | Middleware | ✅ | Not specified | Not specified | Not specified | Not specified | Not specified |
		  | Frontend | Vite, React, Tailwind | Not specified | User-friendly interface | Not specified | Not specified | Not specified |
		  | Data Formats | Not specified | Not specified | Multiple formats | Not specified | Not specified | Not specified |
		  | Benchmarking | Not specified | Not specified | Not specified | ✅ | Not specified | ✅ |
		  | In-database ML | Not specified | Not specified | Not specified | Not specified | ✅ | Not specified |
		  | Optimization Focus | Not specified | ✅ | Not specified | Not specified | Not specified | ✅ |
		  | Multi-modal Support | Not specified | Not specified | ✅ (Audio transcription) | Not specified | Not specified | Not specified |
		  | Hybrid Search | Not specified | Not specified | ✅ | Not specified | Not specified | Not specified |
		  | Chunking Techniques | Not specified | Not specified | Multiple (Token, Sentence, Semantic, etc.) | Not specified | Not specified | Not specified |
		  | Vector Visualization | Not specified | Not specified | ✅ | Not specified | Not specified | Not specified |
		  | Licence | Not specified | Apache 2.0 | BSD-3-Clause | Apache 2.0 | Open-source | Apache 2.0 |
- Notable points
- 1. RAG Stack Lambda offers a full-stack solution with AWS integration and user authentication.
- 2. RAGFlow emphasises deep document understanding and optimization for businesses.
- 3. Verba provides a user-friendly interface with extensive LLM integration and advanced features like hybrid search and multi-modal support.
- 4. BERGEN focuses on standardised benchmarking for RAG pipelines.
- 5. Korvus specializes in in-database machine learning using PostgreSQL
- 6. fastRAG concentrates on efficient and optimised RAG pipelines with state-of-the-art LLMs.
- When choosing a RAG stack, consider your specific requirements, such as deployment preferences, LLM integration needs, and desired features like benchmarking or in-database processing.
- [1] https://github.com/Melkeydev/rag-stack-lambda
  [2] https://gist.github.com/gubatron/79793e1102726174013ffde798df4d1f
  [3] https://github.com/finic-ai/rag-stack/actions
  [4] https://www.timescale.com/blog/rag-is-more-than-just-vector-search/
  [5] https://github.com/weaviate/Verba
  [6] https://airbyte.com/tutorials/end-to-end-rag-using-github-pyairbyte-and-chroma-vector-db
  [7] https://github.com/Andrew-Jang/RAGHub
  [8] https://github.com/Danielskry/Awesome-RAG
- ## Specialized RAG Tools
	- **TRT-LLM-RAG-Windows**
	  This is a developer reference project for creating RAG chatbots on Windows using TensorRT-LLM[4].
	- **GPT-RAG**
	  GPT-RAG core is a RAG pattern running in Azure, using Azure Cognitive Search for retrieval and Azure OpenAI large language models[4].
	- **RAG-Demystified**
	  This project presents an LLM-powered advanced RAG pipeline built from scratch[4].
	- **LARS**
	  LARS is an application for running LLMs locally on your device with your documents, facilitating detailed citations in generated responses[4].
	-
- ## RAG Optimization and Enhancement Tools
- **Sparrow**
- Sparrow focuses on data extraction using machine learning and LLMs[4].
- **Fastembed**
- Fastembed is a fast, accurate, and lightweight Python library for creating state-of-the-art embeddings[4].
- **Self-RAG**
- Self-RAG is a project exploring learning to retrieve, generate, and critique through self-reflection[4].
- **Instructor**
- Instructor serves as a gateway to structured outputs with OpenAI[4].
- **Swirl-Search**
- Swirl is open-source software that simultaneously searches multiple content sources and returns AI-ranked results[4].
- **Kernel-Memory**
  This tool allows indexing and querying of any data using LLMs and natural language, tracking sources and showing citations[4].
- **RAGFoundry**
- RAGFoundry is a framework for specializing LLMs for RAG tasks using fine-tuning[4].
- These projects offer a wide range of capabilities in the RAG ecosystem, from end-to-end solutions to specialized tools for optimization and enhancement. Depending on your specific needs, you can explore these options to find the most suitable RAG solution for your project.

  Citations:
  [1] https://github.com/EthicalML/awesome-production-machine-learning
  [2] https://github.com/Andrew-Jang/RAGHub
  [3] https://github.com/weaviate/Verba
  [4] https://github.com/Jenqyang/LLM-Powered-RAG-System
  [5] https://gist.github.com/gubatron/79793e1102726174013ffde798df4d1f
  [6] https://github.com/coree/awesome-rag
-
- Notes to assimilate
- Understanding Retrieval-Augmented Generation (RAG) with OpenAI | Codemancers https://www.codemancers.com/blog/2024-09-17-rag/? [[Retrieval Augmented Generation - RAG]]
- https://github.com/pathwaycom/pathway [[Retrieval Augmented Generation - RAG]]
- https://braindenburg.com/enterprise-ai-with-rag-crag-flare-eom/ [[Retrieval Augmented Generation - RAG]]
- [win4r/GraphRAG4OpenWebUI: GraphRAG4OpenWebUI integrates Microsoft's GraphRAG technology into Open WebUI, providing a versatile information retrieval API. It combines local, global, and web searches for advanced Q&A systems and search engines. This tool simplifies graph-based retrieval integration in open web environments. (github.com)](https://github.com/win4r/GraphRAG4OpenWebUI) [[Open Webui and Pipelines]] [[Knowledge Graphing]] [[Retrieval Augmented Generation - RAG]]
- [2410.05130v1.pdf (arxiv.org)](https://arxiv.org/pdf/2410.05130) [[Knowledge Graphing]] GraphAgent-Reasoner
- ![Mastering RAG.pdf](../assets/Mastering_RAG_1727962213794_0.pdf)
- https://www.reddit.com/r/LocalLLaMA/comments/1f61oxc/according_to_stanford_even_prograde_rag_systems/
- https://braindenburg.com/enterprise-ai-with-rag-crag-flare-eom/
- [AnswerDotAI/RAGatouille: Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research. (github.com)](https://github.com/AnswerDotAI/RAGatouille)


## Metadata

- **Last Updated**: 2025-11-16
- **Review Status**: Automated remediation with 2025 context
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable