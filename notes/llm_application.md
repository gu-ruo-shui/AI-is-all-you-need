# LLM application dev
## LLM openai interface basic concepts
- text, audio, image, video
  - 推荐看李沐在上交的演讲视频
- general params
    - https://www.bilibili.com/video/BV1nK421Y72d
    - temperature
    - top k
    - top p
    - logprobs
    - top_logprobs
- context size
    - what is token
- core concepts
  - text and prompting
  - images and vision
  - audio and speech
  - structured ouputs
  - function calling
  - conversation state
  - streaming
  - file inputs
  - reasoning
- image
- 模型之间差距很大, 一个好的 base model 能省很多事

## prompt engineering
- [普通人如何通过DeepSeek提升工作效率](https://www.bilibili.com/video/BV1UdAVeXECY)
- https://www.promptingguide.ai/
- https://llmnanban.akmmusai.pro/Introductory/Prompt-Elements/
- https://www.leeboonstra.dev/prompt-engineering/prompt_engineering_guide6/ 系列
- https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering?tabs=chat

## RAG & ReRank
## MCP (function call)
## agent & workflow
- https://arthurchiao.art/blog/build-effective-ai-agent-zh/#11-workflow-vs-agent

## deploy
- ollma
- vllm
- SGLang

## 一些思考

如果把 LLM 看成编程语言的话, 那么基于 MCP 的各种 Server 就可以认为是各种各样的库。
以后的编程范式说不定就变成了， 我们使用自然语言与 LLM 进行对话， 实现各种各样的需求
如果一件事情 需要 2天去做, 我们大概率不会去做, 但是现在只需要半个小时尝试, 说不定愿意尝试了
就好像 python vs C or js vs C

# reference
- https://ninehills.tech/articles/97.html
- https://github.com/openai/openai-cookbook
- https://platform.openai.com/docs/api-reference/chat/create
- https://platform.openai.com/tokenizer
- https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p
- https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model
- https://www.bilibili.com/video/BV1iuXkYHE2B
- 
