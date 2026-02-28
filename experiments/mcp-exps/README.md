- [x] эксперименты с postgres и filesystem доменами из mcpmark
- [] остальные домены из mcpmark и mcp-universe
- [] все описания тулов в начале, а саджесты краткие
- [] намеренно смешать MCP тулы с нескольких доменов
- [] эксперименты с кодексом и другими кодинговыми агентами

## Logfire Note

Pydantic-ai tech stack includes awesome [Logfire](https://logfire.pydantic.dev/docs/) --- observability tool for inspecting LLM tool calls and responces. However, if you want to use it, you'd better use some non-Russian proxy, so your spans are sent without any problem. Like this:
```bash
ALL_PROXY=http://127.0.0.1:1087 uv run run_exp.py --domain fs --experiment-name basic-fs-smoke --agent basic --model gpt-4.1
```
