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

## Результаты экспериментов

- basic agent filesystem: https://logfire-eu.pydantic.dev/public-trace/484cff60-e789-4bee-9a5b-361dcf1a45b2?spanId=9d38217226e09941
- basic agent filesystem self-correction: https://logfire-eu.pydantic.dev/public-trace/5cd5e7b1-2fb1-4357-93c5-83df22d773c1?spanId=edeab683c1cfbbe5

Промежуточный итог: нужно добавить code execution, потому что некоторые задачи съедают очень много токенов так как требуют от ллм вручную делать дата процессинг (типа разделить большой файл на три части)
