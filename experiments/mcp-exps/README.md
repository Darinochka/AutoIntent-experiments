- [x] эксперименты с postgres и filesystem доменами из mcpmark
- [] остальные домены из mcpmark и mcp-universe
- [] все описания тулов в начале, а саджесты краткие
- [] намеренно смешать MCP тулы с нескольких доменов
- [] эксперименты с кодексом и другими кодинговыми агентами
- [] запуск где трейн совпадает с тестом

## Logfire Note

Pydantic-ai tech stack includes awesome [Logfire](https://logfire.pydantic.dev/docs/) --- observability tool for inspecting LLM tool calls and responces. However, if you want to use it, you'd better use some non-Russian proxy, so your spans are sent without any problem. Like this:
```bash
ALL_PROXY=http://127.0.0.1:1087 uv run run_exp.py --domain fs --experiment-name basic-fs-smoke --agent basic --model gpt-4.1
```

## Результаты экспериментов

- basic agent filesystem: https://logfire-eu.pydantic.dev/public-trace/484cff60-e789-4bee-9a5b-361dcf1a45b2?spanId=9d38217226e09941
- basic agent filesystem self-correction: https://logfire-eu.pydantic.dev/public-trace/5cd5e7b1-2fb1-4357-93c5-83df22d773c1?spanId=edeab683c1cfbbe5

Промежуточный итог:
- ~~нужно добавить code execution, потому что некоторые задачи съедают очень много токенов так как требуют от ллм вручную делать дата процессинг (типа разделить большой файл на три части)~~
- убрать таски которые требует от ллм ручной дата процессинг:
    - file_splitting
    - dataset_comparison
    - все задачи с фикстурой LEGAL_DOCUMENT


## бейзлайны filesystem

- gpt-5.4: https://logfire-eu.pydantic.dev/public-trace/abfb4e7c-9bdf-4415-a32c-de34fabcd418?spanId=204b814d12a4b866
- gpt-5.4-mini: https://logfire-eu.pydantic.dev/public-trace/d60a75de-0735-4c5e-b6e7-c0db11f523ec?spanId=dbb83e8eaf3b753d
- gpt-5.4-nano: https://logfire-eu.pydantic.dev/public-trace/c4e1adad-7361-4df5-9aba-1f489bad1213?spanId=aaf57ec6849add25
- claude opus-4.6:
    - https://logfire-eu.pydantic.dev/public-trace/a158dd2a-c3b4-4c31-9649-10a37a3a1d54?spanId=97f442b8a47c7d73
    - https://logfire-eu.pydantic.dev/public-trace/831a49a3-ffc6-4148-a002-a0fb1e55a69b?spanId=397fc0042978e0f0
- claude haiku-4.5:
    - https://logfire-eu.pydantic.dev/public-trace/7a5a604b-40a8-4349-8c42-fa7b20b2ca5f?spanId=9fa54ec12a1bc28c