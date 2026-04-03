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
- claude haiku-4.5: https://logfire-eu.pydantic.dev/public-trace/7a5a604b-40a8-4349-8c42-fa7b20b2ca5f?spanId=9fa54ec12a1bc28c
- qwen3-coder-plus: https://logfire-eu.pydantic.dev/public-trace/af5902e4-9911-4f4b-8226-e8c82a8e1328?spanId=9b5c5e16d003484d
- deepseek v3.2: https://logfire-eu.pydantic.dev/public-trace/33e790ca-d3dc-4c38-9026-b0e79805b1c2?spanId=ab2cb586faf68885

## tool-suggest filesystem

### обучение на примерах опуса:

- эмбединги openai small
- автоинтент не использует OOS detection

```bash
uv run run_exp.py ts-repro \
    --domain fs \
    --experiment-name ts-fs-repro-haiku45-openaismall \
    --tool-retries 5 \
    --model "openrouter:anthropic/claude-haiku-4.5" \
    --max-concurrency 5 \
    --jsonl-repo exported_repos/basic-fs-opus-4-6_true_test_0.jsonl \
    --grouper ho \
    --top-k 5 \
    --formatter-max-len 4096 \
    --selection-target-size 150 \
    --tool-samples 4
```

- haiku-4.5: https://logfire-eu.pydantic.dev/public-trace/126db530-1dcd-452e-8221-8469979c1052?spanId=1df26766fdf66753
- opus-4.6: https://logfire-eu.pydantic.dev/public-trace/d0b29063-dbcf-45cd-bf97-2c4b3d59eb0b?spanId=7f01eefc5c88fd76
- gpt-5.4: https://logfire-eu.pydantic.dev/public-trace/b9cf2ec3-7472-4be2-b6f5-83ac56639204?spanId=aaba1e3fc3380a9b
- gpt-5.4-mini: https://logfire-eu.pydantic.dev/public-trace/e52cbb3d-3835-4b2d-a25d-845f853d1141?spanId=d501604fb19908ba
- gpt-5.4-nano: https://logfire-eu.pydantic.dev/public-trace/ef74786a-0e92-496b-be26-3ab9465d7195?spanId=7c5fe7a6c523d14c
- qwen3-coder-plus: https://logfire-eu.pydantic.dev/public-trace/d6cf1c72-95dc-4d29-980d-cbd93ba65ff1?spanId=fb70f46151e5130f
- deepseek-v3.2: https://logfire-eu.pydantic.dev/public-trace/afb0d39e-060b-4bb8-9ecb-a9453e21c030?spanId=78d5cfbeb310d7f2
