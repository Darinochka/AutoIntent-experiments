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

эмбединги openai small

### без OOS detection

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

### с OOS detection

и `under_represented_behavior="always_include"` 

- haiku-4.5: https://logfire-eu.pydantic.dev/public-trace/78d1cfc9-9b98-418b-ae57-b777487fd8ea?spanId=48699272a41bf40f
- opus-4.6: https://logfire-eu.pydantic.dev/public-trace/14256429-7e85-470f-9b1d-6a499c5b098c?spanId=38e5b1c187acafef
- gpt-5.4: https://logfire-eu.pydantic.dev/public-trace/490e44a1-f403-4cde-b11f-1a86abd93820?spanId=e2769c41b50a0318
- gpt-5.4-mini: https://logfire-eu.pydantic.dev/public-trace/e1a42fb8-9e38-4b63-a922-ffe73453f6dd?spanId=6c7375340663a148
- qwen3-coder-plus: https://logfire-eu.pydantic.dev/public-trace/c12901fc-566b-4b52-b3bf-281e3e37f26a?spanId=b8bf51d4e734f8b8
- deepseek-v3.2: https://logfire-eu.pydantic.dev/public-trace/b0de1c01-7570-4ad9-be11-939ad7ae696e?spanId=d7d321051d2664d0


#### cross-validation

- haiku-4.5:
    - https://logfire-eu.pydantic.dev/public-trace/a40ea1e0-a11f-4e25-ab99-fc8866d70e46?spanId=af8536a54892c45b
    - https://logfire-eu.pydantic.dev/public-trace/499bdc10-a7ec-4b0d-831d-817ec595a51b?spanId=844a931495fd6ec7
    - https://logfire-eu.pydantic.dev/public-trace/90378452-cbe7-41b1-911d-1622ec023b4d?spanId=516c4b3384b62286
    - https://logfire-eu.pydantic.dev/public-trace/d8134193-ee25-405a-90f9-4539f34a0044?spanId=9709bca9354a7097
    - https://logfire-eu.pydantic.dev/public-trace/23b2a39a-fc95-41b5-9fe1-3786193abe6c?spanId=356c8caab8ef368e
- opus-4.6:
    - https://logfire-eu.pydantic.dev/public-trace/947b3160-0075-41a6-8022-bb142f780da7?spanId=d43e619e2351a715
    - https://logfire-eu.pydantic.dev/public-trace/5121a1c6-6c58-4ee3-8151-eedf399ee32d?spanId=f48245b230e40317
    - https://logfire-eu.pydantic.dev/public-trace/ffb9ec89-b39f-4d86-bd76-54359b870815?spanId=5f77602f762beb88
    - https://logfire-eu.pydantic.dev/public-trace/92292e1e-5d09-40e4-a64f-da74f89e87bd?spanId=4a2e7e6aaff73195
    - https://logfire-eu.pydantic.dev/public-trace/0336e1ec-cc81-42f7-ae9d-cc5f96c1a204?spanId=c5bb4d5d5330587f
- gpt-5.4:
    - https://logfire-eu.pydantic.dev/public-trace/476f9505-0956-43d1-a5e0-3872694ab88a?spanId=9104dcf21ca0852b
    - https://logfire-eu.pydantic.dev/public-trace/2c9a4990-956d-40c7-a602-43b8a2f87ceb?spanId=963c1b188cf0f601
    - https://logfire-eu.pydantic.dev/public-trace/dd0cabb1-801d-41d3-9a59-032a699102b2?spanId=5fbb812772da006c
    - https://logfire-eu.pydantic.dev/public-trace/7df0543e-745e-4975-b50e-fc22c8b53bad?spanId=798e4f34c923901e
    - https://logfire-eu.pydantic.dev/public-trace/4fdf9b34-6452-466e-afca-07799489e862?spanId=7d60455f96ff1f4a
- gpt-5.4-mini:
    - https://logfire-eu.pydantic.dev/public-trace/f96a957b-5db1-4972-9397-018053aa1857?spanId=9867a54a3fcd4f2a
    - https://logfire-eu.pydantic.dev/public-trace/45335ca7-cb4a-485f-9684-2afedcbeefce?spanId=73deee9365c25f6f
    - https://logfire-eu.pydantic.dev/public-trace/9f8ae371-b171-4142-9796-d4ba2188fc15?spanId=df097ffe8202c923
    - https://logfire-eu.pydantic.dev/public-trace/5043d59a-eb2c-4111-bad3-ef4635ab0f01?spanId=f8f0da3bfb034bd3
    - https://logfire-eu.pydantic.dev/public-trace/cfa2ebb4-713a-4902-b348-1b7dd8d2d1bd?spanId=5fba0445fbc07824
- gpt-5.4-nano:
    - https://logfire-eu.pydantic.dev/public-trace/06ab1d36-fc75-4d06-9908-f4656ff5c64c?spanId=1ca6d9e816cdd3cf
    - https://logfire-eu.pydantic.dev/public-trace/31c96cec-e9c2-495d-978b-ac2b5eb5c45e?spanId=df72fbed93161f22
    - https://logfire-eu.pydantic.dev/public-trace/c530eb72-2ace-4a78-b80d-350f0605d5e8?spanId=332d29a7165246ef
    - https://logfire-eu.pydantic.dev/public-trace/1602d850-0dd4-4a6a-b3dc-e24243ca2c67?spanId=51c53a63b8b21333
    - https://logfire-eu.pydantic.dev/public-trace/0ad2dbd6-2acb-462d-8e88-204f8d4df41e?spanId=f3e7fec0c5c6ea60
- qwen3-coder-plus:
    - https://logfire-eu.pydantic.dev/public-trace/db27f960-6075-44cb-a88e-2e9e3f0f3041?spanId=4e31ab5347bba8e7
    - https://logfire-eu.pydantic.dev/public-trace/1ed1cde4-9a4d-4d62-860d-cc8a0930ebae?spanId=3859a151d6aeeb03
    - https://logfire-eu.pydantic.dev/public-trace/9ff1129f-3a99-4026-b834-02067fcf2776?spanId=0c6efc7c159ed54c
    - https://logfire-eu.pydantic.dev/public-trace/d0ce7cfd-15ad-4b70-a5ad-e2cfcab128df?spanId=7c774a1cab692ab5
    - https://logfire-eu.pydantic.dev/public-trace/ce805346-1f1a-41b6-a35b-8b3c06838acd?spanId=790b0905f01fd1a0
- deepseek-v3.2:
    - https://logfire-eu.pydantic.dev/public-trace/ba7d18b9-3574-4900-b870-4a2755167651?spanId=22005625f27fb358
    - https://logfire-eu.pydantic.dev/public-trace/61eda002-a80f-49c9-b149-8197813c26ba?spanId=bc71e6478302978a
    - https://logfire-eu.pydantic.dev/public-trace/9b7668e2-5e3b-4e8a-8941-008a38cf2638?spanId=beb2578330b4f46b
    - https://logfire-eu.pydantic.dev/public-trace/551044ed-67bf-41ce-99fc-dbfb713de103?spanId=2b98d3e96a14f1ab
    - https://logfire-eu.pydantic.dev/public-trace/c5a5f727-a8a4-45a6-a3cd-7fb30990f20a?spanId=b0c8914a54606ac8

ну кринж полный

Here is a concise comparison using your current `reports/` JSONL files. `**report.py compare-readme**` (see `src/report/compare_readme.py`) prints the tables below; regenerate anytime:

```bash
cd experiments/mcp-exps && uv run report.py compare-readme
# optional: uv run report.py compare-readme --reports-dir /path/to/reports
```

**Pairing:** each row is **basic-fs** (one trace, 25 tasks) vs **tool-suggest OOS CV** (merged 5-fold aggregate): `cv-readme-*.jsonl`. **GPT-5.4 mini** uses `**cv-gpt54-mini-aggregated.jsonl`** (same five public links as in the mini CV list above).

### Pass rates


| Model         | Hard basic | Hard CV | Soft basic | Soft CV |
| ------------- | ---------- | ------- | ---------- | ------- |
| Haiku 4.5     | 32%        | 8%      | 80.8%      | 19.1%   |
| Opus 4.6      | 72%        | 32%     | 91.0%      | 49.5%   |
| GPT-5.4       | 40%        | 20%     | 76.5%      | 32.6%   |
| GPT-5.4 mini  | 16%        | 12%     | 56.7%      | 48.4%   |
| GPT-5.4 nano  | 8%         | 8%      | 58.0%      | 27.3%   |
| Qwen3 Coder+  | 16%        | 16%     | 64.7%      | 45.4%   |
| DeepSeek V3.2 | 24%        | 4%      | 59.8%      | 22.2%   |


- **Hard** = `passed_tasks / total_tasks` from each JSONL header (all evaluators 1.0 on a task).  
- **Soft** = fraction of **individual evaluator** scores that equal 1.0 across all case rows.  
- Both sides use **N = 25** case rows (full domain × CV disambiguation).

### Usage (per-case mean over case rows; comparable basic vs CV)

Averaging **per-task** `input_tokens` / `output_tokens` / `requests` / `cost` from the JSONL case lines (not merged **header** sums: CV headers add all five traces, which is misleading next to a single-trace basic run).


| Model         | in tok basic | in tok CV | out tok basic | out tok CV | req basic | req CV | cost basic | cost CV |
| ------------- | ------------ | --------- | ------------- | ---------- | --------- | ------ | ---------- | ------- |
| Haiku 4.5     | 422k         | 394k      | 4.6k          | 5.4k       | 20.68     | 16.80  | 0.0000     | 0.0000  |
| Opus 4.6      | 343k         | 192k      | 6.4k          | 4.0k       | 14.52     | 9.92   | 0.0000     | 0.0000  |
| GPT-5.4       | 135k         | 127k      | 1.3k          | 1.0k       | 7.88      | 7.92   | 0.0000     | 0.3077  |
| GPT-5.4 mini  | 63k          | 57k       | 1.0k          | 0.6k       | 8.32      | 5.92   | 0.0000     | 0.0294  |
| GPT-5.4 nano  | 63k          | 79k       | 0.7k          | 0.6k       | 9.88      | 9.40   | 0.0000     | 0.0122  |
| Qwen3 Coder+  | 151k         | 307k      | 1.2k          | 1.2k       | 14.24     | 19.92  | 0.0000     | 0.0000  |
| DeepSeek V3.2 | 285k         | 357k      | 2.2k          | 1.1k       | 16.12     | 13.12  | 0.0000     | 0.0000  |


For raw **header** totals (e.g. summed CV trace usage), use `uv run report.py table --report-path reports/<name>.jsonl`. Some **basic** runs still show **cost 0** in rollups; GPT-5.4 / mini / nano show non-zero cost in CV where Logfire captured it.

### offline metrics

на примерах опуса

#### knn (for debug)


|       | top1   | topk   | mrr    |
| ----- | ------ | ------ | ------ |
| micro | 0.6249 | 0.8812 | 0.7452 |
| macro | 0.6071 | 0.8710 | 0.7233 |


```bash
uv run offline_eval.py --repo exported_repos/basic-fs-opus-4-6_true_test_0.jsonl \
  --split cv \
  --cv-folds 5 \
  --suggester knn \
  --emb-backend openai \
  --emb-model text-embedding-3-small \
  --formatter-max-len 4096 \
  --knn-neighbors 5 \
  --knn-aggregation weighted \
  --topk-metric 5 \
  --task-key case_name
```

#### autointent


|       | top1   | topk   | mrr    |
| ----- | ------ | ------ | ------ |
| micro | 0.7986 | 0.9338 | 0.8590 |
| macro | 0.8106 | 0.9501 | 0.8758 |


```bash
uv run offline_eval.py --repo exported_repos/basic-fs-opus-4-6_true_test_0.jsonl \
  --split cv --cv-folds 5 --random-state 42 \
  --suggester autointent \
  --emb-backend openai --emb-model text-embedding-3-small \
  --formatter-max-len 4096 \
  --selection-target-size 90 --min-samples-per-tool 4 --max-oos 0.2 \
  --no-multilabel \
  --experiment-name offline-fs-opus-autointent \
  --topk-metric 5 --task-key case_name
```

## REDO

заново бейзлайн на опусе: https://logfire-eu.pydantic.dev/public-trace/65b87987-89b6-451f-9195-a592854fbf2f?spanId=4ec2c13a9ad85099

## cv autointent oos

- gpt-5.4: https://logfire-eu.pydantic.dev/public-trace/750b65b7-62d8-4edc-9d52-a16313dcf723?spanId=578aba01ead3aa70
- gpt-5.4-mini: https://logfire-eu.pydantic.dev/public-trace/21f8b151-2c45-4e20-823a-6ef688a03b10?spanId=e6b055ffa95320e1
- gpt-5.4-nano: https://logfire-eu.pydantic.dev/public-trace/ee9722e6-7d95-41e6-b5cf-4bd5d2e9e6e5?spanId=79751e9da3fba556
- qwen3-coder-plus: https://logfire-eu.pydantic.dev/public-trace/b7895224-63f9-4d5a-af37-ff53af2c9c0b?spanId=f3b0dbe311f691c8
- deepseek-v3.2: 

### с аккумуляцией


