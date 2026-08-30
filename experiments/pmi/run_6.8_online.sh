#!/usr/bin/env bash
# Проверка 6.8 ПМИ — онлайн-проверка подсказки инструментов в цикле работы агента.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.8_online.sh
#
# Базовый прогон без подсказки и прогон с подсказкой в режиме воспроизведения
# ts-repro на одних и тех же заданиях (--jsonl-repo — параметр run_exp.py в
# режиме ts-repro; не путать с --repo у offline_eval.py, п. 6.7), затем сводный
# отчет по обоим прогонам.
#
# run_exp.py не пишет результаты локально: сведения о прогоне уходят через Logfire
# (pydantic-ai instrumentation, run_exp.py:_init_logfire), а report.py их оттуда
# скачивает по имени эксперимента (--experiment, дефолтная команда) и складывает
# в каталог results/ каталога эксперимента; печать сводки — отдельная команда
# `report.py table --report-path <файл>` (report.py, дефолтная команда `load` и
# команда `table`). Голый `uv run report.py` без аргументов не работает: у
# дефолтной команды обязательный параметр --experiment. Чтение из Logfire
# использует ключ доступа LOGFIRE_API_KEY (report.py: read_token=os.getenv
# ("LOGFIRE_API_KEY")).
#
# Каталог эксперимента самостоятельный (не входит в предъявленное дерево образца):
# сценарий сам переходит в него и не касается $AUTOINTENT_DIR.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.8 "$@"

EXP_DIR="$PMI_DIR/../mcp-exps"
REPO="${MCPMARK_REPO:-$DATASETS_DIR/mcpmark/basic-fs-opus-4-6_true_test_0.jsonl}"
MODEL="${PMI_AGENT_MODEL:-openrouter:anthropic/claude-haiku-4.5}"
RESULTS_DIR="results"
BASIC_NAME="pmi-basic-fs"
TS_NAME="pmi-ts-fs"

pmi_require_dir "$EXP_DIR" "каталог экспериментов подсказки инструментов"
[[ -f "$REPO" ]] || pmi_die "не найден корпус траекторий $REPO"
pmi_require_env OPENROUTER_API_KEY "ключ доступа к OpenRouter для модели агента $MODEL"
pmi_require_env LOGFIRE_API_KEY "ключ доступа для чтения результатов прогонов из Logfire (report.py)"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    cd "$EXP_DIR"
    uv run run_exp.py --help > /dev/null
    uv run report.py --help > /dev/null
    pmi_log "CHECK OK: корпус и окружение на месте; модель агента: $MODEL"
    exit 0
fi

cd "$EXP_DIR"

pmi_log "базовый прогон без подсказки инструментов"
uv run run_exp.py basic \
    --domain fs \
    --experiment-name "$BASIC_NAME" \
    --model "$MODEL" \
    --max-concurrency 5 \
    2>&1 | tee "$PMI_RUN_DIR/basic.log"

pmi_log "прогон с подсказкой инструментов, кросс-валидация по заданиям"
uv run run_exp.py ts-repro \
    --domain fs \
    --experiment-name "$TS_NAME" \
    --model "$MODEL" \
    --tool-retries 5 \
    --max-concurrency 5 \
    --jsonl-repo "$REPO" \
    --grouper cv \
    --cv-splits 5 \
    --top-k 5 \
    --formatter-max-len 4096 \
    --selection-target-size 150 \
    --tool-samples 4 \
    2>&1 | tee "$PMI_RUN_DIR/ts.log"

pmi_log "сводный отчет по обоим прогонам"
{
    uv run report.py --experiment "$BASIC_NAME" --output-dir "$RESULTS_DIR"
    uv run report.py --experiment "$TS_NAME" --output-dir "$RESULTS_DIR"
    uv run report.py table --report-path "$RESULTS_DIR/$BASIC_NAME.jsonl"
    uv run report.py table --report-path "$RESULTS_DIR/$TS_NAME.jsonl"
} 2>&1 | tee "$PMI_RUN_DIR/report.log"
