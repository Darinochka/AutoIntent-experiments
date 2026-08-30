#!/usr/bin/env bash
# Проверка 6.8 ПМИ — онлайн-проверка подсказки инструментов в цикле работы агента.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.8_online.sh
#
# Базовый прогон и прогон с подсказкой (--jsonl-repo — параметр run_exp.py; не путать
# с --repo у offline_eval.py, п. 6.7), затем сводный отчет по обоим прогонам.
#
# run_exp.py не пишет результаты локально: они уходят в Logfire (run_exp.py:
# _init_logfire, "if-token-present") и скачиваются оттуда командой report.py
# (--experiment ...; report.py table). Нужны три ключа: OPENROUTER_API_KEY — модель
# агента; LOGFIRE_TOKEN или файл каталога эксперимента .logfire/logfire_credentials.json
# — запись в Logfire (без него прогон молча завершится без данных); LOGFIRE_API_KEY —
# чтение отчета в report.py.
#
# Каталог эксперимента самостоятельный, сценарий сам переходит в него.
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

# run_exp.py отправляет прогон в Logfire вызовом logfire.configure(send_to_logfire=
# "if-token-present", ...) (run_exp.py:_init_logfire). Токен на запись читается либо из
# LOGFIRE_TOKEN, либо (при его отсутствии) из файла проектных учётных данных
# .logfire/logfire_credentials.json внутри $EXP_DIR — он создаётся один раз командами
# `logfire auth` и `logfire projects use`, запущенными в каталоге эксперимента
# (logfire/_internal/config_params.py: TOKEN env_vars=[LOGFIRE_TOKEN];
# logfire/_internal/config.py: LogfireCredentials.load_creds_file). При "if-token-present"
# отсутствие токена не прерывает прогон: агент отработает и сценарий завершится кодом 0,
# просто ничего не будет отправлено — отказ обнаружится только на шаге report.py, когда
# .jsonl-файла не окажется. Поэтому здесь эта же проверка выполняется заранее.
LOGFIRE_CREDS_FILE="$EXP_DIR/.logfire/logfire_credentials.json"
if [[ -z "${LOGFIRE_TOKEN:-}" && ! -f "$LOGFIRE_CREDS_FILE" ]]; then
    pmi_die "не настроена запись в Logfire: нет ни переменной LOGFIRE_TOKEN, ни файла
$LOGFIRE_CREDS_FILE (создаётся командами 'logfire auth' и 'logfire projects use',
выполненными в каталоге эксперимента $EXP_DIR). Без этого run_exp.py (if-token-present)
завершится с кодом 0, ничего не отправив в Logfire, и отказ обнаружится только на шаге
report.py — там, где .jsonl файла для прогона попросту не окажется."
fi

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    cd "$EXP_DIR"
    uv run run_exp.py --help > /dev/null
    uv run report.py --help > /dev/null
    pmi_log "CHECK OK: корпус, окружение и запись в Logfire на месте; модель агента: $MODEL"
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
