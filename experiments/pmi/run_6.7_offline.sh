#!/usr/bin/env bash
# Проверка 6.7 ПМИ — офлайн-проверка подсказки инструментов на корпусе траекторий.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.7_offline.sh
#
# Выполняются два прогона на одних и тех же фолдах: образец (--suggester autointent)
# и базовый метод kNN (--suggester knn), с общим начальным значением генератора
# --random-state. Параметр набора данных offline_eval.py называется --repo (путь
# к JSONL), а не --jsonl-repo (это имя принадлежит run_exp.py, п. 6.8).
# Векторные представления вычисляются через OpenAI API (--emb-backend openai);
# для обоих прогонов, включая kNN, требуется OPENAI_API_KEY.
#
# Каталог эксперимента самостоятельный (не входит в предъявленное дерево образца):
# сценарий сам переходит в него и не касается $AUTOINTENT_DIR.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.7 "$@"

EXP_DIR="$PMI_DIR/../mcp-exps"
REPO="${MCPMARK_REPO:-$DATASETS_DIR/mcpmark/basic-fs-opus-4-6_true_test_0.jsonl}"

pmi_require_dir "$EXP_DIR" "каталог экспериментов подсказки инструментов"
[[ -f "$REPO" ]] || pmi_die "не найден корпус траекторий $REPO"
pmi_require_env OPENAI_API_KEY "ключ доступа к OpenAI API для вычисления векторных представлений"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "корпус: $(wc -l < "$REPO") записей"
    cd "$EXP_DIR" && uv run offline_eval.py --help > /dev/null
    pmi_log "CHECK OK: корпус, окружение и ключ доступа на месте"
    exit 0
fi

cd "$EXP_DIR"

for suggester in autointent knn; do
    pmi_log "прогон: $suggester"
    uv run offline_eval.py \
        --repo "$REPO" \
        --suggester "$suggester" \
        --split cv \
        --cv-folds 5 \
        --random-state 42 \
        --emb-backend openai \
        --emb-model text-embedding-3-small \
        --formatter-max-len 4096 \
        --json-out "$PMI_RUN_DIR/$suggester.json" \
        2>&1 | tee "$PMI_RUN_DIR/offline_eval_$suggester.log"
done
