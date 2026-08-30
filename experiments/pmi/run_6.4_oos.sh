#!/usr/bin/env bash
# Проверка 6.4 ПМИ — детекция внедоменных запросов (OOS).
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.4_oos.sh
#
# Ориентировочное время выполнения — несколько часов.
#
# Прогон идёт по предъявленному дереву: pmi_init отказывается начинать работу на
# изменённом дереве, а pmi_assert_tree_untouched проверяет его после (п. 5.2.5 ПМИ).
# Побочные файлы uv (`.venv`, `uv.lock`) внесены в .gitignore образца.
# Аргументы: --check (только проверка предусловий), --help.

PMI_TREE_MODE=in-tree
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.4 "$@"

[[ -f "$DATASETS_DIR/clinc150.json" ]] \
    || pmi_die "не найдено зеркало набора clinc150; выполните prepare_datasets.py"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "CHECK OK: набор clinc150 и дерево образца на месте"
    exit 0
fi

cd "$AUTOINTENT_DIR"
uv run python "$PMI_DIR/lib/oos.py" \
    --datasets-dir "$DATASETS_DIR" \
    --summary "$PMI_RUN_DIR/summary.json" \
    2>&1 | tee "$PMI_RUN_DIR/oos.log"

# Конвейер выше выполняется под `set -euo pipefail` (common.sh): ненулевой код питоновской
# части не маскируется `tee`, а обрывает сценарий здесь же — до постпроверки дерева.
pmi_log "проверка неприкосновенности предъявленного дерева"
pmi_assert_tree_untouched
pmi_log "предъявленное дерево не изменено"
