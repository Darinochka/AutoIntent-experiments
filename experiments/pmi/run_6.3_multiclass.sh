#!/usr/bin/env bash
# Проверка 6.3 ПМИ — качество многоклассовой классификации намерений.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.3_multiclass.sh
#
# Ориентировочное время выполнения — от нескольких часов до суток.
#
# Прогон идёт по предъявленному дереву: pmi_init отказывается начинать работу на
# изменённом дереве, а pmi_assert_tree_untouched проверяет его после (п. 5.2.5 ПМИ).
# Побочные файлы uv (`.venv`, `uv.lock`) внесены в .gitignore образца.
# Аргументы: --check (только проверка предусловий), --help.

PMI_TREE_MODE=in-tree
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.3 "$@"

for name in banking77 hwu64 massive minds14 snips; do
    [[ -f "$DATASETS_DIR/$name.json" ]] \
        || pmi_die "не найдено зеркало набора $name; выполните prepare_datasets.py"
done
pmi_require_dir "$AUTOINTENT_DIR" "предъявленное дерево образца"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "CHECK OK: пять наборов данных и дерево образца на месте"
    exit 0
fi

cd "$AUTOINTENT_DIR"
uv run python "$PMI_DIR/lib/multiclass.py" \
    --datasets-dir "$DATASETS_DIR" \
    --summary "$PMI_RUN_DIR/summary.json" \
    2>&1 | tee "$PMI_RUN_DIR/multiclass.log"

# Конвейер выше выполняется под `set -euo pipefail` (common.sh): ненулевой код питоновской
# части не маскируется `tee`, а обрывает сценарий здесь же — до постпроверки дерева.
pmi_log "проверка неприкосновенности предъявленного дерева"
pmi_assert_tree_untouched
pmi_log "предъявленное дерево не изменено"
