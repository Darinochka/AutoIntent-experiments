#!/usr/bin/env bash
# Проверка 6.2 ПМИ — программный интерфейс обучения и предсказания и механизм
# сериализации обученного алгоритма.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.2_api.sh
#
# Прогон идёт по предъявленному дереву, поэтому оно должно быть чистым: pmi_init
# останавливает работу на изменённом дереве, а pmi_assert_tree_untouched проверяет
# его состояние после прогона (п. 5.2.5 ПМИ). Побочные файлы uv (`.venv`, `uv.lock`)
# внесены в .gitignore образца и дерево не пачкают.
#
# Аргументы: --check (только проверка предусловий), --help.

PMI_TREE_MODE=in-tree
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.2 "$@"

DATASET="$DATASETS_DIR/clinc150_subset.json"

[[ -f "$DATASET" ]] || pmi_die "не найдено зеркало набора данных $DATASET; выполните prepare_datasets.py"
pmi_require_dir "$AUTOINTENT_DIR" "предъявленное дерево образца"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "CHECK OK: набор данных и дерево образца на месте"
    exit 0
fi

cd "$AUTOINTENT_DIR"
uv run python "$PMI_DIR/lib/api_serialization.py" \
    --dataset "$DATASET" \
    --dump-dir "$PMI_RUN_DIR/pipeline" \
    2>&1 | tee "$PMI_RUN_DIR/api.log"

# Конвейер выше выполняется под `set -euo pipefail` (common.sh): ненулевой код питоновской
# части не маскируется `tee`, а обрывает сценарий здесь же — до постпроверки дерева.
# Постпроверка поэтому выполняется только на успешном прогоне и код возврата не подменяет.
pmi_log "проверка неприкосновенности предъявленного дерева"
pmi_assert_tree_untouched
pmi_log "предъявленное дерево не изменено"
