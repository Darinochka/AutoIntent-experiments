#!/usr/bin/env bash
# Проверка 6.10 ПМИ — локальное воспроизведение матрицы конфигураций.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.10_matrix.sh
#
# Работа ведется в копии AutoIntent-matrix/: цикл удаляет файл uv.lock, создает
# .python-version и пересобирает .venv, поэтому предъявленное дерево не трогается.
# Основным свидетельством по матрице служит прогон системы непрерывной интеграции,
# указанный в п. 6.10 ПМИ; этот скрипт выполняет локальное воспроизведение.
# Аргументы: --check (только проверка предусловий), --help.

# Режим copy: pmi_init убеждается, что предъявленное дерево — git-репозиторий с тегом
# $PMI_SAMPLE_TAG и без изменений (иначе `git clone` и `checkout` ниже молча унесут в
# копию чужое состояние). Дальше вся работа идёт в копии.

PMI_TREE_MODE=copy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.10 "$@"

MATRIX_DIR="${MATRIX_DIR:-$PMI_ROOT/AutoIntent-matrix}"
PYTHONS=(3.10 3.11 3.12 3.13 3.14)

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    for py in "${PYTHONS[@]}"; do
        uv python find "$py" > /dev/null 2>&1 \
            || pmi_log "предупреждение: Python $py не установлен, uv загрузит его при прогоне"
    done
    pmi_log "CHECK OK: дерево образца на месте, копия будет создана в $MATRIX_DIR"
    exit 0
fi

pmi_log "создание рабочей копии дерева"
# MATRIX_DIR переопределим переменной окружения, и эта строка — шаблон для
# копируемых pmi-сценариев, поэтому перед rm -rf проверяем, что путь безопасен:
# непустой, абсолютный, и не совпадает с $AUTOINTENT_DIR/$PMI_ROOT и не является
# их родительским каталогом (иначе rm -rf снёс бы предъявленное дерево или весь
# рабочий каталог испытаний).
[[ -n "$MATRIX_DIR" ]] || pmi_die "MATRIX_DIR пуст — rm -rf отменён"
[[ "$MATRIX_DIR" = /* ]] || pmi_die "MATRIX_DIR='$MATRIX_DIR' не является абсолютным путём — rm -rf отменён"
for guarded_dir in "$AUTOINTENT_DIR" "$PMI_ROOT"; do
    guarded_dir="${guarded_dir%/}"
    target_dir="${MATRIX_DIR%/}"
    if [[ "$target_dir" == "$guarded_dir" || "$guarded_dir" == "$target_dir"/* ]]; then
        pmi_die "MATRIX_DIR='$MATRIX_DIR' совпадает с '$guarded_dir' или является его родительским каталогом — rm -rf отменён"
    fi
done
rm -rf "$MATRIX_DIR"
git clone --quiet --no-local "$AUTOINTENT_DIR" "$MATRIX_DIR"
git -C "$MATRIX_DIR" checkout --quiet "$PMI_SAMPLE_TAG"

cd "$MATRIX_DIR"
for PY in "${PYTHONS[@]}"; do
    echo "=== Python $PY ==="
    rm -rf .venv uv.lock
    uv python pin "$PY"
    uv sync --no-cache --extra sentence-transformers
    uv run python -c "import autointent, sys; from importlib.metadata import version; print(sys.version.split()[0], version('autointent'))"
done 2>&1 | tee "$PMI_RUN_DIR/matrix.log"

pmi_log "сборка дистрибутивов"
uv build 2>&1 | tee "$PMI_RUN_DIR/build.log"

pmi_log "зафиксированный манифест зависимостей"
uv export --no-hashes > "$PMI_RUN_DIR/requirements.txt"

pmi_log "проверка неприкосновенности предъявленного дерева"
pmi_assert_tree_untouched
pmi_log "предъявленное дерево не изменено"
