#!/usr/bin/env bash
# Проверка 6.9 ПМИ — контрольный локальный запуск набора тестов с измерением покрытия.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.9_tests.sh
#
# ВНИМАНИЕ. Результат этого запуска в критерий приемки не входит. Совокупное
# покрытие проверяется по сводному отчету прогона системы непрерывной интеграции
# (см. п. 6.9 ПМИ): в pyproject.toml параметр addopts содержит -m "not transformers",
# поэтому локально измеряется строго меньшая величина.
# Аргументы: --check (только проверка предусловий), --help.

# Прогон идёт по предъявленному дереву. Оно должно быть чистым: pmi_init проверяет это
# ДО запуска, pmi_assert_tree_untouched — после (п. 5.2.5 ПМИ). Развертывание меняет
# только .venv/ и uv.lock, а измерение покрытия — .coverage; все они внесены
# в .gitignore образца, поэтому дерево остаётся чистым.
# Флаги --frozen/UV_FROZEN не применяются: uv.lock в образце не версионируется и в
# комплекте предъявления отсутствует, а с --frozen uv завершается ошибкой поиска
# lock-файла. Разрешение зависимостей выполняется штатно на месте.

PMI_TREE_MODE=in-tree
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.9 "$@"

# наличие дерева, его тег и чистоту уже проверил pmi_init (PMI_TREE_MODE=in-tree)
[[ -d "$AUTOINTENT_DIR/tests" ]] || pmi_die "не найден каталог тестов $AUTOINTENT_DIR/tests"

cd "$AUTOINTENT_DIR"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    uv run --group test pytest tests --collect-only -q 2>&1 | tail -5 | tee "$PMI_RUN_DIR/collect.log"
    pmi_log "CHECK OK: дерево образца чистое, набор тестов собирается"
    exit 0
fi

pmi_log "развертывание окружения с группой тестовых зависимостей"
uv sync --group test --extra catboost --extra peft \
    --extra transformers --extra sentence-transformers --extra openai \
    2>&1 | tee "$PMI_RUN_DIR/sync.log"

pmi_log "запуск набора тестов с измерением покрытия"
uv run --group test pytest tests --cov 2>&1 | tee "$PMI_RUN_DIR/pytest.log"

pmi_log "формирование отчетов о покрытии"
uv run --group test coverage report > "$PMI_RUN_DIR/coverage.txt"
uv run --group test coverage xml -o "$PMI_RUN_DIR/coverage.xml"
uv run --group test coverage html -d "$PMI_RUN_DIR/htmlcov"

pmi_assert_tree_untouched

pmi_log "локально измеренное покрытие ниже сводного по причинам, изложенным в п. 6.9 ПМИ"
