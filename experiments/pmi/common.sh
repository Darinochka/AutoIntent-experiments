# Общая часть сценариев проверок ПМИ. Подключается через `source`.
#
# Предоставляет:
#   pmi_init <номер-пункта>  — разбор аргументов, подготовка каталога артефактов
#   pmi_log <сообщение>      — сообщение о ходе выполнения
#   pmi_die <сообщение>      — сообщение об ошибке и выход с кодом 1
#   PMI_CHECK_ONLY           — непусто, если задан --check
#   PMI_RUN_DIR              — каталог артефактов текущего пункта
#   pmi_require_git_tree <каталог> <пояснение>
#                            — каталог существует, является git-репозиторием и содержит
#                              тег предъявленной ревизии
#   pmi_require_pristine_tree
#                            — предъявленное дерево не изменено; проверка ДО прогона
#   pmi_assert_tree_untouched [каталог]
#                            — то же после прогона («проверка недействительна»)
#
# Соглашение о неприкосновенности предъявленного дерева (п. 5.2.5 ПМИ).
# Сценарий, работающий с $AUTOINTENT_DIR, объявляет режим переменной PMI_TREE_MODE,
# выставляемой ДО вызова pmi_init:
#   in-tree  — прогон идёт по предъявленному дереву (пп. 6.2, 6.3, 6.4, 6.5, 6.9).
#              pmi_init проверяет чистоту дерева ДО прогона, а сценарий завершает работу
#              вызовом pmi_assert_tree_untouched. Побочные файлы uv и прогонов
#              (`.venv`, `uv.lock`, `runs/`, `tests/logs`, `.coverage`, `htmlcov/`)
#              внесены в .gitignore образца и дерево не пачкают. Флаги --frozen/UV_FROZEN
#              НЕ применяются: uv.lock в образце не версионируется, в комплекте
#              предъявления его нет, и `uv run --frozen` на таком дереве падает
#              («Unable to find lockfile at uv.lock, but --frozen was provided»).
#   copy     — прогон идёт в копии дерева (пп. 6.10, 6.11). pmi_init проверяет чистоту
#              предъявленного дерева перед клонированием: иначе `git clone` унесёт в копию
#              чужое состояние. В п. 6.10 копия пересоздаётся при каждом запуске; в п. 6.11 —
#              на `--check`, на фазах `prepare` и `all` и когда копии нет, а при запуске
#              отдельной фазы берётся готовой (её создало действие 1 таблицы 12 ПМИ), иначе
#              каждая фаза заново разворачивала бы окружение образца. Принадлежность копии
#              закреплённому коммиту проверяется при любом запуске.
#   не задано — сценарий предъявленного дерева не касается (пп. 6.6, 6.7, 6.8).
# Чистота проверяется ДО прогона: испытание, начатое на изменённом дереве, недействительно,
# и узнавать об этом постфактум бессмысленно.
#
# Переменные окружения:
#   RUNS_DIR       корень каталога артефактов (по умолчанию <корень испытаний>/runs)
#   DATASETS_DIR   каталог офлайн-зеркала наборов данных (по умолчанию <корень>/datasets)

set -euo pipefail

PMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# experiments/pmi -> experiments -> AutoIntent-experiments -> корень испытаний
PMI_ROOT="$(cd "$PMI_DIR/../../.." && pwd)"

RUNS_DIR="${RUNS_DIR:-$PMI_ROOT/runs}"
DATASETS_DIR="${DATASETS_DIR:-$PMI_ROOT/datasets}"
AUTOINTENT_DIR="${AUTOINTENT_DIR:-$PMI_ROOT/AutoIntent}"
PMI_SAMPLE_TAG="${PMI_SAMPLE_TAG:-v0.4.0}"

PMI_CHECK_ONLY=""
PMI_RUN_DIR=""

pmi_log() { printf '\n== %s\n' "$*"; }
pmi_die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

pmi_init() {
    local item="$1"; shift || true
    for arg in "$@"; do
        case "$arg" in
            --check) PMI_CHECK_ONLY=1 ;;
            -h|--help) sed -n '1,20p' "${BASH_SOURCE[1]}"; exit 0 ;;
            *) pmi_die "неизвестный аргумент '$arg' (допустимы: --check, --help)" ;;
        esac
    done
    PMI_RUN_DIR="$RUNS_DIR/$item"
    mkdir -p "$PMI_RUN_DIR"
    [[ -w "$PMI_RUN_DIR" ]] || pmi_die "каталог $PMI_RUN_DIR недоступен для записи"
    case "${PMI_TREE_MODE:-none}" in
        in-tree)
            # прогон идёт по самому дереву: чистота проверяется до начала работы,
            # а сценарий обязан завершиться вызовом pmi_assert_tree_untouched
            pmi_require_pristine_tree
            ;;
        copy)
            # прогон идёт в копии: дерево-источник должно быть чистым до клонирования
            pmi_require_pristine_tree
            ;;
        none) ;;
        *) pmi_die "недопустимое значение PMI_TREE_MODE='${PMI_TREE_MODE:-}' (in-tree, copy или не задано)" ;;
    esac
    pmi_log "пункт $item; артефакты: $PMI_RUN_DIR; режим дерева: ${PMI_TREE_MODE:-none}"
}

pmi_require_dir() {
    [[ -d "$1" ]] || pmi_die "не найден каталог $1 ($2)"
}

pmi_require_env() {
    [[ -n "${!1:-}" ]] || pmi_die "не задана переменная окружения $1 ($2)"
}

# Каталог существует, является git-репозиторием и содержит тег предъявленной ревизии.
# Пояснение $2 — контекст вызывающего (какой пункт ПМИ и зачем); в сообщение об ошибке
# подставляется как есть, специфику пункта сюда не зашивать.
pmi_require_git_tree() {
    pmi_require_dir "$1" "$2"
    git -C "$1" rev-parse --git-dir > /dev/null 2>&1 \
        || pmi_die "каталог $1 не является git-репозиторием ($2); нужен клон репозитория образца с историей и тегами, а не распакованный архив"
    git -C "$1" rev-parse --verify --quiet "refs/tags/$PMI_SAMPLE_TAG" > /dev/null \
        || pmi_die "в $1 нет тега $PMI_SAMPLE_TAG ($2); выполните git fetch --tags в предъявленном дереве"
}

# Предъявленное дерево не изменено. Вызывается ДО прогона.
pmi_require_pristine_tree() {
    pmi_require_git_tree "$AUTOINTENT_DIR" "предъявленное дерево образца, проверка чистоты до начала прогона, п. 5.2.5 ПМИ"
    local dirty
    dirty="$(git -C "$AUTOINTENT_DIR" status --porcelain)"
    [[ -z "$dirty" ]] || pmi_die "предъявленное дерево $AUTOINTENT_DIR изменено до начала прогона (п. 5.2.5 ПМИ), прогон не начат.
Восстановите дерево и повторите:
    git -C \"$AUTOINTENT_DIR\" checkout -- . && git -C \"$AUTOINTENT_DIR\" clean -fd
Изменения:
$dirty"
}

# То же после прогона: расхождение означает, что проверка недействительна.
# Сначала убеждаемся, что каталог вообще существует и остаётся git-репозиторием —
# иначе `git status --porcelain` может упасть до `set -e` (command substitution внутри
# условия), пустой stdout читается как "дерево чистое", и исчезновение дерева
# молча сойдёт за "дерево не тронуто".
pmi_assert_tree_untouched() {
    local tree="${1:-$AUTOINTENT_DIR}"
    pmi_require_git_tree "$tree" "предъявленное дерево образца, проверка неприкосновенности после прогона, п. 5.2.5 ПМИ"
    [[ -z "$(git -C "$tree" status --porcelain)" ]] \
        || pmi_die "предъявленное дерево $tree изменено в ходе прогона — проверка недействительна (п. 5.2.5 ПМИ)"
}
