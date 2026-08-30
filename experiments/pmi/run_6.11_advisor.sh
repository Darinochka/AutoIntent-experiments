#!/usr/bin/env bash
# Проверка 6.11 ПМИ — метод адаптации моделей к вычислительным ресурсам.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.11_advisor.sh --check
#     bash AutoIntent-experiments/experiments/pmi/run_6.11_advisor.sh --phase 1
#
# Аргументы:
#   --phase <1|2|3|tables|verdicts|prepare|all>   что выполнить (по умолчанию all)
#   --check    только предусловия: пересоздать копию дерева и выполнить
#              ./reproduce.sh --check (окружение и вычислительная установка)
#   --help     эта справка
#
# Фазы 1, 2, 3 и tables — фазы сценария воспроизведения
# advisor-calibration-laptop6gb/reproduce.sh; они идут в копии дерева
# AutoIntent-advisor/ на закреплённом коммите b38f3c3a. Фаза verdicts снимает
# отдельные вердикты выполнимости с ПРЕДЪЯВЛЕННОЙ ревизии (действия 2—4
# таблицы 12 ПМИ). Фаза prepare только пересоздаёт копию дерева.
# Журналы — в $RUNS_DIR/6.11/; JSON фаз сценарий воспроизведения складывает
# в advisor-calibration-laptop6gb/results/ (действия 5, 7, 11 таблицы 12).
#
# --- подробности (в справку не попадают) ------------------------------------
#
# Фазы запускаются по одной, потому что таблица 12 ПМИ предписывает между ними
# ручные действия (выписать вердикты фазы 1, убедиться в изоляции процессов
# фазы 2), а фаза 2 идёт от одного до двух часов.
#
# Сценарию воспроизведения передаются две переменные окружения:
#     AUTOINTENT_DIR    — <корень испытаний>/AutoIntent-advisor, дерево, в котором
#                         он работает. Без неё он взял бы значение по умолчанию
#                         (../../../AutoIntent — предъявленное дерево) и выполнил
#                         бы в нём `git checkout --detach`, нарушив п. 5.2.5 ПМИ;
#     AUTOINTENT_COMMIT — закреплённый коммит b38f3c3a (то же значение сценарий
#                         воспроизведения принимает по умолчанию; передаётся явно,
#                         полным SHA, чтобы закрепление было видно в вызове).
# SKIP_SETUP намеренно НЕ задаётся: он отключает не только чекаут, но и `uv sync`,
# разворачивающий окружение копии, без которого последующие `uv run --no-sync`
# окружения не найдут.
#
# Развёртывание окружения копии (индекс пакетов) и фаза verdicts (набор
# DeepPavlov/banking77 из Hugging Face Hub) требуют доступа к сети либо к
# внутренним зеркалам — п. 5.2.3 ПМИ это фиксирует. Git-операции к сети не
# обращаются: и клон, и точечная выборка коммита идут из предъявленного дерева,
# после чего собственный fetch_commit сценария воспроизведения находит коммит
# уже присутствующим (`git cat-file -e`) и в сеть не выходит.
#
# Разбор --phase идёт ДО pmi_init: pmi_init принимает только --check и --help
# и завершает работу на любом другом аргументе.

PMI_PHASE=all
PMI_ARGS=()
while (( $# )); do
    case "$1" in
        # значение обязательно проверять до `shift 2`: при $# = 1 shift завершается
        # неуспешно и НЕ сдвигает параметры, а set -e ещё не активен (common.sh
        # подключается ниже), поэтому цикл `while (( $# ))` стал бы бесконечным
        --phase)
            [[ $# -ge 2 ]] || { printf 'ОШИБКА: --phase требует значения\n' >&2; exit 1; }
            PMI_PHASE="$2"; shift 2 ;;
        --phase=*) PMI_PHASE="${1#*=}"; shift ;;
        *) PMI_ARGS+=("$1"); shift ;;
    esac
done
set -- ${PMI_ARGS[@]+"${PMI_ARGS[@]}"}

# Режим copy: pmi_init убеждается, что предъявленное дерево — git-репозиторий с тегом
# $PMI_SAMPLE_TAG и без изменений (иначе `git clone` унесёт в копию чужое состояние).
# Дальше фазы сценария воспроизведения идут в копии; предъявленное дерево читается
# только фазой verdicts.

PMI_TREE_MODE=copy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.11 "$@"

case "$PMI_PHASE" in
    all) PMI_PHASES=(1 2 3 tables); PMI_VERDICTS=yes ;;
    1|2|3|tables) PMI_PHASES=("$PMI_PHASE"); PMI_VERDICTS=no ;;
    verdicts) PMI_PHASES=(); PMI_VERDICTS=yes ;;
    prepare) PMI_PHASES=(); PMI_VERDICTS=no ;;
    # имя переменной обязательно в скобках: в однобайтовой локали bash включает байты
    # символа «»» в имя, и под set -u сообщение падает с «unbound variable»
    *) pmi_die "неизвестная фаза «${PMI_PHASE}»: допустимы 1, 2, 3, tables, verdicts, prepare, all" ;;
esac

ADVISOR_TREE="${ADVISOR_TREE:-$PMI_ROOT/AutoIntent-advisor}"
ADVISOR_COMMIT="${ADVISOR_COMMIT:-b38f3c3a8612e177e76fe94552180ef43555665c}"
# AUTOINTENT_DIR ниже переопределяется на копию (pmi_init его не трогает — он только
# задаёт значение по умолчанию), поэтому путь предъявленного дерева сохраняется здесь
PMI_SAMPLE_TREE="$AUTOINTENT_DIR"

pmi_require_dir "$PMI_DIR/../advisor-calibration-laptop6gb" "каталог эксперимента калибровки advisor"
EXP_DIR="$(cd "$PMI_DIR/../advisor-calibration-laptop6gb" && pwd)"
[[ -x "$EXP_DIR/reproduce.sh" ]] || pmi_die "не найден исполняемый $EXP_DIR/reproduce.sh"
[[ -f "$EXP_DIR/harness/calibrate_advisor.py" ]] \
    || pmi_die "не найден $EXP_DIR/harness/calibrate_advisor.py — каталог эксперимента неполон"

ADVISOR_COMMIT_SHA="$(git -C "$PMI_SAMPLE_TREE" rev-parse --verify --quiet "$ADVISOR_COMMIT^{commit}")" \
    || pmi_die "в $PMI_SAMPLE_TREE нет коммита $ADVISOR_COMMIT, на котором закреплён сценарий воспроизведения; выполните git fetch --all в предъявленном дереве"

# Копия пересоздаётся на --check и на фазах prepare и all, а также если её нет, — так же,
# как в п. 6.10. Каталог, оставшийся от прошлого прогона, протаскивает в испытание чужое
# состояние рабочего дерева и окружения, а значит делает прогон невоспроизводимым. При
# запуске отдельной фазы копия берётся готовой: её создало действие 1 таблицы 12 ПМИ
# (запуск с --check), иначе каждая фаза заново разворачивала бы окружение образца
# (`uv sync` блока setup сценария воспроизведения).
if [[ -n "$PMI_CHECK_ONLY" || "$PMI_PHASE" == all || "$PMI_PHASE" == prepare || ! -d "$ADVISOR_TREE/.git" ]]; then
    pmi_log "создание рабочей копии дерева для сценария воспроизведения"
    # ADVISOR_TREE переопределяется переменной окружения, поэтому перед rm -rf
    # проверяем, что путь безопасен: непустой, абсолютный, не совпадает с
    # $PMI_SAMPLE_TREE/$PMI_ROOT и не является их родительским каталогом (иначе
    # rm -rf снёс бы предъявленное дерево или весь рабочий каталог испытаний).
    [[ -n "$ADVISOR_TREE" ]] || pmi_die "ADVISOR_TREE пуст — rm -rf отменён"
    [[ "$ADVISOR_TREE" = /* ]] || pmi_die "ADVISOR_TREE='$ADVISOR_TREE' не является абсолютным путём — rm -rf отменён"
    for guarded_dir in "$PMI_SAMPLE_TREE" "$PMI_ROOT" "$EXP_DIR"; do
        guarded_dir="${guarded_dir%/}"
        target_dir="${ADVISOR_TREE%/}"
        if [[ "$target_dir" == "$guarded_dir" || "$guarded_dir" == "$target_dir"/* ]]; then
            pmi_die "ADVISOR_TREE='$ADVISOR_TREE' совпадает с '$guarded_dir' или является его родительским каталогом — rm -rf отменён"
        fi
    done
    rm -rf "$ADVISOR_TREE"
    git clone --quiet --no-local "$PMI_SAMPLE_TREE" "$ADVISOR_TREE"
    # Коммит $ADVISOR_COMMIT достижим в предъявленном дереве только через
    # refs/remotes/origin/*, а `git clone` такие ссылки не переносит: без этой выборки
    # копия коммита не содержала бы, и fetch_commit сценария воспроизведения полез бы
    # за ним в сеть (по имени, затем refs/pull/348/head, затем fetch --all). Выбирается
    # точечно и форсированно (+): массовый refspec 'refs/remotes/origin/*:...' без
    # ведущего + отвергается как non-fast-forward, и set -e убил бы скрипт до проверки
    # ниже. Источник — локальный путь предъявленного дерева, обращения в сеть нет.
    git -C "$ADVISOR_TREE" fetch --quiet "$PMI_SAMPLE_TREE" "+$ADVISOR_COMMIT_SHA:refs/pmi/advisor"
fi
# Проверка идёт и на готовой копии: она ловит копию, оставшуюся от другого коммита.
git -C "$ADVISOR_TREE" rev-parse --verify --quiet "$ADVISOR_COMMIT_SHA^{commit}" > /dev/null \
    || pmi_die "копия $ADVISOR_TREE не содержит коммит ${ADVISOR_COMMIT_SHA:0:8}; пересоздайте её запуском с --check"

export AUTOINTENT_DIR="$ADVISOR_TREE"
export AUTOINTENT_COMMIT="$ADVISOR_COMMIT_SHA"

cd "$EXP_DIR"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    ./reproduce.sh --check 2>&1 | tee "$PMI_RUN_DIR/check.log"
    pmi_log "CHECK OK: копия дерева создана заново, коммит ${ADVISOR_COMMIT_SHA:0:8} в ней есть, окружение и вычислительная установка проверены"
    exit 0
fi

# Код возврата фазы переживает tee: set -o pipefail включён в common.sh.
for phase in ${PMI_PHASES[@]+"${PMI_PHASES[@]}"}; do
    pmi_log "фаза $phase"
    ./reproduce.sh "$phase" 2>&1 | tee "$PMI_RUN_DIR/phase-$phase.log"
done

# Фаза verdicts — действия 2—4 таблицы 12 ПМИ. Вердикты снимаются с ПРЕДЪЯВЛЕННОГО
# дерева, а не с копии: в $ADVISOR_COMMIT подсистема размещена в пакете
# autointent._advisor, импорта autointent.advisor там нет, а проверке подлежит
# подсистема в составе предъявленного образца. В каталоге эксперимента нет
# pyproject.toml, поэтому рабочий каталог для uv задаётся явно.
if [[ "$PMI_VERDICTS" == yes ]]; then
    pmi_log "отдельные вердикты на предъявленной ревизии $PMI_SAMPLE_TAG"
    ( cd "$PMI_SAMPLE_TREE" && uv run python -c 'from autointent.advisor import detect_hardware; print(detect_hardware())' ) \
        2>&1 | tee "$PMI_RUN_DIR/sample-hardware.log"
    # Код возврата inspect не подменяется собственным: он и есть вердикт (0 —
    # выполнимо, ненулевой — бюджет превышен) и печатается в вывод сценария.
    for preset in classic-light transformers-heavy; do
        rc=0
        ( cd "$PMI_SAMPLE_TREE" && uv run autointent-advisor inspect "$preset" \
            --dataset DeepPavlov/banking77 --json ) \
            > "$PMI_RUN_DIR/inspect-$preset.json" 2> "$PMI_RUN_DIR/inspect-$preset.log" || rc=$?
        pmi_log "inspect $preset: код возврата $rc (ожидается 0 для classic-light, ненулевой для transformers-heavy)"
    done
fi

pmi_log "проверка неприкосновенности предъявленного дерева"
pmi_assert_tree_untouched "$PMI_SAMPLE_TREE"
pmi_log "предъявленное дерево не изменено"
