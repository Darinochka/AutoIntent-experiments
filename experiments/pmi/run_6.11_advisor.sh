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
# Фазы 1, 2, 3, tables идут в копии дерева AutoIntent-advisor/ на закреплённом
# коммите b38f3c3a; verdicts снимает вердикты с ПРЕДЪЯВЛЕННОЙ ревизии (действия
# 2—4 таблицы 12 ПМИ); prepare только пересоздаёт копию.
# Журналы, копии JSON и манифест sha256 — в $RUNS_DIR/6.11/. Сценарий
# воспроизведения пишет JSON в advisor-calibration-laptop6gb/results/ (действия
# 5, 7, 11 таблицы 12), ПЕРЕЗАПИСЫВАЯ закоммиченный эталонный набор; вернуть его:
#   git -C AutoIntent-experiments restore experiments/advisor-calibration-laptop6gb/results
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
    || pmi_die "в $PMI_SAMPLE_TREE нет коммита $ADVISOR_COMMIT, на котором закреплён сценарий воспроизведения.
Ветка, которой он принадлежал (feat/issue39-calibration-scripts), в репозитории образца удалена
вместе с закрытием запроса на слияние #348, поэтому \`git fetch --all\` его НЕ добудет. Долговечный
путь — ссылка запроса на слияние (её же использует сам сценарий воспроизведения, refs/pull/348/head):
    git -C \"$PMI_SAMPLE_TREE\" fetch origin refs/pull/348/head
После этого повторите запуск. Шаг требует доступа к репозиторию образца."

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
    #
    # Ссылка-приёмник refs/pmi/advisor обязательна и удалению не подлежит: без неё
    # добранный объект удерживался бы только FETCH_HEAD и стал бы кандидатом на сборку
    # мусора между запуском `--check`/`--phase prepare` и запуском фазы часами позже.
    git -C "$ADVISOR_TREE" fetch --quiet "$PMI_SAMPLE_TREE" "+$ADVISOR_COMMIT_SHA:refs/pmi/advisor"
fi
# Проверка идёт и на готовой копии: она ловит копию, оставшуюся от другого коммита.
git -C "$ADVISOR_TREE" rev-parse --verify --quiet "$ADVISOR_COMMIT_SHA^{commit}" > /dev/null \
    || pmi_die "копия $ADVISOR_TREE не содержит коммит ${ADVISOR_COMMIT_SHA:0:8}; пересоздайте её запуском с --check"

export AUTOINTENT_DIR="$ADVISOR_TREE"
export AUTOINTENT_COMMIT="$ADVISOR_COMMIT_SHA"

# --- снятие собственных свидетельств прогона ---------------------------------
# Сценарий воспроизведения складывает JSON фаз в $EXP_DIR/results/ (OUT_DIR по
# умолчанию) — туда же, где лежит закоммиченный эталонный набор из восьми файлов,
# и его же перезаписывает. Само по себе это не опасно (файлы восстанавливаются
# `git restore`), но делает результат прогона неотличимым от эталона: `collect`
# сценария воспроизведения при промахе шаблона печатает «nothing matched» и НЕ
# завершается ошибкой, а страж фазы tables считает файлы `phase2_*.json`, которые
# в каталоге есть всегда. Оборвавшаяся фаза 2 оставила бы эталонные файлы на месте,
# и последующая фаза tables свела бы эталонные числа с кодом возврата 0 — ровно тот
# исход, который действие 14 таблицы 12 предписывает принять.
# Поэтому после каждой фазы содержимое results/ копируется в $PMI_RUN_DIR/results/
# вместе с манифестом (sha256 и время изменения каждого файла): утверждение «эти
# числа получены этим прогоном» становится проверяемым.
if command -v sha256sum > /dev/null 2>&1; then
    PMI_SHA256=(sha256sum)
elif command -v shasum > /dev/null 2>&1; then
    PMI_SHA256=(shasum -a 256)
else
    pmi_die "не найдено средство подсчёта sha256 (sha256sum или shasum) — снять свидетельства прогона нечем"
fi

# BSD stat (macOS) и GNU stat (Linux) несовместимы по флагам; второй вызов —
# запасной путь, отсюда `|| stat -c`.
pmi_file_mtime() {
    stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$1" 2>/dev/null || stat -c '%y' "$1"
}

# pmi_capture_results <фаза> <код возврата фазы>
# Вызывается и при успехе, и при падении фазы: посмертное состояние results/ —
# ровно то свидетельство, которое нужно оценщику, когда фаза оборвалась.
pmi_capture_results() {
    local phase="$1" phase_rc="$2" dest manifest marker verdict stamp f copy count=0
    dest="$PMI_RUN_DIR/results"
    manifest="$PMI_RUN_DIR/results-manifest-$phase.txt"
    marker="$dest/_snapshot.txt"
    stamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    # однострочно: шапка манифеста разбирается как «строки на # — шапка, остальные — данные»
    if [[ "$phase_rc" -eq 0 ]]; then
        verdict="фаза завершилась успешно"
    else
        verdict="ФАЗА ЗАВЕРШИЛАСЬ ОШИБКОЙ (код возврата $phase_rc) — содержимое results/ могло остаться от предыдущего прогона или быть закоммиченным эталонным набором; за результат текущей проверки не принимать"
    fi
    mkdir -p "$dest"
    {
        printf '# фаза %s; снято %s\n' "$phase" "$stamp"
        printf '# итог: %s\n' "$verdict"
        printf '# источник: %s/results\n' "$EXP_DIR"
        printf '# суммы сняты с КОПИЙ в %s\n' "$dest"
        printf '# sha256  время изменения  имя файла\n'
    } > "$manifest"
    # nullglob: при пустом каталоге цикл не должен получить сам шаблон
    shopt -s nullglob
    for f in "$EXP_DIR/results"/*.json; do
        copy="$dest/$(basename "$f")"
        cp -p "$f" "$copy"
        # сумма и время читаются с КОПИИ, а не с источника: манифест удостоверяет
        # именно содержимое каталога артефактов (cp -p делает их равными, но
        # утверждение о копии проверяемо без обращения к источнику)
        printf '%s  %s  %s\n' \
            "$("${PMI_SHA256[@]}" "$copy" | cut -d' ' -f1)" \
            "$(pmi_file_mtime "$copy")" \
            "$(basename "$copy")" >> "$manifest"
        count=$((count + 1))
    done
    shopt -u nullglob
    # Метка в самом каталоге снимка: оценщик, открывший $PMI_RUN_DIR/results/,
    # должен видеть, после успешной или после упавшей фазы он снят, не сверяясь
    # с журналами.
    {
        printf 'фаза: %s\n' "$phase"
        printf 'код возврата фазы: %s\n' "$phase_rc"
        printf 'итог: %s\n' "$verdict"
        printf 'снято: %s\n' "$stamp"
        printf 'файлов в снимке: %s\n' "$count"
        printf 'манифест: %s\n' "$manifest"
        printf 'журнал фазы: %s\n' "$PMI_RUN_DIR/phase-$phase.log"
    } > "$marker"
    pmi_log "свидетельства фазы $phase (код возврата $phase_rc): $count файлов скопировано в $dest, манифест $manifest, метка $marker"
}

cd "$EXP_DIR"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    ./reproduce.sh --check 2>&1 | tee "$PMI_RUN_DIR/check.log"
    pmi_log "CHECK OK: копия дерева создана заново, коммит ${ADVISOR_COMMIT_SHA:0:8} в ней есть, окружение и вычислительная установка проверены"
    exit 0
fi

# Код возврата фазы переживает tee: set -o pipefail включён в common.sh.
for phase in ${PMI_PHASES[@]+"${PMI_PHASES[@]}"}; do
    # Фаза 1 документа запускает у сценария воспроизведения ДВЕ фазы: 1 и 1b.
    # Фаза 1b (метаданный контрфактуал, ~2 мин, только CPU) в перечне --phase
    # документа не значится, но её результат нужен фазе tables: рендерер вызывается
    # с --counterfactual "$OUT_DIR/phase1b_metadata_counterfactual.json" безусловно.
    # Без запуска 1b фаза tables читала бы файл, закоммиченный в каталоге эксперимента,
    # то есть подмешивала бы в отчёт чужие числа. Сценарий воспроизведения принимает
    # несколько фаз одним вызовом (разбор аргументов накапливает их в PHASES,
    # строки 59—67 reproduce.sh; его собственная справка приводит `./reproduce.sh 1 1b 3`).
    phase_args=("$phase")
    [[ "$phase" == 1 ]] && phase_args=(1 1b)
    pmi_log "фаза $phase (${phase_args[*]})"
    # Код возврата снимается в переменную, а не отдаётся на откуп set -e: снятие
    # свидетельств обязано отработать ИМЕННО на упавшей фазе — ради этого случая
    # оно и заведено. `|| phase_rc=$?` отменяет действие set -e для этого конвейера;
    # сам код возврата берётся с учётом pipefail (включён в common.sh), то есть
    # падение ./reproduce.sh не теряется в tee.
    phase_rc=0
    ./reproduce.sh "${phase_args[@]}" 2>&1 | tee "$PMI_RUN_DIR/phase-$phase.log" || phase_rc=$?
    pmi_capture_results "$phase" "$phase_rc"
    if [[ "$phase_rc" -ne 0 ]]; then
        # выходим кодом самой фазы: успешное снятие свидетельств не должно
        # превращать упавшую проверку в успешную
        printf 'ОШИБКА: фаза %s завершилась с кодом возврата %s; журнал %s, снимок results/ и манифест сняты\n' \
            "$phase" "$phase_rc" "$PMI_RUN_DIR/phase-$phase.log" >&2
        exit "$phase_rc"
    fi
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
