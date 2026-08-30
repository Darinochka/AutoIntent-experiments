#!/usr/bin/env bash
# Проверка 6.6 ПМИ — многометочная классификация на наборе GoEmotions.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.6_multilabel.sh
#
# Фаза 1 выбирает пару целевых метрик на разделе validation, фаза 2 строит
# кривую способности на непересекающемся разделе test. Параметр --eval-split
# обязателен: по умолчанию sweep.py оценивает на разделе validation
# (SweepConfig.eval_split, src/sweep.py), и без явного значения фаза 2
# выродилась бы в повтор фазы 1.
#
# Каталог эксперимента самостоятельный (не входит в предъявленное дерево образца):
# сценарий сам переходит в него и не касается $AUTOINTENT_DIR.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.6 "$@"

EXP_DIR="$PMI_DIR/../multilabel-goemotions"
pmi_require_dir "$EXP_DIR" "каталог эксперимента многометочной классификации"
[[ -f "$EXP_DIR/sweep.py" ]] || pmi_die "не найден $EXP_DIR/sweep.py"

DEVICE="${DEVICE:-cuda}"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    cd "$EXP_DIR"
    uv run sweep.py --dry-run 2>&1 | tee "$PMI_RUN_DIR/dry-run.log"
    pmi_log "CHECK OK: план прогонов построен без ошибок конфигурации"
    exit 0
fi

cd "$EXP_DIR"

pmi_log "подготовка набора данных"
uv run prepare_data.py 2>&1 | tee "$PMI_RUN_DIR/prepare.log"

pmi_log "фаза 1 — выбор пары целевых метрик на разделе validation"
uv run sweep.py --device "$DEVICE" --sizes 10 --balances classwise --seeds 1 2 \
    2>&1 | tee "$PMI_RUN_DIR/phase1.log"

pmi_log "фаза 2 — кривая способности на разделе test"
uv run sweep.py --device "$DEVICE" \
    --eval-split test \
    --sizes 5 10 25 50 100 \
    --balances classwise \
    --scoring-metrics scoring_neg_coverage \
    --decision-metrics decision_f1 \
    --seeds 1 2 3 \
    2>&1 | tee "$PMI_RUN_DIR/phase2.log"

pmi_log "построение кривой"
uv run plot_curve.py 2>&1 | tee "$PMI_RUN_DIR/plot.log"
