#!/usr/bin/env bash
# Проверка 6.3 ПМИ — прогон базовых AutoML-фреймворков для сопоставления.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.3_baselines.sh
#
# Обертка над опубликованным кодом эксперимента (каталог sweep-whole-search-space):
# модели, пресеты и начальные значения генератора не переопределяются.
# Каждый фреймворк ставится в отдельное временное окружение uv: их зависимости
# между собой несовместимы. H2O поднимает собственный сервер и требует Java.
# Наборы данных читаются с хаба HuggingFace: офлайн-зеркало datasets/ хранит их
# в формате образца и опубликованным кодом не используется.
# Ориентировочное время выполнения — до суток.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.3 "$@"

SWEEP_DIR="$PMI_DIR/../sweep-whole-search-space"
DATASETS=(DeepPavlov/banking77 DeepPavlov/hwu64 DeepPavlov/massive DeepPavlov/minds14 DeepPavlov/snips)

pmi_require_dir "$SWEEP_DIR" "опубликованный код базовых фреймворков"
for module in frameworks_duration.py automl_frameworks.py; do
    [[ -f "$SWEEP_DIR/$module" ]] || pmi_die "не найден $SWEEP_DIR/$module"
done
command -v uv >/dev/null || pmi_die "не найден uv: им ставятся окружения базовых фреймворков"
command -v java >/dev/null \
    || pmi_die "не найдена java: сервер H2O не стартует без JRE (нужна версия 8 или новее)"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "CHECK OK: опубликованный код на месте, uv и java доступны"
    exit 0
fi

export WANDB_MODE=offline  # опубликованные модули импортируют wandb; ключ доступа не нужен
export TOKENIZERS_PARALLELISM=false

rm -f "$PMI_RUN_DIR/baselines.json"  # журнал накопительный, старые записи не смешиваем

run_framework() {  # <фреймворк> <дополнительные зависимости для uv>...
    local framework="$1"; shift
    local dataset
    for dataset in "${DATASETS[@]}"; do
        pmi_log "$framework на $dataset"
        uv run --no-project \
            --with pandas --with datasets --with scikit-learn --with wandb "$@" \
            python "$PMI_DIR/lib/baselines.py" \
                --framework "$framework" \
                --dataset "$dataset" \
                --out "$PMI_RUN_DIR/baselines.json"
    done
}

{
    run_framework gluon-default --with 'autogluon>=1.3.0'
    run_framework gluon-high --with 'autogluon>=1.3.0'
    run_framework h2o --with 'h2o>=3.46.0.7'
    run_framework lama --with 'lightautoml[nlp]>=0.4.1' --with 'torch<2.7.0'

    python3 - "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" <<'PY'
import json
import sys
from collections import defaultdict

records = json.load(open(sys.argv[1], encoding="utf-8"))
per_framework = defaultdict(dict)
for record in records:
    per_framework[record["framework"]][record["dataset"]] = record["accuracy"]
summary = {
    framework: {**per_dataset, "avg": round(sum(per_dataset.values()) / len(per_dataset), 2)}
    for framework, per_dataset in per_framework.items()
}
with open(sys.argv[2], "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))
PY
} 2>&1 | tee "$PMI_RUN_DIR/baselines.log"
