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
# Версия интерпретатора закреплена (см. PMI_PYTHON ниже) по requires-python
# опубликованного эксперимента.
# Наборы данных читаются с хаба HuggingFace: офлайн-зеркало datasets/ хранит их
# в формате образца и опубликованным кодом не используется.
# Ориентировочное время выполнения — до суток.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.3 "$@"

SWEEP_DIR="$PMI_DIR/../sweep-whole-search-space"
DATASETS=(DeepPavlov/banking77 DeepPavlov/hwu64 DeepPavlov/massive DeepPavlov/minds14 DeepPavlov/snips)

# Опубликованный эксперимент объявляет requires-python = "==3.12.*"
# (sweep-whole-search-space/pyproject.toml). Без явного закрепления `uv run --no-project`
# взял бы интерпретатор по умолчанию машины: на 3.13 разрешение `torch<2.7.0` и autogluon
# даёт другой набор версий или падает, то есть окружение молча перестаёт быть тем, в котором
# получены справочные значения ПМИ. Это то же расхождение, что и подмена пресета или сида.
PMI_PYTHON=3.12

pmi_require_dir "$SWEEP_DIR" "опубликованный код базовых фреймворков"
for module in frameworks_duration.py automl_frameworks.py; do
    [[ -f "$SWEEP_DIR/$module" ]] || pmi_die "не найден $SWEEP_DIR/$module"
done
command -v uv >/dev/null || pmi_die "не найден uv: им ставятся окружения базовых фреймворков"
uv python find "$PMI_PYTHON" >/dev/null 2>&1 \
    || pmi_die "uv не находит интерпретатор Python $PMI_PYTHON, требуемый опубликованным
   экспериментом (requires-python = \"==3.12.*\"); установите его: uv python install $PMI_PYTHON"
command -v java >/dev/null \
    || pmi_die "не найдена java: сервер H2O не стартует без JRE (нужна версия 8 или новее)"

if [[ -n "$PMI_CHECK_ONLY" ]]; then
    pmi_log "CHECK OK: опубликованный код на месте, uv, Python $PMI_PYTHON и java доступны"
    exit 0
fi

export WANDB_MODE=offline  # опубликованные модули импортируют wandb; ключ доступа не нужен
export TOKENIZERS_PARALLELISM=false

# Оба артефакта удаляются вместе. Если снести только накопительный журнал, прерванный
# повторный прогон оставил бы свежий частичный baselines.json рядом с полным
# baselines_summary.json от прошлого раза — пара свидетельств, которая выглядит
# согласованной и таковой не является.
FAILURES="$PMI_RUN_DIR/baselines_failures.txt"
rm -f "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" "$FAILURES"

# Отказ одного фреймворка не прерывает остальные: lama идет последней и в критерий
# приемки 6.3 не входит (известный дефект np.argmax), поэтому её падение через ~20 часов
# не должно уничтожать свидетельство по пятнадцати уже удавшимся прогонам. Каждый отказ
# записывается в FAILURES, а итоговый код возврата сценария его отражает.
run_framework() {  # <фреймворк> <дополнительные зависимости для uv>...
    local framework="$1"; shift
    local dataset code
    for dataset in "${DATASETS[@]}"; do
        pmi_log "$framework на $dataset"
        if uv run --no-project --python "$PMI_PYTHON" \
            --with pandas --with datasets --with scikit-learn --with wandb "$@" \
            python "$PMI_DIR/lib/baselines.py" \
                --framework "$framework" \
                --dataset "$dataset" \
                --out "$PMI_RUN_DIR/baselines.json"
        then
            :
        else
            code=$?
            echo "ОШИБКА: $framework на $dataset завершился с кодом $code;" \
                 "запись в журнал не добавлена, прогон продолжается"
            echo "$framework $dataset код=$code" >> "$FAILURES"
        fi
    done
}

{
    run_framework gluon-default --with 'autogluon>=1.3.0'
    run_framework gluon-high --with 'autogluon>=1.3.0'
    run_framework h2o --with 'h2o>=3.46.0.7'
    run_framework lama --with 'lightautoml[nlp]>=0.4.1' --with 'torch<2.7.0'

    # Сводка считается всегда, в том числе по неполному журналу: частичный результат
    # тоже свидетельство, если честно помечен. Поле "n" — число наборов, вошедших в avg;
    # без него частичное среднее нельзя отличить от полного.
    pmi_log "сводка по журналу базовых фреймворков"
    python3 - "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" \
              "${#DATASETS[@]}" <<'PY'
import json
import os
import sys
from collections import defaultdict

journal, out_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])

if os.path.isfile(journal):
    with open(journal, encoding="utf-8") as file:
        records = json.load(file)
else:
    records = []
    print(f"ВНИМАНИЕ: журнал {journal} отсутствует — ни один прогон не дал записи")

per_framework = defaultdict(dict)
for record in records:
    per_framework[record["framework"]][record["dataset"]] = record["accuracy"]
summary = {
    framework: {
        **per_dataset,
        "avg": round(sum(per_dataset.values()) / len(per_dataset), 2),
        "n": len(per_dataset),
    }
    for framework, per_dataset in per_framework.items()
}
for framework, values in summary.items():
    if values["n"] != expected:
        print(
            f"ВНИМАНИЕ: {framework} — {values['n']} наборов из {expected};"
            " avg посчитан по неполному составу и полным результатом не является"
        )
with open(out_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))
PY
} 2>&1 | tee "$PMI_RUN_DIR/baselines.log"

# Проверка отказов — после конвейера: run_framework выполняется в подоболочке конвейера,
# поэтому состояние передаётся файлом, а не переменной.
if [[ -f "$FAILURES" ]]; then
    pmi_log "прогон завершён с отказами (сводка по удавшимся прогонам сохранена):"
    cat "$FAILURES"
    exit 1
fi
pmi_log "все $(( ${#DATASETS[@]} * 4 )) прогонов базовых фреймворков завершились успешно"
