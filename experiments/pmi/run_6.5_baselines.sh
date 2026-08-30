#!/usr/bin/env bash
# Проверка 6.5 ПМИ — прогон базовых AutoML-фреймворков в режиме малого объема данных.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.5_baselines.sh
#
# Обертка над опубликованным кодом эксперимента (lib/baselines.py): подвыборка строится
# функцией create_few_shot_split с начальным значением генератора 42, AutoGluon берется
# с пресетом по умолчанию (аргумент presets в коде не передается), H2O AutoML — поверх
# word2vec. Модели, пресеты, ограничения по времени и сиды не переопределяются.
# Всего 36 прогонов: 3 набора x 6 значений n x 2 фреймворка; точки на полном обучающем
# наборе здесь нет (см. оговорку ниже).
# Каждый фреймворк ставится в отдельное временное окружение uv: их зависимости
# несовместимы. H2O поднимает собственный сервер и требует Java. Версия интерпретатора
# закреплена (см. PMI_PYTHON ниже) по requires-python опубликованного эксперимента.
# Наборы данных читаются с хаба HuggingFace: офлайн-зеркало datasets/ хранит их
# в формате образца и опубликованным кодом не используется.
# Ориентировочное время выполнения — до суток.
# Аргументы: --check (только проверка предусловий), --help.
#
# Почему прогона на полном обучающем наборе здесь нет. В опубликованном эксперименте
# (experiments/fewshot/data/comparison_few_shot.csv) точка full есть у H2O (по одной
# записи на каждый из трех наборов) и отсутствует у AutoGluon, поэтому сопоставление
# ведется по общим для обоих фреймворков точкам n = 4...128, величина деградации от
# полного набора вычисляется только для образца (runs/6.5/summary.json,
# degradation_at_16), а данные H2O на полном наборе в сопоставление не включаются.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.5 "$@"

SWEEP_DIR="$PMI_DIR/../sweep-whole-search-space"
DATASETS=(DeepPavlov/hwu64 DeepPavlov/minds14 DeepPavlov/snips)
SHOTS=(4 8 16 32 64 128)

# Опубликованный эксперимент объявляет requires-python = "==3.12.*"
# (sweep-whole-search-space/pyproject.toml). Без явного закрепления `uv run --no-project`
# взял бы интерпретатор по умолчанию машины, что молча меняет окружение относительно
# того, в котором получены справочные значения ПМИ.
PMI_PYTHON=3.12

pmi_require_dir "$SWEEP_DIR" "опубликованный код базовых фреймворков"
[[ -f "$SWEEP_DIR/frameworks_duration.py" ]] || pmi_die "не найден $SWEEP_DIR/frameworks_duration.py"
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
# повторный прогон оставил бы свежий частичный baselines.json рядом с полной
# baselines_summary.json от прошлого раза — пара свидетельств, которая выглядит
# согласованной и таковой не является.
FAILURES="$PMI_RUN_DIR/baselines_failures.txt"
rm -f "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" "$FAILURES"

# Отказ одного прогона не прерывает остальные: 36 прогонов идут до суток, и падение
# одного из них не должно уничтожать свидетельство по уже удавшимся. Каждый отказ
# записывается в FAILURES, а итоговый код возврата сценария его отражает
# (та же схема, что в run_6.3_baselines.sh).
run_framework() {  # <фреймворк> <дополнительные зависимости для uv>...
    local framework="$1"; shift
    local dataset shots code
    for dataset in "${DATASETS[@]}"; do
        for shots in "${SHOTS[@]}"; do
            pmi_log "$framework на $dataset, n=$shots"
            if uv run --no-project --python "$PMI_PYTHON" \
                --with pandas --with datasets --with scikit-learn --with wandb "$@" \
                python "$PMI_DIR/lib/baselines.py" \
                    --framework "$framework" \
                    --dataset "$dataset" \
                    --few-shot "$shots" \
                    --out "$PMI_RUN_DIR/baselines.json"
            then
                :
            else
                code=$?
                echo "ОШИБКА: $framework на $dataset при n=$shots завершился с кодом $code;" \
                     "запись в журнал не добавлена, прогон продолжается"
                echo "$framework $dataset n=$shots код=$code" >> "$FAILURES"
            fi
        done
    done
}

{
    run_framework gluon-default --with 'autogluon>=1.3.0'
    run_framework h2o --with 'h2o>=3.46.0.7'

    # Сводка считается всегда, в том числе по неполному журналу: частичный результат
    # тоже свидетельство, если честно помечен. Группировка — по ПАРЕ (framework, few_shot):
    # в 6.5 у одного фреймворка шесть конфигураций, различающихся только числом примеров
    # на класс, и агрегатор 6.3 (по одному framework) слил бы их в одну строку.
    # Поле "n" — число наборов, вошедших в avg; без него частичное среднее нельзя
    # отличить от полного.
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

per_point = defaultdict(dict)
for record in records:
    per_point[(record["framework"], record["few_shot"])][record["dataset"]] = record["accuracy"]

summary: dict = {}
for (framework, shots), per_dataset in per_point.items():
    summary.setdefault(framework, {})[shots] = {
        **per_dataset,
        "avg": round(sum(per_dataset.values()) / len(per_dataset), 2),
        "n": len(per_dataset),
    }
    if len(per_dataset) != expected:
        print(
            f"ВНИМАНИЕ: {framework} при n={shots} — {len(per_dataset)} наборов из {expected};"
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
pmi_log "все $(( ${#DATASETS[@]} * ${#SHOTS[@]} * 2 )) прогонов базовых фреймворков завершились успешно"
