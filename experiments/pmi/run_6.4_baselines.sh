#!/usr/bin/env bash
# Проверка 6.4 ПМИ — прогон базовых AutoML-фреймворков для сопоставления по OOS.
#
# Запуск из корня рабочего каталога испытаний:
#     bash AutoIntent-experiments/experiments/pmi/run_6.4_baselines.sh
#
# Обертка над опубликованным кодом эксперимента (automl_frameworks.py): модели,
# пресеты и начальные значения генератора не переопределяются. Внедоменные запросы
# задаются дополнительной меткой класса функцией load_data опубликованного модуля.
# Каждый фреймворк ставится в отдельное временное окружение uv: их зависимости
# несовместимы. H2O поднимает собственный сервер и требует Java.
# Версия интерпретатора закреплена (см. PMI_PYTHON ниже) по requires-python
# опубликованного эксперимента.
# Набор данных читается с хаба HuggingFace: офлайн-зеркало datasets/ хранит его
# в формате образца и опубликованным кодом не используется.
# Ориентировочное время выполнения — несколько часов.
# Аргументы: --check (только проверка предусловий), --help.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
pmi_init 6.4 "$@"

SWEEP_DIR="$PMI_DIR/../sweep-whole-search-space"
DATASET=DeepPavlov/clinc150

# Опубликованный эксперимент объявляет requires-python = "==3.12.*"
# (sweep-whole-search-space/pyproject.toml). Без явного закрепления `uv run --no-project`
# взял бы интерпретатор по умолчанию машины, что молча меняет окружение относительно
# того, в котором получены справочные значения ПМИ.
PMI_PYTHON=3.12

pmi_require_dir "$SWEEP_DIR" "опубликованный код базовых фреймворков"
[[ -f "$SWEEP_DIR/automl_frameworks.py" ]] || pmi_die "не найден $SWEEP_DIR/automl_frameworks.py"
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

export WANDB_MODE=offline  # опубликованный модуль импортирует wandb; ключ доступа не нужен
export TOKENIZERS_PARALLELISM=false

# Оба артефакта удаляются вместе. Если снести только накопительный журнал, прерванный
# повторный прогон оставил бы свежий частичный baselines.json рядом с полной
# baselines_summary.json от прошлого раза — пара свидетельств, которая выглядит
# согласованной и таковой не является.
FAILURES="$PMI_RUN_DIR/baselines_failures.txt"
rm -f "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" "$FAILURES"

# Отказ одного фреймворка не прерывает второй: каждый отказ записывается в FAILURES,
# а итоговый код возврата сценария его отражает (та же схема, что в run_6.3_baselines.sh).
run_framework() {  # <фреймворк> <дополнительные зависимости для uv>...
    local framework="$1"; shift
    local code
    pmi_log "$framework на $DATASET"
    if uv run --no-project --python "$PMI_PYTHON" \
        --with pandas --with datasets --with scikit-learn --with wandb "$@" \
        python "$PMI_DIR/lib/baselines_oos.py" \
            --framework "$framework" \
            --dataset "$DATASET" \
            --out "$PMI_RUN_DIR/baselines.json"
    then
        :
    else
        code=$?
        echo "ОШИБКА: $framework на $DATASET завершился с кодом $code;" \
             "запись в журнал не добавлена, прогон продолжается"
        echo "$framework $DATASET код=$code" >> "$FAILURES"
    fi
}

{
    run_framework gluon-high --with 'autogluon>=1.3.0'
    run_framework h2o --with 'h2o>=3.46.0.7'

    # Сводка считается всегда, в том числе по неполному журналу: частичный результат
    # тоже свидетельство, если честно помечен.
    pmi_log "сводка по журналу базовых фреймворков"
    python3 - "$PMI_RUN_DIR/baselines.json" "$PMI_RUN_DIR/baselines_summary.json" <<'PY'
import json
import os
import sys

journal, out_path = sys.argv[1], sys.argv[2]

if os.path.isfile(journal):
    with open(journal, encoding="utf-8") as file:
        records = json.load(file)
else:
    records = []
    print(f"ВНИМАНИЕ: журнал {journal} отсутствует — ни один прогон не дал записи")

summary = {record["framework"]: record for record in records}
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
pmi_log "оба прогона базовых фреймворков завершились успешно"
