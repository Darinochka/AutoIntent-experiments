import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)

try:
    from autointent.configs import HFModelConfig
    from autointent.modules.scoring import BertScorer
except ImportError as e:
    print(f"Ошибка импорта. Проверить пакет autointent {e}")
    exit(1)

MODEL_NAME = "prajjwal1/bert-tiny"
MODEL_CONFIG = HFModelConfig(model_name=MODEL_NAME, tokenizer_config={}, device="cpu")
NUM_EPOCHS = 1
BATCH_SIZE = 2

train_utterances = [
    "Покажи погоду в Москве",
    "Какая завтра погода?",
    "включи свет на кухне",
    "сделай свет поярче",
]
train_labels = ["weather", "weather", "lights", "lights"]

test_utterances = [
    "прогноз погоды на сегодня",
    "выключи свет",
]

scorer_original = BertScorer(
    model_config=MODEL_CONFIG,
    num_train_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    report_to="none",
)

print("--- Обучение модели ---")
scorer_original.fit(train_utterances, train_labels)
print("Обучение завершено.")

predictions_before = None
try:
    print("--- Предсказание ДО сохранения ---")
    predictions_before = scorer_original.predict(test_utterances)
    print("Предсказания ДО:\n", predictions_before)
except Exception as e:
    print(f"Ошибка при первом предсказании: {e}")
    exit(1)


temp_dir_path = Path(tempfile.mkdtemp(prefix="bert_scorer_test_"))
print(f"\n--- Сохранение в {temp_dir_path} ---")

try:
    scorer_original.dump(str(temp_dir_path))
    print("Сохранение завершено.")
except Exception as e:
    print(f"Ошибка во время dump: {e}")

scorer_loaded = BertScorer(
    model_config=MODEL_CONFIG,
    num_train_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    report_to="none",
)

print("\n--- Загрузка ---")
try:
    scorer_loaded.load(str(temp_dir_path))
    print("Загрузка завершена.")

    model_loaded = hasattr(scorer_loaded, "_model") and scorer_loaded._model is not None
    tokenizer_loaded = (
        hasattr(scorer_loaded, "_tokenizer") and scorer_loaded._tokenizer is not None
    )
    print(f"Атрибут _model {'ЗАГРУЖЕН' if model_loaded else 'НЕ ЗАГРУЖЕН'}")
    print(f"Атрибут _tokenizer {'ЗАГРУЖЕН' if tokenizer_loaded else 'НЕ ЗАГРУЖЕН'}")

except Exception as e:
    print(f"Ошибка во время load: {e}")

predictions_after = None
test_passed = False
try:
    print("\n--- Предсказание ПОСЛЕ загрузки ---")
    predictions_after = scorer_loaded.predict(test_utterances)
    print("Предсказания ПОСЛЕ:\n", predictions_after)

    if predictions_before is not None and predictions_after is not None:
        if predictions_before.shape == predictions_after.shape and np.allclose(
            predictions_before, predictions_after, atol=1e-6
        ):
            print("\nТест ПРОЙДЕН: Предсказания совпадают.")
            test_passed = True
        else:
            print("\nТест ПРОВАЛЕН: Предсказания различаются.")
    else:
        print("\nТест ПРОВАЛЕН: Не удалось получить предсказания до или после.")

except RuntimeError as e:
    print(
        f"\nТест ПРОВАЛЕН: Произошла ошибка RuntimeError во время predict после load: {e}"
    )
    if "Model is not trained" in str(e):
        print("Модель не была загружена.")
except Exception as e:
    print(
        f"\nТест ПРОВАЛЕН: Произошла НЕПРЕДВИДЕННАЯ ошибка во время predict после load: {e}"
    )

finally:
    print(f"\n--- Удаление временной директории {temp_dir_path} ---")
    shutil.rmtree(temp_dir_path)
    print("Очистка завершена.")

    if not test_passed:
        print("\nИТОГ: dump/load НЕ работает корректно для BertScorer.")
    else:
        print("\nИТОГ: dump/load работает корректно для BertScorer.")
