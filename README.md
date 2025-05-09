# mlops-vehicle-insurance

Has bad design and bugs. Needs a lot more work. I hope to get some advice.

The model itself should predict whether claim was paid, but it wasn't tuned yet and tends to just always predict false.

# Usage:

python run.py -mode update - adds batch to the training data and fits the model with it. Data is from data/motor_data11-14lats.csv.
You can adjust batchsize in run.py

python run.py -mode summary - get some information about data quality and scores. Crashes if too little data is added

python run.py -mode inference -file <filename> - get predictions for data in file. Crashes if too little data is added

# TODO
- not using ohe on MAKE
- better data quality analysis
- more model quality scores
- partial_fit
- better handling of streamed data
- better tuned model
- second model for CLAIM_PAID regression
- separate directories for stages
- data quality metaparameters and data drift detection
- interface for using existing model vesioning system
- performance metrics

# Прогресс:
## 1. Сбор данных (3–10 баллов):

(a) Обязательная часть:

i. Функционал сбора потоковых данных: разделение исходного набора на
батчи и эмуляция потока (1 балл); :white_check_mark: 

ii. Разработка хранилища сырых данных: файловая система (1 балл) или
БД (2 балла); :white_check_mark: 

iii. Расчет метапараметров: (1-2 балла).

(b) Дополнительные баллы:

i. Создание конфигурационного файла (.py или YAML/JSON/TOML/XML)
с гиперпараметрами сбора (1 балл);

ii. Интеграция с несколькими источниками данных (2 балла);

iii. Система логирования (с использованием библиотек или вручную) и
обработки ошибок при сборе данных (1-2 балла).

## 2. Анализ данных (2–10 баллов):

(a) Обязательная часть:

i. Оценка и хранение показателей качества данных (data quality) (1-2
балла); :white_check_mark: 

ii. Базовая очистка данных на основе порогов допустимых значений ка-
чества (1 балл). :white_check_mark: 

(b) Дополнительные баллы:

i. Автоматический EDA (1-2 балла);

ii. Добавление Feature Engineering (1-2 балла);

iii. Генерация отчетов о качестве данных (1 балла);

iv. Мониторинг и обработка ситуаций data drift (1-2 балл).

## 3. Подготовка данных (входит в pipeline построения модели) (0–5 бал-
лов):

(a) Обязательная часть зависит от используемой модели ML:

i. Обработка пропусков (0-1 балл); :white_check_mark: 

ii. Обработка категориальных переменных (0-1 балл); :white_check_mark: 

iii. Обработка числовых переменных (0-1 балл). :white_check_mark: 

(b) Дополнительные баллы:

i. Создание нескольких вариантов предобработки с дальнейшим перебо-
ром при поиске лучшей модели (1-2 балла).

## 4. Обучение/дообучение модели (входит в pipeline построения модели)
(1–5 баллов):

(a) Обязательная часть: построение модели (LR, kNN или дерево решений) (1
балл).

(b) Дополнительные баллы:

4i. Реализация дообучения предыдущей модели (без обучения модели с
нуля) (1-2 балла);

ii. Разработка нескольких моделей с различной устойчивостью к входным
данным (1-2 балла).

## 5. Валидация модели (2–10 баллов):

(a) Обязательная часть:

i. Оценка качества модели/моделей (hold-out/CV/TimeSeriesCV) (1-3
балла); :white_check_mark: 

ii. Разработка хранилища версий моделей и контроль качества (1-2 бал-
ла). :white_check_mark:

(b) Дополнительные баллы:

i. Интерпретация прогнозов (визуализация структуры дерева, оценка ко-
эффициентов LR, демонстрация ближайших соседей, LIME, SHAP) (1-
3 балла);

ii. Мониторинг и обработка ситуаций model drift (1–2 балла).

## 6. Обслуживание модели (1–6 баллов):

(a) Обязательная часть: выбор и упаковка (сериализация) финальной модели
(или нескольких) (1-2 балла); :white_check_mark: 

(b) Дополнительные баллы:

i. Мониторинг производительности (времени применения/памяти) (1-2
балла);

ii. Обеспечение гибкого прогноза на основе данных (выбор модели при
разреженных данных или аномальных значениях) (1-2 балла).

## 7. Управление программой (3–11 баллов):

(a) Обязательная часть:

i. Создание скрипта управления конвейером и обработка запросов
(Inference, Update, Summary) (2 балла); :white_check_mark: 

ii. Написание документации к программной реализации (README и
requirements) (1 балл).

(b) Дополнительные баллы:

i. Построение расширенного отчета (dashboard) о работе системы (1-2
балла);

ii. Создание конфигурационного файла параметров всех компонентов си-
стемы (размер батча, допустимое число пропусков, тип валидации и
т.д.) (1-2 балла);

iii. Создание Meta Learning модели: оценка влияющих гиперпараметров и
динамика метрик качества моделей и данных (2 балла).

iv. Построение архитектуры на основе паттернов проектирования (напри-
мер, MVC (model/view/controller)) ПО (1-2 балла).
