# Self-supervised and contrastive learning of visual embeddings
Данный репозиторий содержит код, реализованный в рамках проекта "Self-supervised and contrastive learning of visual embeddings".

## **1 Цели и предпосылки**
### **1.1 Обоснованность разработки проекта**
### **Цель**
Цель данного исследования: повышение качества эмбеддингов для видео или отрезков видео, путем обучения модели методами контрастивного обучения.

Задачи:
- улучшение эмбеддингов для видео - повышение похожести для кадров из разных частей видео (и/или разных видео)
- повысить консистентность идентификации эмоций на соседних кадрах

### **Польза от ML модели**
Результаты данного исследования и полученный сервис обучения могут применяться в разных доменах, основную часть которых составляет видео модальность, для повышения консистентности классификации, нахождения похожих видео-фрагментов, и, потенциально, повышения качества трекинга объектов на видео.

### **Критерии успеха:**
- вектора, получаемые из полученной модели имеют разделимые распределения, а именно

$$cosine\ similarity_{одинаковые\ эмоции}≥cosine\ similarity_{противоположные\ эмоции}$$

### 1.2 Что должно быть реализовано
- Сервис обучения модели для получения эмбеддингов видео
- Демонстрационный сервис для тестирования модели
### 1.3 Предпосылки решения
Планируется обучить модель и собрать результаты экспериментов и описание использованных методов в виде статьи.

**Ожидания на текущем этапе:**
- Проект ведётся в GitHub
- Документация и описание экспериментов - Notion
- Код: В JupyterNotebook + py модули, соответствует PEP8, разбит на модули
- Стек: PyTorch
- Интерфейс: Streamlit

## **2 Методология**
### **2.1 Постановка задачи**
Обучение модели-эмбеддера для видео.
### **2.2  Этапы решения задачи**
### **2.2.1 Этап 0 – Выбор данных**

|**Название датасета**|**Ссылка на данные**|**Разметка**|**Доступность**|
| :-: | :-: | :-: | :-: |
|ABAW|https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/|Для каждого фрейма|Доступен|
|AFEW|https://ibug.doc.ic.ac.uk/resources/afew-va-database/|Для всего видео|Доступен|
|RAVDESS|https://zenodo.org/records/1188976|Для всего видео|Доступен|
|CK+|http://www.jeffcohn.net/Resources/|-|Устарел|

На основании приведенных выше критериев был выбран датасет ABAW.

### **2.2.2 Этап 1 – Выбор базовой модели**
В качестве базовой модели была выбрана [models/affectnet_emotions/enet_b0_8_best_vgaf.pt](https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b0_8_best_vgaf.pt), так как данная модель была предобучена на задачу распознавания эмоций и показала хорошие результаты на бенчмарке.

Эксперименты с моделью были проведены в следующих условиях:
- в виде эмбеддера для каждого кадра, обучаемого на каждом шаге
- в виде замороженного эмбеддера, вектора которого используются для обучения сети поверх 

### **2.2.3 Этап 2 – Проведение экспериментов с параметрами обучения и способом обучения модели**
**Проведенные эксперименты можно разделить на три типа:**
- эксперименты с архитектурой
- эксперименты с лосс-функцией - выбор оптимальной функции, эксперименты с ее компонеентами
- эксперименты с семплингом - детали выбора негативных и hard-негативных примеров
### **2.2.4 Этап 3 – Подготовка демонстрационного сервиса**
Демонстрационный интерфейс для визуальной проверки работы модели


## 3. Запуск обучения

### Способ 1: Обучение в ноутбуке
1. Предварительно установить необходимое окружение:

```
pip install requirements.txt
```

2. Запустить jupyter или jupyter lab. Для обучения использовать `notebooks/train.ipynb`, для оценки `notebooks/evaluation.ipynb`

### Способ 2: Обучение через скрипты


1. Предварительно установить необходимое окружение:

```
pip install requirements.txt
```

2. Запустить обучение с помощью команды:

```
python main.py
```
3. Передавать настройки обучения либо через консоль, либо меняя аргументы в файле `main.py`:

```python
parser.add_argument('--dataset-dir', default='/content/dataset/train')
parser.add_argument('--val-dataset-dir', default='/content/dataset/val')
parser.add_argument('--test-dataset-dir', default='/content/dataset/test')
parser.add_argument('--checkpoint-dir', default='/content/drive/experiments', help='Checkpoint directory')
parser.add_argument('--model-path', default='/content/drive/enet_b0_8_best_vgaf.pt',
                    help='Model directory')
parser.add_argument('--model-class', default='ImageEmbedder', help='')

parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--batch-size', type=int, default=8, help='Number of examples for each iteration')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--optim-betas', type=list, nargs='+', default=[0.9, 0.999], help='Optimizer betas')
parser.add_argument('--weight-decay', type=float, default=0.01, help='Optimizer weight decay')

parser.add_argument('--loss', type=str, default='TripletLoss', help='Could be one of ArcFaceLoss, TripletLoss, ContrastiveCrossEntropy')

parser.add_argument('--seed', type=int, default=1004, help='Random seed value')
parser.add_argument('--device', default='cuda', help='Device to use for training: cpu or cuda')
```



