<h1>Распознавание эмоций</h1>
<h3>Структура:</h3>
<ul>
    <li>main.py - код приложения</li>
    <li>config.json - параметры приложения</li>
    <li>detection_model.caffemodel и proto.txt - нейросеть для обнаружения лиц</li>
    <li>main_notebook.ipynb - тетрадь с обучением нейросети</li>
    <li>emotion_recognition.h5 - обученная нейросеть по распознаванию эмоций</li>
    <li>examples - примеры для настройки приложения</li>
</ul>

<h1>Инструкция по установке</h1>
<h3>Для того, чтобы установить программу на свой компьютер, нужно:</h3>
<ol>
    <li>Скачать репозиторий на свой компьютер</li>
    <li>Загрузить все необходимые библиоткеи с помощью ввода в консоль команды pip install -r requirements.txt</li>
    <li>Собрать код в исполняемый файл с помощью команды pyinstaller --onefile --icon=icon.ico --noconsole main.py</li>
    <li>Скопировать в папку dist папку examples и файлы config.json, detection_model.caffemodel, proto.txt, emotion_recognition.h5.</li>
</ol>
