# RPicam

Контроль входа/выхода из помещения по лицам: две камеры (ВХОД и ВЫХОД),
детекция (YOLO) + распознавание (InsightFace ArcFace), журнал посещаемости в
SQLite с выгрузкой в CSV. Интерфейс — PyQt5.

```
main.py            GUI и потоки захвата видео
recognition.py     детекция/распознавание, база эмбеддингов, подписи по-русски
attendance_db.py   SQLite: персонал и журнал входов/выходов
qt_compat.py       Linux-совместимость Qt (см. раздел «Linux» ниже)
setup_linux.sh     установка зависимостей на Fedora/Ubuntu одной командой
```

Рядом с `main.py` должны лежать:

- `best.pt` — веса детектора лиц;
- `known_faces/<person_id>/*.jpg` — фотографии персонала (подпапка на человека);
- `face_db.pkl` — кэш эмбеддингов (создаётся автоматически при первом запуске).

---

## Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Порядок импортов в `main.py` (cv2/torch/supervision → PyQt5) на Windows менять
нельзя: иначе возможен конфликт DLL (`OSError WinError 1114` при загрузке
`c10.dll`).

---

## Linux (Fedora)

### Быстрый вариант

```bash
./setup_linux.sh
source .venv/bin/activate
python main.py
```

Скрипт ставит системные библиотеки для Qt (`libxcb*`, `libxkbcommon*`,
`mesa-libGL`, …), создаёт `.venv` и ставит OpenCV в **headless**-сборке.

### Вручную

```bash
# 1. Системные библиотеки, без которых Qt-плагин xcb не загружается
sudo dnf install -y libxcb xcb-util-wm xcb-util-image xcb-util-keysyms \
    xcb-util-renderutil xcb-util-cursor libxkbcommon libxkbcommon-x11 \
    mesa-libGL mesa-libEGL libX11-xcb libSM libICE fontconfig dbus-libs glib2 \
    python3-devel gcc gcc-c++ make

# 2. Окружение (системный Python в Fedora — «externally managed», pip --user в него не пустит)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools numpy cython

# 3. Зависимости
pip install -r requirements-base.txt

# 4. OpenCV без Qt — ключевой шаг для Linux
pip uninstall -y opencv-python opencv-contrib-python
pip install --no-deps opencv-python-headless

python main.py
```

`--no-deps` в шаге 4 обязателен: `ultralytics` и `supervision` объявляют
зависимость от `opencv-python` и иначе pip затянет его обратно.

### Почему venv, а не «напрямую в Fedora»

Это не дань моде — для данного проекта есть четыре конкретные причины.

**1. Системный Python в Fedora закрыт для pip (PEP 668).**
Рядом с интерпретатором лежит маркер `/usr/lib/python3.14/EXTERNALLY-MANAGED`,
и pip отказывается ставить пакеты и в систему, и (в большинстве комбинаций pip + Fedora)
через `--user`. Обход только один — `--break-system-packages`, то есть
осознанно снять защиту. Внутри venv pip работает как обычно.

**2. Общий site-packages отменит наш главный фикс.**
Лечение конфликта Qt — это `pip uninstall opencv-python` +
`pip install --no-deps opencv-python-headless`. В общем каталоге
`~/.local/lib/python3.14/site-packages` **любой** другой проект, которому
понадобится `ultralytics`, `mediapipe` или просто `opencv-python`, молча вернёт
обычную сборку поверх headless — и RPicam снова перестанет запускаться, уже без
всякой видимой причины. `cv2` — это одно и то же имя каталога у обоих пакетов,
соседствовать они не могут. В своём venv никто чужой нам ничего не
переустановит.

**3. `~/.local` привязан к конкретной минорной версии Python.**
Каталог называется `python3.14`, и после обновления Fedora до Python 3.15 он
станет просто невидим для нового интерпретатора — пересобирать придётся всё
(torch, cv2, numpy, insightface). С venv то же самое лечится одной командой:
`rm -rf .venv && ./setup_linux.sh`. А если сразу сделать окружение на более
консервативном Python, оно переживёт несколько обновлений дистрибутива:

```bash
sudo dnf install -y python3.12
PYTHON=python3.12 ./setup_linux.sh
```

Это, кстати, не только про надёжность: для Python 3.14 у PyTorch пока нет
CUDA-сборок (ставится CPU-вариант), а `insightface` на 3.14 собирается из
исходников. На 3.11–3.12 весь стек (torch, onnxruntime, insightface, PyQt5)
ставится готовыми колесами.

**4. Системным Python пользуется сама Fedora.**
Если перезаписать там numpy/Qt/что-нибудь ещё, поломаться могут системные
утилиты. venv живёт в папке проекта, не требует root и удаляется вместе с ней.

> А на Windows я ставил напрямую — и работало.
>
> Потому что там «напрямую» и есть изолированная установка: Python с
> python.org принадлежит только вам, никакого системного пакетного менеджера
> рядом нет и PEP 668 не применяется. Linux-venv — это аналог того же самого
> положения дел, а не дополнительная надстройка.

Если venv всё же принципиально не хочется, рабочий минимум —
`pip install --user --break-system-packages -r requirements-linux.txt`, но пункт
2 выше при этом остаётся в силе: окружение RPicam будет общим со всеми
остальными вашими проектами.

### Почему на Linux ломается, а на Windows — нет

Ошибка выглядит так:

```
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.
QObject::moveToThread: Current thread (0x...) is not the object's thread (0x...).
Cannot move to target thread (0x...)
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
    "/home/USER/.local/lib/python3.14/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
```

Причина — **не** Fedora, Wayland и не «сломанная установка». Виноват пакет
`opencv-python`: он собран с GUI-бэкендом Qt5, возит собственную копию Qt5 и при
импорте перезаписывает переменную окружения, по которой Qt ищет платформенные
плагины. В `cv2/config-3.py` буквально написано:

```python
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qt", "plugins")
    os.environ["QT_QPA_FONTDIR"] = os.path.join(..., "qt", "fonts")
```

`QT_QPA_PLATFORM_PLUGIN_PATH` имеет наивысший приоритет, поэтому PyQt5 находит в
`cv2/qt/plugins` **чужой** `libqxcb.so`, собранный против Qt5Core из OpenCV
(auditwheel переименовывает её в `libQt5Core-<hash>.so.5.15.19`), и не может его
загрузить. Отсюда и `QObject::moveToThread` — это не отдельная проблема, а тот же
сбой: экземпляр плагина создаёт одна Qt5Core, а `moveToThread` ему вызывает
другая, и их thread-local storage не совпадает. Список «Available platform
plugins» при этом перечисляет плагины PyQt5 — падает именно PyQt5, споткнувшись
о каталог cv2.

На Windows ветка `sys.platform.startswith("linux")` не срабатывает, поэтому там
всё и запускалось.

**Два уровня защиты в этом репозитории:**

1. `opencv-python-headless` — Qt в OpenCV отсутствует как класс, конфликтовать
   нечему. GUI от OpenCV приложению всё равно не нужен: ни `cv2.imshow`, ни
   `cv2.namedWindow` в коде не используются.
2. `qt_compat.py` — страховка на случай, если обычный `opencv-python` всё-таки
   установлен (например, его вернул pip как зависимость `ultralytics`).
   `main.py` вызывает `qt_compat.preload_qt()` **до** `import cv2` и
   `qt_compat.restore_qt_paths()` **сразу после** — пути к плагинам PyQt5
   возвращаются на место. На Windows обе функции — no-op.

### Диагностика

```bash
python qt_compat.py                    # что за OpenCV/PyQt5, каких библиотек не хватает
QT_DEBUG_PLUGINS=1 python main.py      # подробный лог загрузки плагинов Qt
RPICAM_QT_DEBUG=1 python main.py       # лог самого qt_compat
```

`python qt_compat.py` прогоняет `ldd` по `libqxcb.so` и печатает готовые команды
`dnf`/`apt` для недостающих библиотек.

### Wayland

Предупреждение

```
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.
```

безвредно: Qt5 в GNOME на Wayland рисует через XWayland. Для нативного Wayland:

```bash
QT_QPA_PLATFORM=wayland python main.py
```

Если окно так и не появляется — проверьте, что сессия вообще графическая:
`echo $DISPLAY $WAYLAND_DISPLAY`.

### Камеры

В Linux индекс камеры — это `/dev/videoN`. Список устройств:

```bash
v4l2-ctl --list-devices        # пакет: sudo dnf install v4l-utils
```

Номера камер входа/выхода задаются в интерфейсе перед запуском.
