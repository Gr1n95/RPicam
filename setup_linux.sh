#!/usr/bin/env bash
#
# Установка RPicam на Linux (в первую очередь Fedora; Ubuntu/Debian тоже умеет).
#
#   ./setup_linux.sh                # полная установка
#   SKIP_SYSTEM_PKGS=1 ./setup_linux.sh   # только Python-зависимости
#   PYTHON=python3.13 ./setup_linux.sh    # другой интерпретатор
#
# Скрипт специально ставит OpenCV в headless-сборке: обычный opencv-python
# тащит собственную копию Qt5, из-за которой PyQt5 на Linux падает с
#   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
#       ".../site-packages/cv2/qt/plugins" even though it was found.
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SKIP_SYSTEM_PKGS="${SKIP_SYSTEM_PKGS:-0}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
    warn "нет sudo и мы не root — системные пакеты придётся поставить вручную"
fi

# ─────────────────────────────────────────────────────────────────────
# 1. Системные библиотеки: их требует Qt-плагин xcb и сборка insightface
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_SYSTEM_PKGS" = "1" ]; then
    say "Пропускаю установку системных пакетов (SKIP_SYSTEM_PKGS=1)"
elif command -v dnf >/dev/null 2>&1; then
    say "Fedora/RHEL: ставлю системные зависимости"
    $SUDO dnf install -y \
        libxcb xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil \
        xcb-util-cursor libxkbcommon libxkbcommon-x11 \
        mesa-libGL mesa-libEGL libX11-xcb libSM libICE \
        fontconfig dbus-libs glib2 \
        python3-devel gcc gcc-c++ make
elif command -v apt-get >/dev/null 2>&1; then
    say "Debian/Ubuntu: ставлю системные зависимости"
    $SUDO apt-get update -y
    $SUDO apt-get install -y \
        libxcb1 libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-sync1 \
        libxcb-xfixes0 libxcb-xkb1 libxcb-glx0 libxcb-icccm4 libxcb-image0 \
        libxcb-keysyms1 libxcb-render-util0 libxcb-cursor0 libxcb-util1 \
        libxkbcommon0 libxkbcommon-x11-0 libgl1 libegl1 libx11-xcb1 \
        libsm6 libice6 libfontconfig1 libdbus-1-3 libglib2.0-0 \
        python3-dev python3-venv build-essential
else
    warn "Незнакомый пакетный менеджер — поставьте xcb/xkbcommon/mesa-libGL вручную"
fi

# ─────────────────────────────────────────────────────────────────────
# 2. Виртуальное окружение
#    (на Fedora системный Python «externally managed», pip --user в него не пустит)
# ─────────────────────────────────────────────────────────────────────
if [ ! -x "$VENV_DIR/bin/python" ]; then
    say "Создаю виртуальное окружение $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
say "Python: $(python -V) -> $(command -v python)"

python -m pip install --upgrade pip wheel setuptools
# cython и numpy нужны ДО insightface — он собирается из исходников
python -m pip install --upgrade "numpy>=1.24" cython

# ─────────────────────────────────────────────────────────────────────
# 3. Python-зависимости (OpenCV — только headless)
# ─────────────────────────────────────────────────────────────────────
say "Ставлю зависимости из requirements-base.txt"
python -m pip install -r requirements-base.txt

say "Убираю обычный opencv-python и ставлю headless-сборку"
python -m pip uninstall -y opencv-python opencv-contrib-python \
    opencv-python-headless opencv-contrib-python-headless || true
python -m pip install --no-deps "opencv-python-headless>=4.8"

# ─────────────────────────────────────────────────────────────────────
# 4. Диагностика Qt
# ─────────────────────────────────────────────────────────────────────
say "Проверяю Qt"
python qt_compat.py || true

cat <<EOF

$(printf '\033[1;32mГотово.\033[0m')

Запуск:
    source $VENV_DIR/bin/activate
    python main.py

Перед первым запуском положите рядом с main.py:
    best.pt         — веса детектора лиц (YOLO)
    known_faces/    — фото персонала, по подпапке на человека

Если окно всё равно не появляется:
    QT_DEBUG_PLUGINS=1 python main.py     # подробный лог загрузки плагинов Qt
    python qt_compat.py                   # чего не хватает в системе

Сессия GNOME на Wayland: предупреждение
    «Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome»
безвредно — Qt5 рисует через XWayland. Для нативного Wayland:
    QT_QPA_PLATFORM=wayland python main.py
EOF
