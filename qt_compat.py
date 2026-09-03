"""
qt_compat — «клей» между Qt из PyQt5 и Qt, который тащит за собой OpenCV.

Зачем нужен (проблема на Linux)
-------------------------------
Пакет ``opencv-python`` собран с GUI-бэкендом Qt5 и возит с собой СВОЮ копию
Qt5: библиотеки в ``opencv_python.libs/`` (auditwheel переименовывает их,
например ``libQt5Core-9088e21b.so.5.15.19``) и плагин
``cv2/qt/plugins/platforms/libqxcb.so``.

А теперь главное. В ``cv2/config-3.py``, который выполняется при ``import cv2``,
написано буквально следующее:

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qt", "plugins")
        os.environ["QT_QPA_FONTDIR"] = os.path.join(..., "qt", "fonts")

То есть OpenCV безусловно перезаписывает переменную окружения, по которой Qt
ищет платформенные плагины, и указывает на СВОЙ каталог. ``QT_QPA_PLATFORM_PLUGIN_PATH``
имеет наивысший приоритет, поэтому PyQt5 находит там чужой ``libqxcb.so`` и
пытается его загрузить. А тот слинкован с переименованной Qt5Core из OpenCV, то
есть с совершенно другим экземпляром Qt — загрузка падает, и Qt аварийно
завершает процесс:

    QObject::moveToThread: Current thread (0x...) is not the object's thread (0x...)
    Cannot move to target thread (0x...)

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
        "/home/USER/.local/lib/python3.14/site-packages/cv2/qt/plugins"
        even though it was found.

    This application failed to start because no Qt platform plugin could be
    initialized.

Первые две строчки — не отдельная проблема, а тот же сбой: загрузчик плагинов Qt
создаёт QObject силами Qt5Core из OpenCV, а ``moveToThread`` ему вызывает Qt5Core
из PyQt5 — у них разная thread-local storage, поэтому потоки «не совпадают».
Список «Available platform plugins» в сообщении при этом перечисляет плагины
PyQt5 (wayland-xcomposite-glx, webgl и т. д.) — ещё одно подтверждение, что
падает именно PyQt5, споткнувшись о каталог cv2.

На Windows этого нет (там ``sys.platform.startswith("linux")`` ложно, и cv2
свою Qt не возит), поэтому весь модуль на Windows просто ничего не делает и
порядок импортов в main.py остаётся прежним.

Что делаем
----------
1. ``preload_qt()``       — вызывать ДО ``import cv2``. На Linux заранее
                            прописываем путь к плагинам PyQt5 и первыми
                            загружаем её Qt5Core/Qt5Gui.
2. ``restore_qt_paths()`` — вызывать СРАЗУ ПОСЛЕ ``import cv2``. Вычищаем
                            пути к плагинам, подставленные OpenCV, и
                            возвращаем пути к плагинам PyQt5.
3. ``doctor()``           — самодиагностика: ``python qt_compat.py``.

Радикальный (и самый надёжный) вариант
--------------------------------------
GUI от OpenCV этому приложению не нужен: ни ``cv2.imshow``, ни
``cv2.namedWindow`` в коде не используются. Поэтому на Linux можно просто
убрать Qt из OpenCV целиком:

    pip uninstall -y opencv-python opencv-contrib-python
    pip install --no-deps opencv-python-headless

(``ultralytics`` и ``supervision`` объявляют зависимость от ``opencv-python``,
поэтому headless ставим с ``--no-deps``, иначе pip вернёт обычную сборку.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")

#: Нужна ли вообще возня с путями Qt. На Windows cv2 свою Qt не подсовывает.
NEEDS_FIX = IS_LINUX

#: Сюда кладём найденный каталог плагинов PyQt5, чтобы не искать его дважды.
_STATE: dict = {}

_VERBOSE = bool(os.environ.get("RPICAM_QT_DEBUG"))


def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"[qt_compat] {msg}")


def _warn(msg: str) -> None:
    print(f"[qt_compat] {msg}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────
#  Где лежат плагины PyQt5
# ────────────────────────────────────────────────────────────────
def _pyqt5_dir() -> str | None:
    """Каталог установленного пакета PyQt5 (импорт самого PyQt5 Qt не грузит)."""
    try:
        import PyQt5  # noqa: F401  — только чтобы узнать путь
    except Exception as exc:                                    # pragma: no cover
        _log(f"PyQt5 не найден: {exc}")
        return None
    return os.path.dirname(os.path.abspath(PyQt5.__file__))


def find_pyqt5_plugins() -> str | None:
    """Каталог с плагинами PyQt5 (внутри него есть подкаталог ``platforms``)."""
    if _STATE.get("plugins"):
        return _STATE["plugins"]

    base = _pyqt5_dir()
    candidates = []
    if base:
        candidates += [os.path.join(base, rel) for rel in
                       ("Qt5/plugins", "Qt/plugins", "plugins")]

    for path in candidates:
        if os.path.isdir(os.path.join(path, "platforms")):
            _STATE["plugins"] = path
            _log(f"плагины PyQt5: {path}")
            return path

    # запасной вариант — спросить у самой Qt (системная сборка PyQt5, conda, …)
    try:
        from PyQt5 import QtCore
        path = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
        if path and os.path.isdir(os.path.join(path, "platforms")):
            _STATE["plugins"] = path
            _log(f"плагины PyQt5 (QLibraryInfo): {path}")
            return path
    except Exception as exc:                                    # pragma: no cover
        _log(f"QLibraryInfo недоступен: {exc}")

    return None


def _pyqt5_libs_dir() -> str | None:
    """Каталог с библиотеками Qt из PyQt5 (нужен для диагностики)."""
    base = _pyqt5_dir()
    if not base:
        return None
    for rel in ("Qt5/lib", "Qt/lib", "lib"):
        path = os.path.join(base, rel)
        if os.path.isdir(path):
            return path
    return None


# ────────────────────────────────────────────────────────────────
#  Основная логика
# ────────────────────────────────────────────────────────────────
def preload_qt() -> None:
    """
    Вызывать ПЕРЕД ``import cv2``.

    На Linux: заранее прописываем путь к плагинам PyQt5 и загружаем
    libQt5Core/libQt5Gui из PyQt5 — чтобы к моменту ``import cv2`` в процессе
    уже была «правильная» Qt, а переменные окружения потом просто
    восстанавливались функцией :func:`restore_qt_paths`.

    На Windows/macOS — no-op, порядок импортов там критичен по-другому
    (см. комментарий про supervision/torch в main.py).
    """
    if not NEEDS_FIX:
        return

    plugins = find_pyqt5_plugins()
    if plugins:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(
            [plugins] + _other_plugin_paths(plugins))

    try:
        from PyQt5 import QtCore, QtGui        # noqa: F401 — грузим Qt5Core/Qt5Gui
        _log("Qt из PyQt5 загружена первой")
    except Exception as exc:                                    # pragma: no cover
        _warn(f"не удалось предварительно загрузить PyQt5: {exc}")


def restore_qt_paths() -> None:
    """
    Вызывать СРАЗУ ПОСЛЕ ``import cv2`` (и ещё раз перед QApplication).

    Отменяет то, что сделал cv2: убирает из окружения его каталоги плагинов и
    возвращает пути к плагинам PyQt5. Идемпотентна — можно звать сколько угодно.
    """
    if not NEEDS_FIX:
        return

    plugins = _STATE.get("plugins") or find_pyqt5_plugins()

    if plugins:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(
            [plugins] + _other_plugin_paths(plugins))
        # шрифты: если cv2 подсунул свой fontdir — снимаем, у PyQt5 своего нет
        fontdir = os.environ.get("QT_QPA_FONTDIR", "")
        if _is_cv2_path(fontdir):
            os.environ.pop("QT_QPA_FONTDIR", None)
        _log(f"пути Qt восстановлены -> {plugins}")
    else:
        # PyQt5 не нашли: как минимум убираем подставленное cv2,
        # тогда Qt возьмёт свои пути по умолчанию.
        for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH"):
            if _is_cv2_path(os.environ.get(var, "")):
                os.environ.pop(var, None)
                _log(f"убрал {var} (указывал на cv2)")

    _check_platform_plugin(plugins)


def _other_plugin_paths(exclude: str) -> list:
    """Текущий QT_PLUGIN_PATH без каталогов, принадлежащих cv2."""
    current = os.environ.get("QT_PLUGIN_PATH", "")
    out = []
    for path in current.split(os.pathsep):
        if path and not _is_cv2_path(path) and os.path.abspath(path) != os.path.abspath(exclude):
            out.append(path)
    return out


def _is_cv2_path(path: str) -> bool:
    """Путь ведёт внутрь пакета cv2 (там лежит чужая копия Qt)."""
    if not path:
        return False
    norm = path.replace("\\", "/").lower()
    return "/cv2/" in norm or norm.endswith("/cv2")


# ────────────────────────────────────────────────────────────────
#  Проверки «а точно ли запустится?»
# ────────────────────────────────────────────────────────────────
def _check_platform_plugin(plugins: str | None) -> None:
    """
    Мягкая проверка перед созданием QApplication: если плагин xcb отсутствует
    или не может слинковаться, подсказываем, что доустановить.
    """
    if not NEEDS_FIX or not plugins or _STATE.get("checked"):
        return
    _STATE["checked"] = True

    xcb = os.path.join(plugins, "platforms", "libqxcb.so")
    if not os.path.exists(xcb):
        _warn(f"не найден плагин xcb: {xcb}")
        _warn("проверьте установку PyQt5:  pip install --force-reinstall PyQt5")
        return

    missing = missing_libs(xcb)
    if not missing:
        return

    fedora, debian, unknown = [], [], []
    for lib in missing:
        pkgs = PACKAGE_HINTS.get(lib)
        if pkgs:
            if pkgs[0] not in fedora:
                fedora.append(pkgs[0])
            if pkgs[1] not in debian:
                debian.append(pkgs[1])
        else:
            unknown.append(lib)

    _warn(f"Qt-плагину xcb не хватает системных библиотек: {', '.join(missing)}")
    if fedora:
        _warn("  Fedora        : sudo dnf install -y " + " ".join(sorted(fedora)))
    if debian:
        _warn("  Debian/Ubuntu : sudo apt install -y " + " ".join(sorted(debian)))
    if unknown:
        _warn(f"  без подсказки : {', '.join(unknown)} — см. `python qt_compat.py`")


#: (Fedora, Debian/Ubuntu) — куда смотреть, если ldd показывает «not found».
PACKAGE_HINTS = {
    "libxcb-xinerama.so.0":     ("libxcb", "libxcb-xinerama0"),
    "libxcb-randr.so.0":        ("libxcb", "libxcb-randr0"),
    "libxcb-shape.so.0":        ("libxcb", "libxcb-shape0"),
    "libxcb-sync.so.1":         ("libxcb", "libxcb-sync1"),
    "libxcb-xfixes.so.0":       ("libxcb", "libxcb-xfixes0"),
    "libxcb-xkb.so.1":          ("libxcb", "libxcb-xkb1"),
    "libxcb-glx.so.0":          ("libxcb", "libxcb-glx0"),
    "libxcb-icccm.so.4":        ("xcb-util-wm", "libxcb-icccm4"),
    "libxcb-image.so.0":        ("xcb-util-image", "libxcb-image0"),
    "libxcb-keysyms.so.1":      ("xcb-util-keysyms", "libxcb-keysyms1"),
    "libxcb-render-util.so.0":  ("xcb-util-renderutil", "libxcb-render-util0"),
    "libxcb-cursor.so.0":       ("xcb-util-cursor", "libxcb-cursor0"),
    "libxcb-shm.so.0":          ("libxcb", "libxcb-shm0"),
    "libxcb-render.so.0":       ("libxcb", "libxcb-render0"),
    "libxcb-util.so.1":         ("xcb-util", "libxcb-util1"),
    "libxkbcommon.so.0":        ("libxkbcommon", "libxkbcommon0"),
    "libxkbcommon-x11.so.0":    ("libxkbcommon-x11", "libxkbcommon-x11-0"),
    "libGL.so.1":               ("mesa-libGL", "libgl1"),
    "libEGL.so.1":              ("mesa-libEGL", "libegl1"),
    "libX11-xcb.so.1":          ("libX11", "libx11-xcb1"),
    "libSM.so.6":               ("libSM", "libsm6"),
    "libICE.so.6":              ("libICE", "libice6"),
    "libfontconfig.so.1":       ("fontconfig", "libfontconfig1"),
    "libdbus-1.so.3":           ("dbus-libs", "libdbus-1-3"),
    "libglib-2.0.so.0":         ("glib2", "libglib2.0-0"),
}

#: Одной строкой для Fedora, если не хочется разбираться поштучно.
FEDORA_DNF_LINE = (
    "sudo dnf install -y libxcb xcb-util-wm xcb-util-image xcb-util-keysyms "
    "xcb-util-renderutil xcb-util-cursor libxkbcommon libxkbcommon-x11 "
    "mesa-libGL mesa-libEGL libX11-xcb libSM libICE fontconfig dbus-libs glib2"
)


def missing_libs(so_path: str) -> list:
    """Список «not found» из вывода ldd для указанной .so (пусто, если всё ок)."""
    ldd = shutil.which("ldd")
    if not ldd or not os.path.exists(so_path):
        return []
    try:
        proc = subprocess.run([ldd, so_path], capture_output=True, text=True, timeout=30)
    except Exception:                                           # pragma: no cover
        return []
    out = []
    for line in (proc.stdout + proc.stderr).splitlines():
        if "not found" in line:
            name = line.strip().split()[0]
            if name and name not in out:
                out.append(name)
    return out


def cv2_qt_info() -> dict:
    """Что за OpenCV установлен и возит ли он с собой Qt."""
    info = {"installed": False, "version": None, "headless": None, "qt_dir": None}
    try:
        import cv2
    except Exception:
        return info
    info["installed"] = True
    info["version"] = getattr(cv2, "__version__", "?")
    try:
        from cv2.version import ci_build, headless              # noqa: F401
        info["headless"] = bool(headless)
    except Exception:
        try:
            from cv2 import config
            info["headless"] = bool(getattr(config, "HEADLESS", False))
        except Exception:
            info["headless"] = None
    qt_dir = os.path.join(os.path.dirname(os.path.abspath(cv2.__file__)), "qt")
    info["qt_dir"] = qt_dir if os.path.isdir(qt_dir) else None
    return info


# ────────────────────────────────────────────────────────────────
#  Самодиагностика:  python qt_compat.py
# ────────────────────────────────────────────────────────────────
def doctor() -> int:
    """Печатает всё, что нужно знать про Qt/OpenCV на этой машине."""
    print("=" * 68)
    print("RPicam — диагностика Qt")
    print("=" * 68)
    print(f"Python          : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Платформа       : {sys.platform}")
    print(f"Сессия          : XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE', '?')}, "
          f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '-')}, "
          f"DISPLAY={os.environ.get('DISPLAY', '-')}")
    print(f"QT_QPA_PLATFORM : {os.environ.get('QT_QPA_PLATFORM', '(не задана, Qt выберет сама)')}")

    print("-" * 68)
    try:
        import PyQt5.QtCore as qc
        print(f"PyQt5           : {qc.QT_VERSION_STR} (Qt runtime {qc.qVersion()})")
    except Exception as exc:
        print(f"PyQt5           : НЕ УСТАНОВЛЕН ({exc})")

    plugins = find_pyqt5_plugins()
    print(f"Плагины PyQt5   : {plugins or 'не найдены'}")
    print(f"Библиотеки PyQt5: {_pyqt5_libs_dir() or 'не найдены'}")

    print("-" * 68)
    cv = cv2_qt_info()
    if not cv["installed"]:
        print("OpenCV          : НЕ УСТАНОВЛЕН")
    else:
        headless = {True: "да (Qt внутри нет — хорошо)",
                    False: "нет (возит свою Qt5 — источник конфликта)",
                    None: "не удалось определить"}[cv["headless"]]
        print(f"OpenCV          : {cv['version']}, headless: {headless}")
        print(f"Каталог cv2/qt  : {cv['qt_dir'] or 'отсутствует'}")

    print("-" * 68)
    if plugins:
        xcb = os.path.join(plugins, "platforms", "libqxcb.so")
        if os.path.exists(xcb):
            miss = missing_libs(xcb)
            if miss:
                print(f"libqxcb.so      : НЕ ХВАТАЕТ {len(miss)} системных библиотек:")
                fedora, debian = set(), set()
                for lib in miss:
                    pkgs = PACKAGE_HINTS.get(lib)
                    print(f"    - {lib}" + (f"   -> {pkgs[0]} / {pkgs[1]}" if pkgs else ""))
                    if pkgs:
                        fedora.add(pkgs[0])
                        debian.add(pkgs[1])
                print()
                if fedora:
                    print("  Fedora       : sudo dnf install -y " + " ".join(sorted(fedora)))
                if debian:
                    print("  Debian/Ubuntu: sudo apt install -y " + " ".join(sorted(debian)))
            else:
                print("libqxcb.so      : все зависимости на месте ✔")
        else:
            print(f"libqxcb.so      : ОТСУТСТВУЕТ ({xcb})")
            print("                  переустановите PyQt5: pip install --force-reinstall PyQt5")

    print("=" * 68)
    if cv["installed"] and cv["headless"] is False:
        print("РЕКОМЕНДАЦИЯ: убрать Qt из OpenCV —")
        print("    pip uninstall -y opencv-python opencv-contrib-python")
        print("    pip install --no-deps opencv-python-headless")
    if IS_LINUX:
        print("Если xcb всё равно не грузится, доустановите системные библиотеки —")
        print("   " + FEDORA_DNF_LINE)
    print("Подробный лог Qt:  QT_DEBUG_PLUGINS=1 python main.py")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(doctor())
