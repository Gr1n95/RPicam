"""
Система учёта посещаемости по лицам — графический интерфейс (PyQt5).

Возможности:
  • Живое видео с камеры с детекцией и распознаванием лиц.
  • Автоматический учёт входа/выхода (чередование) с защитой от повторов.
  • База данных персонала (SQLite) с журналом входов/выходов.
  • Добавление персонала: ФИО, должность + загрузка фото из файлов.
  • Просмотр журнала, фильтр по дате, экспорт в CSV.

Запуск:  python main.py
Зависимости:  pip install -r requirements.txt          (Windows)
              pip install -r requirements-linux.txt    (Linux, OpenCV без Qt)

Linux: пакет opencv-python тащит собственную копию Qt5 и конфликтует с PyQt5
       («Could not load the Qt platform plugin "xcb" in .../cv2/qt/plugins»).
       Модуль qt_compat.py это обходит; подробности и диагностика — в README.md.
"""
import os
import sys
import time
import shutil

# Linux: ДО импорта cv2 загружаем Qt из PyQt5 и фиксируем пути к её плагинам.
# Иначе opencv-python подтянет свою копию Qt5, подменит QT_QPA_PLATFORM_PLUGIN_PATH
# на .../site-packages/cv2/qt/plugins — и PyQt5 не сможет инициализировать xcb.
# На Windows qt_compat ничего не делает, порядок импортов ниже сохраняется.
import qt_compat
qt_compat.preload_qt()

# ВАЖНО (Windows): supervision нужно импортировать ДО PyQt5,
# иначе возможен конфликт DLL (OSError WinError 1114 при загрузке c10.dll).
# Поэтому recognition (тянет torch) импортируется первым.
import cv2
qt_compat.restore_qt_paths()   # cv2 только что подставил свои пути Qt — отменяем
import numpy as np
import supervision as sv

import recognition as rec
from recognition import (
    FaceRecognizer, FaceDatabase, crop_face, load_model, draw_label_ru,
    KNOWN_FACES_DIR, FACE_DB_PATH, MODEL_PATH, CONFIDENCE,
    SIMILARITY_THRESHOLD, RECOGNITION_CTX_ID,
)
from attendance_db import AttendanceDB

# PyQt5 импортируем ПОСЛЕ torch-стека.
from PyQt5 import QtCore, QtGui, QtWidgets


# ════════════════════════════════════════════════════════════════
#  Поток обработки видео
# ════════════════════════════════════════════════════════════════
class VideoThread(QtCore.QThread):
    """
    Один поток = одна камера с ФИКСИРОВАННЫМ направлением:
      direction='IN'  — камера входа,
      direction='OUT' — камера выхода.
    Сигнал frame_ready несёт (направление, кадр), чтобы окно понимало, куда рисовать.
    """
    frame_ready = QtCore.pyqtSignal(str, np.ndarray)     # (direction, кадр BGR)
    event_logged = QtCore.pyqtSignal(str, str, float)    # (person_id, direction, similarity)
    status = QtCore.pyqtSignal(str)

    def __init__(self, model, recognizer, face_db, att_db, cam_index, direction):
        super().__init__()
        self.model = model
        self.recognizer = recognizer
        self.face_db = face_db
        self.att_db = att_db
        self.cam_index = cam_index
        self.direction = direction               # 'IN' или 'OUT'
        self._running = False

    def run(self):
        role = "ВХОД" if self.direction == "IN" else "ВЫХОД"
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cam_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            time.sleep(0.3)
        if not cap.isOpened():
            self.status.emit(f"[{role}] Не удалось открыть камеру {self.cam_index}")
            return

        self._running = True
        self.status.emit(f"[{role}] Камера {self.cam_index} запущена")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status.emit(f"[{role}] Потерян сигнал с камеры {self.cam_index}")
                break

            results = self.model(frame, conf=CONFIDENCE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                det_conf = (float(detections.confidence[i])
                            if detections.confidence is not None else 0.0)

                face_crop = crop_face(frame, bbox)
                emb = self.recognizer.embed(face_crop)

                if emb is not None:
                    person_id, sim = self.face_db.search(emb, SIMILARITY_THRESHOLD)
                else:
                    person_id, sim = None, 0.0

                if emb is None:
                    color = (0, 200, 200)
                    label = f"no embed {det_conf:.0%}"
                elif person_id is not None:
                    name = self.att_db.get_staff_name(person_id)
                    # направление задаётся камерой
                    state, direction = self.att_db.register_directed(
                        person_id, self.direction, sim)
                    if state == "logged":
                        color = (0, 255, 0)
                        self.event_logged.emit(person_id, direction, sim)
                        dir_txt = "ВХОД" if direction == "IN" else "ВЫХОД"
                        label = f"{name} | {dir_txt}"
                    elif state == "duplicate":
                        # уже в нужном состоянии — показываем серым, не пишем
                        color = (160, 160, 160)
                        label = f"{name} (уже {'внутри' if self.direction=='IN' else 'снаружи'})"
                    else:  # cooldown
                        color = (0, 200, 0)
                        label = f"{name} {sim*100:.0f}%"
                else:
                    color = (0, 0, 255)
                    label = "UNKNOWN"

                # Отрисовка рамки + подписи 
                frame = draw_label_ru(frame, (x1, y1, x2, y2), label, color)

            self.frame_ready.emit(self.direction, frame)

        cap.release()
        self.status.emit(f"[{role}] Камера остановлена")

    def stop(self):
        self._running = False
        self.wait(2000)


#  Диалог добавления персонала
class AddStaffDialog(QtWidgets.QDialog):
    def __init__(self, parent, recognizer, face_db, att_db):
        super().__init__(parent)
        self.recognizer = recognizer
        self.face_db = face_db
        self.att_db = att_db
        self.image_paths = []

        self.setWindowTitle("Добавление сотрудника")
        self.setMinimumWidth = 460
        self.resize(480, 360)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.ed_id = QtWidgets.QLineEdit()
        self.ed_id.setPlaceholderText("например: ivanov_ii (латиницей, без пробелов)")
        self.ed_name = QtWidgets.QLineEdit()
        self.ed_name.setPlaceholderText("Иванов Иван Иванович")
        self.ed_pos = QtWidgets.QLineEdit()
        self.ed_pos.setPlaceholderText("Инженер")

        form.addRow("ID (логин):", self.ed_id)
        form.addRow("ФИО:", self.ed_name)
        form.addRow("Должность:", self.ed_pos)
        layout.addLayout(form)

        btn_pick = QtWidgets.QPushButton("Выбрать фото лица…")
        btn_pick.clicked.connect(self.pick_images)
        layout.addWidget(btn_pick)

        self.lbl_files = QtWidgets.QLabel("Фото не выбраны")
        self.lbl_files.setWordWrap(True)
        self.lbl_files.setStyleSheet("color: #555;")
        layout.addWidget(self.lbl_files)

        layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pick_images(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выберите фото лица", "",
            "Изображения (*.jpg *.jpeg *.png *.bmp)")
        if paths:
            self.image_paths = paths
            names = ", ".join(os.path.basename(p) for p in paths)
            self.lbl_files.setText(f"Выбрано {len(paths)} фото: {names}")

    def save(self):
        person_id = self.ed_id.text().strip()
        full_name = self.ed_name.text().strip()
        position = self.ed_pos.text().strip()

        if not person_id or not full_name:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Заполните ID и ФИО.")
            return
        if not self.image_paths:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одно фото лица.")
            return

        # 1) Считаем эмбеддинги
        ok, errors = self.face_db.add_person_from_images(
            person_id, self.image_paths, self.recognizer)

        if ok == 0:
            msg = "Не удалось извлечь ни одного лица.\n\n" + "\n".join(errors)
            QtWidgets.QMessageBox.critical(self, "Ошибка", msg)
            return

        # 2) Копируем фото в known_faces/<id>/ для возможности пересборки
        person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)
        for idx, src in enumerate(self.image_paths):
            ext = os.path.splitext(src)[1] or ".jpg"
            try:
                shutil.copy(src, os.path.join(person_dir, f"{idx}{ext}"))
            except Exception:
                pass

        # 3) Сохраняем базу эмбеддингов и запись в SQLite
        self.face_db.save()
        self.att_db.add_staff(person_id, full_name, position)

        info = f"Сотрудник добавлен.\nУспешно обработано фото: {ok}"
        if errors:
            info += "\n\nПропущены:\n" + "\n".join(errors)
        QtWidgets.QMessageBox.information(self, "Готово", info)
        self.accept()


#  Главное окно
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model, recognizer, face_db, att_db):
        super().__init__()
        self.model = model
        self.recognizer = recognizer
        self.face_db = face_db
        self.att_db = att_db
        # два независимых потока: вход и выход
        self.thread_in = None
        self.thread_out = None

        self.setWindowTitle("Система учёта посещаемости по лицам (2 камеры)")
        self.resize(1100, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Вертикальный сплиттер
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(splitter)

        # ── Верхняя часть: два видео рядом + управление (в своём контейнере)
        top_widget = QtWidgets.QWidget()
        top = QtWidgets.QVBoxLayout(top_widget)
        top.setContentsMargins(0, 0, 0, 0)
        videos = QtWidgets.QHBoxLayout()

        # Камера входа
        in_box = QtWidgets.QVBoxLayout()
        cap_in = QtWidgets.QLabel("🟢 КАМЕРА ВХОДА")
        cap_in.setAlignment(QtCore.Qt.AlignCenter)
        cap_in.setStyleSheet("font-weight:bold; color:#1a8a1a;")
        in_box.addWidget(cap_in)
        self.video_in = QtWidgets.QLabel("Выключена")
        self.video_in.setAlignment(QtCore.Qt.AlignCenter)
        self.video_in.setMinimumSize(360, 200)
        self.video_in.setStyleSheet("background:#162016; color:#7a7; border-radius:6px;")
        in_box.addWidget(self.video_in, 1)
        videos.addLayout(in_box, 1)

        # Камера выхода
        out_box = QtWidgets.QVBoxLayout()
        cap_out = QtWidgets.QLabel("🔴 КАМЕРА ВЫХОДА")
        cap_out.setAlignment(QtCore.Qt.AlignCenter)
        cap_out.setStyleSheet("font-weight:bold; color:#b22;")
        out_box.addWidget(cap_out)
        self.video_out = QtWidgets.QLabel("Выключена")
        self.video_out.setAlignment(QtCore.Qt.AlignCenter)
        self.video_out.setMinimumSize(360, 200)
        self.video_out.setStyleSheet("background:#201616; color:#a77; border-radius:6px;")
        out_box.addWidget(self.video_out, 1)
        videos.addLayout(out_box, 1)

        top.addLayout(videos, 1)

        # Панель управления
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Камера входа:"))
        self.cam_in_spin = QtWidgets.QSpinBox()
        self.cam_in_spin.setRange(0, 10)
        self.cam_in_spin.setValue(0)
        ctrl.addWidget(self.cam_in_spin)

        ctrl.addSpacing(16)
        ctrl.addWidget(QtWidgets.QLabel("Камера выхода:"))
        self.cam_out_spin = QtWidgets.QSpinBox()
        self.cam_out_spin.setRange(0, 10)
        self.cam_out_spin.setValue(1)
        ctrl.addWidget(self.cam_out_spin)

        ctrl.addSpacing(16)
        self.btn_start = QtWidgets.QPushButton("▶ Старт обеих")
        self.btn_start.clicked.connect(self.start_cameras)
        self.btn_stop = QtWidgets.QPushButton("■ Стоп")
        self.btn_stop.clicked.connect(self.stop_cameras)
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)
        top.addLayout(ctrl)

        # Камеры сверху — добавляем контейнер в сплиттер
        splitter.addWidget(top_widget)

        # ── Нижняя часть: вкладки БД (журнал входов/выходов и персонал) ──
        bottom_widget = QtWidgets.QWidget()
        bottom = QtWidgets.QVBoxLayout(bottom_widget)
        bottom.setContentsMargins(0, 0, 0, 0)
        tabs = QtWidgets.QTabWidget()

        # --- Вкладка: журнал ---
        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_tab)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Дата:"))
        self.date_edit = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        filter_row.addWidget(self.date_edit)
        self.chk_today = QtWidgets.QCheckBox("Только за дату")
        filter_row.addWidget(self.chk_today)
        btn_refresh = QtWidgets.QPushButton("Обновить")
        btn_refresh.clicked.connect(self.refresh_log)
        filter_row.addWidget(btn_refresh)
        btn_export = QtWidgets.QPushButton("Экспорт CSV")
        btn_export.clicked.connect(self.export_csv)
        filter_row.addWidget(btn_export)
        log_layout.addLayout(filter_row)

        self.log_table = QtWidgets.QTableWidget(0, 4)
        self.log_table.setHorizontalHeaderLabels(["ФИО", "Направление", "Время", "Сходство"])
        self.log_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.log_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        log_layout.addWidget(self.log_table)
        tabs.addTab(log_tab, "Журнал входов/выходов")

        # --- Вкладка: персонал ---
        staff_tab = QtWidgets.QWidget()
        staff_layout = QtWidgets.QVBoxLayout(staff_tab)

        staff_btns = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("➕ Добавить сотрудника")
        btn_add.clicked.connect(self.add_staff)
        btn_del = QtWidgets.QPushButton("🗑 Удалить выбранного")
        btn_del.clicked.connect(self.delete_staff)
        staff_btns.addWidget(btn_add)
        staff_btns.addWidget(btn_del)
        staff_btns.addStretch(1)
        staff_layout.addLayout(staff_btns)

        self.staff_table = QtWidgets.QTableWidget(0, 4)
        self.staff_table.setHorizontalHeaderLabels(["ID", "ФИО", "Должность", "Статус"])
        self.staff_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.staff_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.staff_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        staff_layout.addWidget(self.staff_table)
        tabs.addTab(staff_tab, "Персонал")

        bottom.addWidget(tabs)
        # Журнал/персонал снизу — добавляем в сплиттер
        splitter.addWidget(bottom_widget)

        # Начальные высоты блоков (камеры : журнал ≈ 3 : 2) и поведение при растягивании
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 340])
        splitter.setChildrenCollapsible(False)   # блоки нельзя схлопнуть в ноль
        self.splitter = splitter

        # Статус-бар
        self.statusBar().showMessage("Готово. База: "
                                     f"{len(self.face_db.embeddings)} чел.")

        # Автообновление журнала
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_log)
        self.timer.timeout.connect(self.refresh_staff)
        self.timer.start(3000)

        self.refresh_log()
        self.refresh_staff()

    # ── Камеры ──
    def start_cameras(self):
        cam_in = self.cam_in_spin.value()
        cam_out = self.cam_out_spin.value()
        if cam_in == cam_out:
            QtWidgets.QMessageBox.warning(
                self, "Камеры",
                "Камера входа и камера выхода не могут иметь одинаковый номер.")
            return

        if not (self.thread_in and self.thread_in.isRunning()):
            self.thread_in = VideoThread(
                self.model, self.recognizer, self.face_db, self.att_db,
                cam_index=cam_in, direction="IN")
            self.thread_in.frame_ready.connect(self.update_frame)
            self.thread_in.event_logged.connect(self.on_event)
            self.thread_in.status.connect(lambda m: self.statusBar().showMessage(m))
            self.thread_in.start()

        if not (self.thread_out and self.thread_out.isRunning()):
            self.thread_out = VideoThread(
                self.model, self.recognizer, self.face_db, self.att_db,
                cam_index=cam_out, direction="OUT")
            self.thread_out.frame_ready.connect(self.update_frame)
            self.thread_out.event_logged.connect(self.on_event)
            self.thread_out.status.connect(lambda m: self.statusBar().showMessage(m))
            self.thread_out.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cam_in_spin.setEnabled(False)
        self.cam_out_spin.setEnabled(False)

    def stop_cameras(self):
        for t in (self.thread_in, self.thread_out):
            if t:
                t.stop()
        self.thread_in = None
        self.thread_out = None
        for lbl, txt in ((self.video_in, "Выключена"), (self.video_out, "Выключена")):
            lbl.setText(txt)
            lbl.setPixmap(QtGui.QPixmap())
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cam_in_spin.setEnabled(True)
        self.cam_out_spin.setEnabled(True)

    @QtCore.pyqtSlot(str, np.ndarray)
    def update_frame(self, direction, frame):
        label = self.video_in if direction == "IN" else self.video_out
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            label.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    @QtCore.pyqtSlot(str, str, float)
    def on_event(self, person_id, direction, sim):
        name = self.att_db.get_staff_name(person_id)
        dir_txt = "ВХОД" if direction == "IN" else "ВЫХОД"
        self.statusBar().showMessage(f"{name}: {dir_txt} ({sim*100:.0f}%)")
        self.refresh_log()
        self.refresh_staff()

    # ── Журнал ──
    def refresh_log(self):
        date_filter = (self.date_edit.date().toString("yyyy-MM-dd")
                       if self.chk_today.isChecked() else None)
        rows = self.att_db.get_log(limit=500, date_filter=date_filter)
        self.log_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            direction = "🟢 Вход" if row["direction"] == "IN" else "🔴 Выход"
            sim = f"{(row['confidence'] or 0) * 100:.0f}%"
            values = [row["full_name"] or row["person_id"], direction,
                      row["timestamp"].replace("T", " "), sim]
            for c, v in enumerate(values):
                self.log_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))

    def export_csv(self):
        date_filter = (self.date_edit.date().toString("yyyy-MM-dd")
                       if self.chk_today.isChecked() else None)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить журнал", "attendance_log.csv", "CSV (*.csv)")
        if path:
            self.att_db.export_log_csv(path, date_filter)
            QtWidgets.QMessageBox.information(self, "Готово", f"Журнал сохранён:\n{path}")

    # ── Персонал ──
    def refresh_staff(self):
        staff = self.att_db.get_staff()
        inside = set(self.att_db.who_is_inside())
        self.staff_table.setRowCount(len(staff))
        for r, s in enumerate(staff):
            status = "В помещении" if s["person_id"] in inside else "Снаружи"
            values = [s["person_id"], s["full_name"], s["position"] or "", status]
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(v))
                if c == 3:
                    item.setForeground(QtGui.QColor("#1a8a1a") if s["person_id"] in inside
                                       else QtGui.QColor("#888"))
                self.staff_table.setItem(r, c, item)

    def add_staff(self):
        dlg = AddStaffDialog(self, self.recognizer, self.face_db, self.att_db)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.refresh_staff()
            self.statusBar().showMessage(
                f"Сотрудник добавлен. Всего в базе: {len(self.face_db.embeddings)} чел.")

    def delete_staff(self):
        row = self.staff_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Удаление", "Выберите сотрудника в таблице.")
            return
        person_id = self.staff_table.item(row, 0).text()
        name = self.staff_table.item(row, 1).text()
        ans = QtWidgets.QMessageBox.question(
            self, "Удаление",
            f"Удалить «{name}» ({person_id}) из базы?\n"
            "Записи журнала сохранятся, но распознаваться человек больше не будет.")
        if ans == QtWidgets.QMessageBox.Yes:
            self.att_db.remove_staff(person_id)
            self.face_db.remove(person_id)
            self.face_db.save()
            # удалить фото
            person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir, ignore_errors=True)
            self.refresh_staff()

    def closeEvent(self, event):
        self.stop_cameras()
        event.accept()


# ════════════════════════════════════════════════════════════════
def main():
    # страховка: к этому моменту окружение могло быть испорчено cv2/торчем —
    # возвращаем пути к плагинам PyQt5 ещё раз (вызов идемпотентный).
    qt_compat.restore_qt_paths()
    app = QtWidgets.QApplication(sys.argv)

    # Сплэш с прогрессом загрузки тяжёлых моделей
    splash = QtWidgets.QSplashScreen()
    splash.showMessage("Загрузка моделей, подождите…",
                       QtCore.Qt.AlignCenter, QtCore.Qt.white)
    splash.show()
    app.processEvents()

    model = load_model(MODEL_PATH)
    recognizer = FaceRecognizer(ctx_id=RECOGNITION_CTX_ID)

    face_db = FaceDatabase(FACE_DB_PATH)
    if not face_db.load():
        print("[INFO] Кэш базы лиц не найден — строю из папки known_faces…")
        face_db.build_from_folder(KNOWN_FACES_DIR, recognizer)
        face_db.save()

    att_db = AttendanceDB()

    # проверка, что для каждого человека из эмбеддингов есть запись в SQLite
    existing = {s["person_id"] for s in att_db.get_staff()}
    for pid in face_db.embeddings:
        if pid not in existing:
            att_db.add_staff(pid, pid, "")

    win = MainWindow(model, recognizer, face_db, att_db)
    splash.finish(win)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
