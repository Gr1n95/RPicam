"""
Модуль детекции и распознавания лиц.
Основан на исходном main.py (YOLO + InsightFace ArcFace).
"""
import os
import pickle
import warnings
import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# insightface до сих пор вызывает skimage.transform.SimilarityTransform.estimate()
# (insightface/utils/face_align.py), который в scikit-image 0.26 помечен как
# устаревший. Предупреждение печатается на каждое распознанное лицо и засоряет
# лог, хотя ни наша ошибка, ни ошибка insightface это не является.
# Глушим точечно — только FutureWarning из этого модуля и только с этим текстом.
# Полностью проблема уйдёт, когда insightface перейдёт на SimilarityTransform
# .from_estimate(); тогда фильтр можно удалить.
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)

# ─── Настройки распознавания ───────────────────────────────────
# Базовая папка = папка, где лежит этот файл. Все пути строим от неё,
# чтобы программа работала из любой рабочей директории.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
CONFIDENCE = 0.4
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")  # known_faces/<person_id>/*.jpg
FACE_DB_PATH = os.path.join(BASE_DIR, "face_db.pkl")     # кэш эмбеддингов
SIMILARITY_THRESHOLD = 0.45         # порог косинусного сходства (0..1)
RECOGNITION_CTX_ID = -1             # 0 = GPU (CUDA), -1 = CPU
CROP_PADDING = 0.25                 # запас при вырезании лица (доля от bbox)


def resolve_onnx_runtime(ctx_id: int = RECOGNITION_CTX_ID):
    """
    Выбирает Execution Providers для ONNX Runtime — только реально доступные.

    InsightFace передаёт список providers в onnxruntime.InferenceSession как
    есть. Если попросить CUDAExecutionProvider, а установлен обычный
    onnxruntime (без CUDA), ORT печатает предупреждение и сам откатывается
    на CPU:

        UserWarning: Specified provider 'CUDAExecutionProvider' is not in
        available provider names. Available providers:
        'AzureExecutionProvider, CPUExecutionProvider'

    Поэтому сначала спрашиваем у ORT, что он умеет, и просим только это.

    Возвращает (providers, effective_ctx_id). Если GPU запрошен, но недоступен,
    effective_ctx_id понижается до -1, чтобы InsightFace не пытался
    инициализировать CUDA тем ctx_id, который мы всё равно не используем.
    """
    cpu, cuda = "CPUExecutionProvider", "CUDAExecutionProvider"

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception as exc:
        print(f"[WARN] onnxruntime не отвечает ({exc}) — работаю на CPU.")
        return [cpu], -1

    if ctx_id < 0:
        # CPU-режим: CUDA не запрашиваем вовсе — иначе ORT и печатает тот самый
        # UserWarning, хотя на поведение он не влияет.
        return [cpu], -1

    if cuda in available:
        return [cuda, cpu], ctx_id

    print(f"[WARN] Запрошен GPU (RECOGNITION_CTX_ID={ctx_id}), но установленный "
          f"onnxruntime CUDAExecutionProvider не содержит.")
    print(f"       Доступные провайдеры: {', '.join(available) or '(нет)'}")
    print("       Для GPU нужны видеокарта NVIDIA и пакет onnxruntime-gpu:")
    print("         pip uninstall -y onnxruntime && pip install onnxruntime-gpu")
    print("       Продолжаю на CPU.")
    return [cpu], -1


def load_model(path: str = MODEL_PATH) -> YOLO:
    print(f"[INFO] Загрузка модели детекции: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл модели не найден: {path}\n"
            f"Положите файл best.pt в папку: {BASE_DIR}"
        )
    return YOLO(path)


class FaceRecognizer:
    """Обёртка над InsightFace для извлечения 512-мерных эмбеддингов ArcFace."""

    def __init__(self, ctx_id: int = RECOGNITION_CTX_ID):
        print("[INFO] Загрузка модели распознавания (InsightFace buffalo_l)...")
        providers, ctx_id = resolve_onnx_runtime(ctx_id)
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        self.providers = providers
        print(f"[OK] Модель распознавания загружена "
              f"(провайдеры: {', '.join(providers)}).")

    def embed(self, face_image: np.ndarray):
        """Возвращает нормализованный эмбеддинг для изображения с лицом, иначе None."""
        if face_image is None or face_image.size == 0:
            return None
        faces = self.app.get(face_image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding


class FaceDatabase:
    """Хранилище эмбеддингов известных людей."""

    def __init__(self, db_path: str = FACE_DB_PATH):
        self.db_path = db_path
        self.embeddings = {}            # person_id -> [emb1, emb2, ...]
        self._matrix = None             # матрица всех эмбеддингов (N, 512)
        self._labels = None             # список меток размером N

    def add(self, person_id: str, embedding: np.ndarray):
        self.embeddings.setdefault(person_id, []).append(embedding)
        self._rebuild_matrix()

    def remove(self, person_id: str):
        """Удаляет все эмбеддинги человека из базы."""
        if person_id in self.embeddings:
            del self.embeddings[person_id]
            self._rebuild_matrix()

    def build_from_folder(self, folder: str, recognizer: FaceRecognizer):
        """Сканирует known_faces/<person_id>/*.jpg и строит базу."""
        if not os.path.isdir(folder):
            print(f"[ОШИБКА] Папка {folder} не существует.")
            return
        for person_id in sorted(os.listdir(folder)):
            person_dir = os.path.join(folder, person_id)
            if not os.path.isdir(person_dir):
                continue
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  [WARN] Не прочитать: {img_path}")
                    continue
                emb = recognizer.embed(img)
                if emb is None:
                    print(f"  [WARN] Лицо не найдено: {img_path}")
                    continue
                self.embeddings.setdefault(person_id, []).append(emb)
                print(f"  [OK] {person_id} / {img_name}")
        self._rebuild_matrix()
        print(f"[INFO] База готова: {len(self.embeddings)} человек, "
              f"{0 if self._matrix is None else len(self._matrix)} эмбеддингов.")

    def add_person_from_images(self, person_id: str, image_paths, recognizer: FaceRecognizer):
        """Добавляет человека из списка путей к фото. Возвращает (кол-во удачных, список ошибок)."""
        ok, errors = 0, []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                errors.append(f"Не удалось прочитать: {os.path.basename(img_path)}")
                continue
            emb = recognizer.embed(img)
            if emb is None:
                errors.append(f"Лицо не найдено: {os.path.basename(img_path)}")
                continue
            self.embeddings.setdefault(person_id, []).append(emb)
            ok += 1
        if ok:
            self._rebuild_matrix()
        return ok, errors

    def _rebuild_matrix(self):
        labels, vectors = [], []
        for pid, embs in self.embeddings.items():
            for e in embs:
                labels.append(pid)
                vectors.append(e)
        self._labels = labels
        self._matrix = np.vstack(vectors) if vectors else None

    def search(self, query_emb: np.ndarray, threshold: float = SIMILARITY_THRESHOLD):
        """Возвращает (person_id или None, similarity)."""
        if self._matrix is None or query_emb is None:
            return None, 0.0
        sims = self._matrix @ query_emb
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        return (self._labels[idx] if best >= threshold else None), best

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"[INFO] База сохранена в {self.db_path}")

    def load(self) -> bool:
        if not os.path.exists(self.db_path):
            return False
        with open(self.db_path, "rb") as f:
            self.embeddings = pickle.load(f)
        self._rebuild_matrix()
        print(f"[INFO] База загружена из {self.db_path}: {len(self.embeddings)} человек.")
        return True


# Отрисовка текста
# cv2.putText не умеет рисовать кириллицу (показывает '?'), поэтому
# подписи к рамкам рисуем шрифтом TrueType через PIL
_FONT_CANDIDATES = [
    os.path.join(BASE_DIR, "DejaVuSans.ttf"),
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_font_cache = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Возвращает (и кэширует) шрифт с поддержкой кириллицы нужного размера."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        # запасной вариант — встроенный шрифт PIL (кириллицу обычно тянет хуже,
        # но программа хотя бы не упадёт)
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_label_ru(frame: np.ndarray, bbox, label: str, color_bgr, font_size: int = 22):
    """
    Рисует прямоугольник вокруг лица и подпись с поддержкой русского языка.
    color_bgr — цвет в формате OpenCV (B, G, R). Возвращает новый кадр (BGR).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)

    # рамка
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

    # размеры текста
    try:
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
    except Exception:
        tw, th = draw.textsize(label, font=font)

    pad = 4
    bg_y2 = y1
    bg_y1 = y1 - th - 2 * pad
    if bg_y1 < 0:                       # если не помещается сверху — рисуем под рамкой
        bg_y1 = y1
        bg_y2 = y1 + th + 2 * pad

    # подложка под текст
    draw.rectangle([x1, bg_y1, x1 + tw + 2 * pad, bg_y2], fill=color_rgb)
    # сам текст (чёрный для контраста)
    draw.text((x1 + pad, bg_y1 + pad), label, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def crop_face(frame: np.ndarray, bbox, padding: float = CROP_PADDING) -> np.ndarray:
    """Вырезает регион лица с запасом, чтобы InsightFace надёжнее нашёл лицо."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * padding), int(bh * padding)
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)
    return frame[y1:y2, x1:x2]
