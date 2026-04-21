import cv2
import numpy as np
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from src.source import Source

class CalibrationContext:
    """Гибкий контейнер для данных калибровки с автоматическим восстановлением типов при загрузке."""
    def __init__(self, schema_version: int = 1):
        self._data: Dict[str, Any] = {"schema_version": schema_version}

    # 🔹 Динамический доступ (атрибуты)
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # 🔹 Доступ в стиле словаря
    def __contains__(self, key: str) -> bool: return key in self._data
    def __getitem__(self, key: str) -> Any: return self._data[key]
    def __setitem__(self, key: str, value: Any) -> None: self._data[key] = value
    def get(self, key: str, default: Any = None) -> Any: return self._data.get(key, default)

    # 🔹 Сериализация в JSON-совместимые типы
    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        if isinstance(obj, dict): return {k: CalibrationContext._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, set)): return [CalibrationContext._to_serializable(i) for i in obj]
        return obj

    # 🔹 Сохранение (атомарное)
    def save(self, filepath: str) -> None:
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if dir_path: os.makedirs(dir_path, exist_ok=True)
        
        temp_path = filepath + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_serializable(self._data), f, indent=4, ensure_ascii=False)
        os.replace(temp_path, filepath)
        print(f"💾 CalibrationContext saved: {filepath}")

    # 🔹 Загрузка с восстановлением типов
    def load(self, filepath: str) -> 'CalibrationContext':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self._restore_types(data)
        self._data.update(data)
        print(f"📂 CalibrationContext loaded: {filepath}")
        return self

    @staticmethod
    def _restore_types(data: Dict[str, Any]) -> None:
        """Восстанавливает numpy-массивы и кортежи после десериализации JSON"""
        # Координаты и размеры → tuple
        for key in ['roi', 'init_position', 'sample_size']:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
                
        # Цвета → np.ndarray (float64, как возвращает np.median)
        for key in ['sample_color', 'background_color']:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key], dtype=np.float64)


class _RectSelector:
    """Внутренний помощник для интерактивного выделения прямоугольника"""
    def __init__(self, window_name: str, delay_ms: int = 20):
        self.window_name = window_name
        self.delay_ms = delay_ms
        self.start_pt: Optional[Tuple[int, int]] = None
        self.end_pt: Optional[Tuple[int, int]] = None
        self.drawing = False
        self.rect: Optional[Tuple[int, int, int, int]] = None

    def _callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pt = (x, y)
            self.end_pt = (x, y)
            self.drawing = True
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_pt = (x, y)
            self.drawing = False
            x1, y1 = self.start_pt
            x2, y2 = self.end_pt
            self.rect = (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))

    def run(self, frame: np.ndarray, prompt: str) -> Optional[Tuple[int, int, int, int]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._callback)
        text_frame = frame.copy()
        cv2.putText(text_frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        while True:
            disp = text_frame.copy()
            if self.start_pt and self.end_pt:
                cv2.rectangle(disp, self.start_pt, self.end_pt, (0, 255, 0), 2)
            cv2.imshow(self.window_name, disp)
            
            key = cv2.waitKey(self.delay_ms) & 0xFF
            if key in (13, 32): return self.rect
            elif key in (27, ord('q')): return None
            elif key == ord('r'): self.rect = None; self.start_pt = None; self.end_pt = None


class Setup(ABC):
    """Базовый класс для этапа калибровки"""
    def __init__(self, window_name: str = "Calibration", delay_ms: int = 20):
        self.window_name = window_name
        self.delay_ms = delay_ms

    @abstractmethod
    def run(self, frame: np.ndarray, context: CalibrationContext) -> np.ndarray: ...


class SetupROI(Setup):
    """Этап 1: Выделение области интереса (ROI)"""
    def run(self, frame: np.ndarray, context: CalibrationContext) -> np.ndarray:
        selector = _RectSelector(self.window_name, self.delay_ms)
        roi = selector.run(frame, "1️⃣ Select labirint area (Enter: Ok, Esc: Skip)")
        if roi:
            context.roi = roi
            x, y, w, h = roi
            return frame[y:y+h, x:x+w].copy()
        return frame


class SetupSample(Setup):
    """Этап 2: Выделение образца, расчёт цвета (медиана) и размера"""
    def run(self, frame: np.ndarray, context: CalibrationContext) -> np.ndarray:
        selector = _RectSelector(self.window_name, self.delay_ms)
        rect = selector.run(frame, "2️⃣ Select object area (Enter: Ok, Esc: Skip)")
        if rect:
            sx, sy, sw, sh = rect
            context.init_position = (sx + sw // 2, sy + sh // 2)
            context.sample_size = (sw, sh)
            
            sample_region = frame[sy:sy+sh, sx:sx+sw]
            context.sample_color = np.median(sample_region, axis=(0, 1))
            
            h, w = frame.shape[:2]
            bg_mask = np.zeros((h, w), dtype=bool)
            bg_mask[sy:sy+sh, sx:sx+sw] = True
            bg_pixels = frame[~bg_mask]
            if bg_pixels.size > 0:
                context.background_color = np.median(bg_pixels, axis=0)
        return frame


class SetupSizeCalibration(Setup):
    """Этап калибровки: выделение области известного размера для расчёта px/mm"""
    def __init__(self, width_mm: float, height_mm: float, window_name: str = "Calibration", delay_ms: int = 20):
        if width_mm <= 0 or height_mm <= 0:
            raise ValueError("Physical dimensions must be strictly positive.")
        self.width_mm = width_mm
        self.height_mm = height_mm
        super().__init__(window_name, delay_ms)

    def run(self, frame: np.ndarray, context: CalibrationContext) -> np.ndarray:
        prompt = f"Select {self.width_mm}x{self.height_mm} mm etalon (Enter: OK, Esc: Skip)"
        selector = _RectSelector(self.window_name, self.delay_ms)
        rect = selector.run(frame, prompt)
        
        if rect and rect[2] > 0 and rect[3] > 0:  # x, y, w, h
            _, _, w_px, h_px = rect
            px_per_mm_x = w_px / self.width_mm
            px_per_mm_y = h_px / self.height_mm

            # Записываем в контекст
            context.px_per_mm = (px_per_mm_x, px_per_mm_y)
            context.scale_mm = (self.width_mm, self.height_mm)
            context.scale_px = (w_px, h_px)
            context.is_scaled = True
            
        return frame


def run_setups(source: Source, setups: List[Setup]) -> CalibrationContext:
    """Запускает цепочку этапов настройки. Возвращает заполненный CalibrationContext."""
    if not source.open():
        raise RuntimeError("❌ Не удалось открыть источник для калибровки")

    try:
        ret, frame = source.read()
        if not ret or frame is None:
            raise RuntimeError("❌ Не удалось прочитать первый кадр")

        context = CalibrationContext()
        current_frame = frame.copy()

        for setup in setups:
            current_frame = setup.run(current_frame, context)

        return context
    finally:
        source.release()
        cv2.destroyAllWindows()