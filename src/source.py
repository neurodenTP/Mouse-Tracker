import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class Source(ABC):
    """Базовый класс для получения и предобработки кадров"""
    @abstractmethod
    def open(self) -> bool: ...
    
    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]: ...
    
    @abstractmethod
    def release(self) -> None: ...
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Обрезка ROI и базовая подготовка кадра"""
        if hasattr(self, 'roi') and self.roi:
            x, y, w, h = self.roi
            h_max, w_max = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, w_max - x), min(h, h_max - y)
            frame = frame[y:y+h, x:x+w]
        return frame


class SourceCamera(Source):
    """Источник: видео с камеры устройства"""
    def __init__(self, device_id: int = 0, roi: Optional[Tuple[int, int, int, int]] = None, fps: float = 30.0):
        self.device_id = device_id
        self.roi = roi
        self.cap: Optional[cv2.VideoCapture] = None
        self._user_fps = fps  # Фоллбэк, если камера не отдаёт FPS

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        return self.cap.isOpened()

    @property
    def fps(self) -> float:
        """Возвращает реальный FPS камеры или заданный пользователем"""
        if self.cap and self.cap.isOpened():
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if 0 < cap_fps < 1000:
                return cap_fps
        return self._user_fps

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.cap is None: return False, np.array([])
        return self.cap.read()

    def release(self) -> None:
        if self.cap: self.cap.release()


class SourceFile(Source):
    """Источник: видеофайл с диска"""
    def __init__(self, path: str, roi: Optional[Tuple[int, int, int, int]] = None, fps: float = 0.0):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self.path = path
        self.roi = roi
        self.cap: Optional[cv2.VideoCapture] = None
        self._user_fps = fps  # ✅ Добавлен параметр fps (0.0 = использовать из файла)

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        return self.cap.isOpened()

    @property
    def fps(self) -> float:
        """Возвращает FPS из файла или заданный пользователем"""
        if self._user_fps > 0:
            return self._user_fps  # Приоритет у явно заданного FPS
        if self.cap and self.cap.isOpened():
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if 0 < cap_fps < 1000:
                return cap_fps
        return 30.0  # Дефолтное значение

    @property
    def frame_count(self) -> int:
        """Возвращает общее количество кадров"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap and self.cap.isOpened() else 0

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.cap is None:
            return False, np.array([])
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()