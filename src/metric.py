import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
from src.detector import DetectionResult

@dataclass
class FrameRecord:
    """Запись о кадре для истории трекера"""
    frame_idx: int
    center: Optional[Tuple[int, int]]
    timestamp_s: float
    detected: bool

class Metric(ABC):
    """Базовый класс для отдельной метрики"""
    @abstractmethod
    def update(self, det: DetectionResult, record: FrameRecord, history: Deque[FrameRecord]) -> Dict[str, float]:
        """
        Вызывается каждый кадр для расчёта метрики.
        :param det: Полный результат детекции (включая mask!)
        :param record: Запись о кадре (центр, время)
        :param history: История предыдущих кадров
        :return: Словарь {имя_метрики: значение}
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Сброс внутреннего состояния метрики"""
        ...


class SpeedMetric(Metric):
    """Расчёт мгновенной скорости (пиксели/секунду)"""
    def __init__(self, smoothing_frames: int = 3):
        self.smoothing_frames = smoothing_frames
        self._last_speed: float = 0.0

    def update(self, det: DetectionResult, record: FrameRecord, history: Deque[FrameRecord]) -> Dict[str, float]:
        if not record.detected or len(history) < 2:
            self._last_speed = 0.0
            return {"speed_px_s": 0.0}
        
        recent = list(history)[-self.smoothing_frames:]
        if len(recent) < 2:
            self._last_speed = 0.0
            return {"speed_px_s": 0.0}
        
        first, last = recent[0], recent[-1]
        if first.center and last.center:
            dx = last.center[0] - first.center[0]
            dy = last.center[1] - first.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            dt = last.timestamp_s - first.timestamp_s
            if dt > 0:
                self._last_speed = dist / dt
            else:
                self._last_speed = 0.0
        else:
            self._last_speed = 0.0
        
        return {"speed_px_s": float(self._last_speed)}

    def reset(self) -> None:
        self._last_speed = 0.0


class DistanceMetric(Metric):
    """Накопление пройденной дистанции (пиксели)"""
    def __init__(self):
        self._total_distance: float = 0.0
        self._last_position: Optional[Tuple[int, int]] = None

    def update(self, det: DetectionResult, record: FrameRecord, history: Deque[FrameRecord]) -> Dict[str, float]:
        if not record.detected:
            return {"distance_px": float(self._total_distance)}
        
        if self._last_position and record.center:
            dx = record.center[0] - self._last_position[0]
            dy = record.center[1] - self._last_position[1]
            self._total_distance += np.sqrt(dx**2 + dy**2)
        
        self._last_position = record.center
        return {"distance_px": float(self._total_distance)}

    def reset(self) -> None:
        self._total_distance = 0.0
        self._last_position = None


class PauseMetric(Metric):
    """Детекция пауз (время без движения)"""
    def __init__(self, speed_threshold: float = 2.0, min_pause_s: float = 0.5):
        self.speed_threshold = speed_threshold
        self.min_pause_s = min_pause_s
        self._pause_start: Optional[float] = None
        self._current_pause_duration: float = 0.0
        self._total_pause_time: float = 0.0
        self._pause_count: int = 0

    def update(self, det: DetectionResult, record: FrameRecord, history: Deque[FrameRecord]) -> Dict[str, float]:
        current_speed = 0.0
        if record.detected and len(history) >= 2:
            last = list(history)[-1]
            prev = list(history)[-2]
            if last.center and prev.center and last.timestamp_s > prev.timestamp_s:
                dx = last.center[0] - prev.center[0]
                dy = last.center[1] - prev.center[1]
                dt = last.timestamp_s - prev.timestamp_s
                current_speed = np.sqrt(dx**2 + dy**2) / dt
        
        is_paused = current_speed < self.speed_threshold
        
        if is_paused:
            if self._pause_start is None:
                self._pause_start = record.timestamp_s
            self._current_pause_duration = record.timestamp_s - self._pause_start
        else:
            if self._pause_start is not None:
                if self._current_pause_duration >= self.min_pause_s:
                    self._total_pause_time += self._current_pause_duration
                    self._pause_count += 1
            self._pause_start = None
            self._current_pause_duration = 0.0
        
        return {
            "is_paused": 1.0 if is_paused else 0.0,
            "pause_duration_s": float(self._current_pause_duration),
            "total_pause_time_s": float(self._total_pause_time),
            "pause_count": float(self._pause_count)
        }

    def reset(self) -> None:
        self._pause_start = None
        self._current_pause_duration = 0.0
        self._total_pause_time = 0.0
        self._pause_count = 0


class MetricHeatmap(Metric):
    """
    Метрика: накопление тепловой карты посещений по МАСКЕ детекции.
    """
    def __init__(self, decay: float = 0.97, blur_kernel: int = 25):
        self.decay = decay
        self.blur_kernel = blur_kernel | 1
        self.heatmap: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

    def update(self, det: DetectionResult, record: FrameRecord, history: Deque[FrameRecord]) -> Dict[str, float]:
        # 🔄 Инициализация буфера по первому кадру
        if self.heatmap is None and det.mask is not None:
            h, w = det.mask.shape
            self.heatmap = np.zeros((h, w), dtype=np.float32)
            self._frame_shape = (h, w)
        
        if self.heatmap is None:
            return {"heatmap_ready": 0.0}
        
        # 🔥 Накопление "тепла" по МАСКЕ (не по центру!)
        if det.mask is not None:
            # Добавляем тепло только там, где маска > 0
            self.heatmap[det.mask > 0] += 1.0
        
        # 🌫 Затухание + размытие
        self.heatmap *= self.decay
        self.heatmap = cv2.GaussianBlur(self.heatmap, (self.blur_kernel, self.blur_kernel), 0)
        
        return {"heatmap_ready": 1.0}

    def reset(self) -> None:
        self.heatmap = None
        self._frame_shape = None

    def get_heatmap(self) -> Optional[np.ndarray]:
        """Возвращает текущий буфер тепловой карты (float32)"""
        return self.heatmap

    def get_heatmap_normalized(self) -> Optional[np.ndarray]:
        """Возвращает тепловую карту в формате uint8 [0-255] для отрисовки"""
        if self.heatmap is None:
            return None
        
        hm_max = self.heatmap.max()
        if hm_max < 1e-5:
            return np.zeros_like(self.heatmap, dtype=np.uint8)
        
        # Логарифмическая нормализация
        log_heatmap = np.log1p(self.heatmap)
        log_max = np.log1p(hm_max)
        return (log_heatmap / log_max * 255).astype(np.uint8)


class MetricsTracker:
    """Контейнер для управления метриками и хранения истории кадров"""
    def __init__(self, metrics: List[Metric], history_size: int = 30, fps: float = 30.0):
        self.metrics = metrics
        self.history_size = history_size
        self.fps = fps
        self.history: Deque[FrameRecord] = deque(maxlen=history_size)
        self.frame_idx = 0

    def update(self, det: DetectionResult) -> Dict[str, float]:
        """
        Обновляет все метрики на основе текущего DetectionResult.
        Передаёт ПОЛНЫЙ det в каждую метрику (включая mask!).
        """
        current_time = self.frame_idx / self.fps
        
        # Создание записи о кадре (для истории позиций)
        record = FrameRecord(
            frame_idx=self.frame_idx,
            center=det.center,
            timestamp_s=current_time,
            detected=det.center is not None
        )
        
        # Сохранение в историю
        self.history.append(record)
        
        # 🔄 Обновление всех метрик с передачей полного det
        all_metrics: Dict[str, float] = {}
        for metric in self.metrics:
            metric_data = metric.update(det, record, self.history)  # 👈 det передаётся целиком!
            all_metrics.update(metric_data)
        
        self.frame_idx += 1
        return all_metrics

    def reset(self) -> None:
        self.history.clear()
        self.frame_idx = 0
        for metric in self.metrics:
            metric.reset()

    def get_summary(self) -> Dict[str, float]:
        """Возвращает сводку по всем метрикам"""
        summary = {
            "total_frames": float(self.frame_idx),
            "total_duration_s": float(self.frame_idx / self.fps)
        }
        for metric in self.metrics:
            if hasattr(metric, '_total_distance'):
                summary["total_distance_px"] = float(metric._total_distance)
            if hasattr(metric, '_total_pause_time'):
                summary["total_pause_time_s"] = float(metric._total_pause_time)
            if hasattr(metric, '_pause_count'):
                summary["pause_count"] = float(metric._pause_count)
        return summary