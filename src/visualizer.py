import cv2
import numpy as np
from abc import ABC, abstractmethod
from src.detector import DetectionResult
from typing import Tuple, List, Optional

class Visualizer(ABC):
    """Базовый класс для отрисовки данных на кадре"""
    @abstractmethod
    def draw(self, frame: np.ndarray, det: DetectionResult) -> np.ndarray: ...

class VisualizerCenterPoint(Visualizer):
    """Рисует точку в центре обнаруженного объекта"""
    def __init__(self, color: Tuple[int, int, int] = (0, 255, 0), radius: int = 8, thickness: int = -1):
        self.color = color
        self.radius = radius
        self.thickness = thickness

    def draw(self, frame: np.ndarray, det: DetectionResult) -> np.ndarray:
        if det.center is not None:
            cv2.circle(frame, det.center, self.radius, self.color, self.thickness)
            # Опционально: подпись координат
            cv2.putText(frame, f"({det.center[0]},{det.center[1]})", 
                        (det.center[0] + 10, det.center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        return frame
    
    
class VisualizerContour(Visualizer):
    """Обводит все контуры, найденные в маске детекции"""
    def __init__(self, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        self.color = color
        self.thickness = thickness

    def draw(self, frame: np.ndarray, det: DetectionResult) -> np.ndarray:
        if det.mask is not None:
            # Ищем все внешние контуры
            contours, _ = cv2.findContours(det.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # contourIdx=-1 означает отрисовку всех контуров из списка
                cv2.drawContours(frame, contours, -1, self.color, self.thickness)
        elif det.bbox is not None:
            # Fallback: если маски нет, рисуем ограничивающий прямоугольник
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, self.thickness)
        return frame
    

class VisualizerHeatmap(Visualizer):
    """
    Визуализатор: только отрисовка тепловой карты на кадре.
    Получает данные из MetricHeatmap через ссылку.
    """
    def __init__(self, metric_heatmap,
                 colormap: int = cv2.COLORMAP_INFERNO,
                 opacity: float = 0.65):
        self.metric = metric_heatmap
        self.colormap = colormap
        self.opacity = opacity

    def draw(self, frame: np.ndarray, det: DetectionResult) -> np.ndarray:
        hm_norm = self.metric.get_heatmap_normalized()
        if hm_norm is None:
            return frame
        
        hm_colored = cv2.applyColorMap(hm_norm, self.colormap)
        return cv2.addWeighted(frame, 1.0 - self.opacity, hm_colored, self.opacity, 0)
