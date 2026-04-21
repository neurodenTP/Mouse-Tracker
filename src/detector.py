import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Union, List, Any
from abc import ABC, abstractmethod

@dataclass
class DetectionResult:
    center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    mask: Optional[np.ndarray] = None                # uint8, 255 = объект
    confidence: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult: ...
    def auto_calibrate_color(self, sample_bgr: np.ndarray, background_bgr: Optional[np.ndarray] = None) -> None: ...
    def auto_calibrate_size(self, sample_size: Union[Tuple[int, int], np.ndarray, List[int]]) -> None: ...
    def set_initial_position(self, pos: Tuple[int, int]) -> None: ...


class DetectorStub(Detector):
    def __init__(self): self._initial_position: Optional[Tuple[int, int]] = None
    def set_initial_position(self, pos: Tuple[int, int]) -> None: self._initial_position = pos
    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        cx, cy = self._initial_position if self._initial_position else (w // 2, h // 2)
        cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
        size, half = 50, 25
        x, y = max(0, cx - half), max(0, cy - half)
        w_bbox, h_bbox = min(size, w - x), min(size, h - y)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), half, 255, -1)
        return DetectionResult(center=(cx, cy), bbox=(x, y, w_bbox, h_bbox), mask=mask, confidence=1.0, metrics={"is_stub": 1.0})


class DetectorColorThreshold(Detector):
    """Детектор на основе порогового фильтра в HSV с морфологией и слиянием контуров"""
    
    def __init__(self, 
                 # HSV параметры
                 h_mean: int = 0, h_delta: int = 15,
                 s_mean: int = 0, s_delta: int = 40,
                 v_mean: int = 0, v_delta: int = 50,
                 # Размер объекта
                 size_mean: float = 1000.0,
                 size_ratio: float = 2.0,
                 # Морфология
                 morph_kernel: int = 7,
                 morph_close_iterations: int = 2,
                 # Слияние контуров
                 merge_gap_px: float = 0.0,          # Явный порог в пикселях
                 merge_by_size: bool = True,         # Авто-расчёт порога из size_mean
                 merge_size_factor: float = 0.5      # merge_gap = sqrt(size_mean) * factor
                 ):
        # HSV
        self.h_mean, self.h_delta = h_mean, h_delta
        self.s_mean, self.s_delta = s_mean, s_delta
        self.v_mean, self.v_delta = v_mean, v_delta
        
        # Размер
        self.size_mean = size_mean
        self.size_ratio = size_ratio
        
        # Морфология
        self.morph_kernel = morph_kernel | 1  # Нечётное
        self.morph_close_iterations = morph_close_iterations
        
        # Слияние
        self.merge_gap_px = merge_gap_px
        self.merge_by_size = merge_by_size
        self.merge_size_factor = merge_size_factor
        self._merge_gap_computed: float = merge_gap_px
        
        # Внутреннее состояние
        self.lower_hsv: np.ndarray = np.zeros(3, dtype=np.uint8)
        self.upper_hsv: np.ndarray = np.zeros(3, dtype=np.uint8)
        self._min_area: int = 1
        self._max_area: int = 50000
        self._initial_position: Optional[Tuple[int, int]] = None
        
        self._update_hsv_bounds()
        self._update_size_bounds()

    def _update_hsv_bounds(self) -> None:
        h, s, v = int(self.h_mean), int(self.s_mean), int(self.v_mean)
        dh, ds, dv = int(self.h_delta), int(self.s_delta), int(self.v_delta)
        self.lower_hsv[:] = [max(0, h - dh), max(0, s - ds), max(0, v - dv)]
        self.upper_hsv[:] = [min(179, h + dh), min(255, s + ds), min(255, v + dv)]

    def _update_size_bounds(self) -> None:
        self._min_area = int(max(1, self.size_mean / self.size_ratio))
        self._max_area = int(self.size_mean * self.size_ratio)
        
        # 🔄 Авто-расчёт порога слияния из размера мыши
        if self.merge_by_size and self.size_mean > 0:
            # sqrt(area) ≈ характерный линейный размер объекта
            self._merge_gap_computed = np.sqrt(self.size_mean) * self.merge_size_factor
        else:
            self._merge_gap_computed = self.merge_gap_px

    def auto_calibrate_color(self, sample_bgr: Any, background_bgr: Optional[Any] = None) -> None:
        sample_arr = np.asarray(sample_bgr)
        sample_uint8 = sample_arr.astype(np.uint8)
        h, s, v = cv2.cvtColor(np.array([[sample_uint8]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        self.h_mean, self.s_mean, self.v_mean = h, s, v
        self._update_hsv_bounds()

    def auto_calibrate_size(self, sample_size: Any) -> None:
        size_arr = np.asarray(sample_size)
        w, h = int(size_arr[0]), int(size_arr[1])
        self.size_mean = w * h
        self._update_size_bounds()  # 👈 Пересчитывает _merge_gap_computed

    def set_initial_position(self, pos: Tuple[int, int]) -> None:
        self._initial_position = pos

    def _merge_nearby_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Объединяет контуры, если расстояние между центрами < порога"""
        gap = self._merge_gap_computed
        if gap <= 0 or len(contours) < 2:
            return contours
        
        # Вычисляем центры масс для всех контуров
        centers = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
            else:
                centers.append(None)
        
        merged = []
        used = [False] * len(contours)
        
        for i, c1 in enumerate(contours):
            if used[i]:
                continue
            
            group = [c1]
            used[i] = True
            cx1, cy1 = centers[i]
            
            if cx1 is None:
                merged.append(c1)
                continue
            
            for j, c2 in enumerate(contours):
                if i == j or used[j]:
                    continue
                
                cx2, cy2 = centers[j]
                if cx2 is None:
                    continue
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < gap:
                    group.append(c2)
                    used[j] = True
            
            # Сливаем группу через выпуклую оболочку
            if len(group) > 1:
                all_pts = np.vstack([c.reshape(-1, 2) for c in group])
                merged.append(cv2.convexHull(all_pts))
            else:
                merged.append(group[0])
        
        return merged

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # 🔧 Морфологическая очистка
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iterations)
        
        # 🔍 Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 📏 Фильтрация по площади
        valid_contours = [
            c for c in contours 
            if self._min_area <= cv2.contourArea(c) <= self._max_area
        ]
        
        # 🔗 Слияние близких контуров
        if self._merge_gap_computed > 0:
            valid_contours = self._merge_nearby_contours(valid_contours)
        
        if not valid_contours:
            return DetectionResult()
            
        # 🎯 Выбор крупнейшего объекта
        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback на boundingRect если моменты не посчитались
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
        
        # Чистая маска
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        
        # Метрики
        bbox_area = w * h
        obj_area = cv2.contourArea(largest)
        confidence = float(obj_area / bbox_area) if bbox_area > 0 else 0.0
        
        return DetectionResult(
            center=(cx, cy),
            bbox=(x, y, w, h),
            mask=clean_mask,
            confidence=confidence,
            metrics={
                "contour_area": float(obj_area),
                "size_mean": self.size_mean,
            }
        )
    
    
class DetectorColorThresholdKalman(DetectorColorThreshold):
    """Детектор с фильтром Калмана для сглаживания траектории"""
    
    def __init__(self, 
                 h_mean: int = 0, h_delta: int = 15,
                 s_mean: int = 0, s_delta: int = 40,
                 v_mean: int = 0, v_delta: int = 50,
                 size_mean: float = 1000.0,
                 size_ratio: float = 2.0,
                 morph_kernel: int = 7,
                 morph_close_iterations: int = 2,
                 merge_gap_px: float = 0.0,
                 merge_by_size: bool = True,
                 merge_size_factor: float = 0.5,
                 # Параметры Калмана
                 dt: float = 0.0333,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 1e-2
                 ):
        super().__init__(
            h_mean=h_mean, h_delta=h_delta,
            s_mean=s_mean, s_delta=s_delta,
            v_mean=v_mean, v_delta=v_delta,
            size_mean=size_mean, size_ratio=size_ratio,
            morph_kernel=morph_kernel,
            morph_close_iterations=morph_close_iterations,
            merge_gap_px=merge_gap_px,
            merge_by_size=merge_by_size,
            merge_size_factor=merge_size_factor
        )
        
        # Параметры Калмана
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Состояние Калмана
        self.kalman: Optional[cv2.KalmanFilter] = None
        self.measurement: Optional[np.ndarray] = None
        self._kalman_initialized = False
    
    def _init_kalman(self, x: float, y: float) -> None:
        """Инициализирует фильтр Калмана с начальной позицией"""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1e3
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.measurement = np.array([[x], [y]], dtype=np.float32)
        self._kalman_initialized = True
    
    def _predict_kalman(self) -> Optional[Tuple[float, float]]:
        """Предсказывает следующую позицию"""
        if not self._kalman_initialized or self.kalman is None:
            return None
        prediction = self.kalman.predict()
        return (prediction[0, 0], prediction[1, 0])
    
    def _correct_kalman(self, x: float, y: float) -> Tuple[float, float]:
        """Корректирует предсказание на основе измерения"""
        if not self._kalman_initialized or self.kalman is None:
            return (x, y)
        self.measurement[0, 0] = x
        self.measurement[1, 0] = y
        corrected = self.kalman.correct(self.measurement)
        return (corrected[0, 0], corrected[1, 0])
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Детектирует объект и применяет фильтр Калмана"""
        det = super().detect(frame)
        
        if det.center is None:
            if self._kalman_initialized:
                predicted = self._predict_kalman()
                if predicted:
                    det.center = (int(predicted[0]), int(predicted[1]))
                    det.metrics['kalman_predicted'] = 1.0
                    det.metrics['kalman_corrected'] = 0.0
            return det
        
        x, y = det.center
        
        if not self._kalman_initialized:
            self._init_kalman(float(x), float(y))
            det.metrics['kalman_predicted'] = 0.0
            det.metrics['kalman_corrected'] = 0.0
            return det
        
        predicted = self._predict_kalman()
        x_corrected, y_corrected = self._correct_kalman(float(x), float(y))
        det.center = (int(x_corrected), int(y_corrected))
        det.metrics['kalman_predicted'] = 1.0
        det.metrics['kalman_corrected'] = 1.0
        
        if predicted:
            dist = np.sqrt((x - predicted[0])**2 + (y - predicted[1])**2)
            det.metrics['kalman_prediction_error'] = float(dist)
        
        return det
    
    def reset_kalman(self) -> None:
        """Сбрасывает фильтр Калмана"""
        self.kalman = None
        self.measurement = None
        self._kalman_initialized = False