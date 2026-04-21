import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, TextIO
import csv

from src.detector import DetectionResult
from src.visualizer import Visualizer

class Renderer(ABC):
    """Базовый класс для вывода/сохранения обработанного кадра"""
    
    def __init__(self, visualizers: List[Visualizer]):
        self.visualizers = visualizers
        self.quit_requested: bool = False

    def _apply_visualizers(self, frame: np.ndarray, det: DetectionResult) -> np.ndarray:
        """Последовательно применяет все визуализаторы к копии кадра"""
        out = frame.copy()
        for vis in self.visualizers:
            out = vis.draw(out, det)
        return out

    @abstractmethod
    def render(self, frame: np.ndarray, det: DetectionResult) -> None: ...
    
    @abstractmethod
    def close(self) -> None: ...


class RendererDisplay(Renderer):
    """Вывод видео в окно реального времени"""
    def __init__(self, visualizers: List[Visualizer], window_name: str = "Mouse Tracker", fps: float = 30.0):
        super().__init__(visualizers)
        self.window_name = window_name
        self.delay_ms = max(1, int(1000 / fps))

    def render(self, frame: np.ndarray, det: DetectionResult) -> None:
        annotated = self._apply_visualizers(frame, det)
        cv2.imshow(self.window_name, annotated)
        
        key = cv2.waitKey(self.delay_ms) & 0xFF
        if key in [ord('q'), 27]:  # 'q' или Esc
            self.quit_requested = True

    def close(self) -> None:
        cv2.destroyAllWindows()


class RendererFile(Renderer):
    """Записывает обработанное видео с визуализациями в файл"""
    def __init__(self, 
                 visualizers: List[Visualizer], 
                 output_path: str, 
                 fps: float = 30.0, 
                 codec: str = "mp4v"):
        super().__init__(visualizers)
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self._initialized: bool = False
        
        # Создаём директорию вывода, если её нет
        dir_path = os.path.dirname(os.path.abspath(output_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def render(self, frame: np.ndarray, det: DetectionResult) -> None:
        # Ленивая инициализация VideoWriter (требуется размер кадра)
        if not self._initialized:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            
            if not self.writer.isOpened():
                raise RuntimeError(f"❌ Не удалось инициализировать запись видео в {self.output_path}")
            
            print(f"🎬 Запись видео начата: {self.output_path} ({w}x{h} @ {self.fps} FPS)")
            self._initialized = True
            
        # Применяем цепочку визуализаторов и пишем кадр
        annotated = self._apply_visualizers(frame, det)
        self.writer.write(annotated)

    def close(self) -> None:
        if self.writer and self._initialized:
            self.writer.release()
            self.writer = None
            self._initialized = False
            print(f"💾 Видео сохранено: {self.output_path}")
            

class RendererCSV(Renderer):
    """Записывает метрики трекинга в CSV. Динамически обнаруживает колонки из det.metrics."""
    
    def __init__(self, visualizers: List[Visualizer], output_path: str, fps: float = 30.0):
        super().__init__(visualizers)
        self.output_path = output_path
        self.fps = fps
        self.file: Optional[TextIO] = None
        self.writer = None
        self.frame_idx = 0
        self._is_opened = False
        self._fieldnames: Optional[List[str]] = None  # Запоминаем заголовок

        dir_path = os.path.dirname(os.path.abspath(output_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def render(self, frame: np.ndarray, det: DetectionResult) -> None:
        # Ленивая инициализация: сканируем метрики и создаём заголовок при первом кадре
        if not self._is_opened:
            self.file = open(self.output_path, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            
            # 🔍 Динамическое построение заголовка
            base_fields = [
                'frame_idx', 'timestamp_ms', 'detected',
                'center_x', 'center_y',
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'confidence', 'area'
            ]
            # Добавляем все ключи из metrics текущего кадра
            metric_fields = sorted(det.metrics.keys()) if det.metrics else []
            self._fieldnames = base_fields + metric_fields
            
            self.writer.writerow(self._fieldnames)
            self._is_opened = True
            print(f"📊 CSV заголовок: {len(self._fieldnames)} колонок")

        # Извлечение базовых данных
        detected = 1 if det.center is not None else 0
        cx, cy = det.center if det.center else (None, None)
        bbox = det.bbox if det.bbox else (None, None, None, None)
        area = det.metrics.get('contour_area', 0.0)
        timestamp_ms = (self.frame_idx / self.fps) * 1000

        # 🔑 Формирование строки данных в порядке заголовка
        row_dict = {
            'frame_idx': self.frame_idx,
            'timestamp_ms': f"{timestamp_ms:.2f}",
            'detected': detected,
            'center_x': cx if cx is not None else "",
            'center_y': cy if cy is not None else "",
            'bbox_x': bbox[0] if bbox[0] is not None else "",
            'bbox_y': bbox[1] if bbox[1] is not None else "",
            'bbox_w': bbox[2] if bbox[2] is not None else "",
            'bbox_h': bbox[3] if bbox[3] is not None else "",
            'confidence': f"{det.confidence:.4f}",
            'area': f"{area:.2f}"
        }
        
        # Добавляем все метрики
        for key in det.metrics:
            row_dict[key] = det.metrics[key]
        
        # Запись строки в порядке fieldnames
        row = [row_dict.get(field, "") for field in self._fieldnames]
        self.writer.writerow(row)

        # 🔒 Гарантия сохранности
        self.file.flush()
        self.frame_idx += 1

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()
            self._is_opened = False
            print(f"💾 Трекинг-данные сохранены: {self.output_path} ({self.frame_idx} кадров)")
            

class RendererSnapshot(Renderer):
    """Сохраняет обработанные кадры в PNG (периодически и/или в конце)"""
    def __init__(self, 
                 visualizers: List[Visualizer], 
                 output_dir: str, 
                 interval_frames: int = 0,      # 0 = не сохранять периодически
                 save_at_end: bool = False,     # Сохранить один кадр после завершения
                 end_filename: str = "final_snapshot.png"):
        super().__init__(visualizers)
        self.output_dir = output_dir
        self.interval_frames = interval_frames
        self.save_at_end = save_at_end
        self.end_filename = end_filename
        self.frame_count = 0
        self._last_saved_frame: Optional[np.ndarray] = None  # Последний обработанный кадр
        os.makedirs(output_dir, exist_ok=True)

    def render(self, frame: np.ndarray, det: DetectionResult) -> None:
        self.frame_count += 1
        
        # Сохраняем последний кадр для возможного использования в close()
        if self.save_at_end:
            self._last_saved_frame = self._apply_visualizers(frame, det).copy()
        
        # Периодические снимки (если interval_frames > 0)
        if self.interval_frames > 0 and self.frame_count % self.interval_frames == 0:
            annotated = self._apply_visualizers(frame, det)
            filename = f"snapshot_{self.frame_count:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, annotated)
            print(f"📸 Snapshot saved: {filename}")

    def close(self) -> None:
        # 🔹 Финальный снимок после завершения обработки
        if self.save_at_end and self._last_saved_frame is not None:
            filepath = os.path.join(self.output_dir, self.end_filename)
            cv2.imwrite(filepath, self._last_saved_frame)
            print(f"📸 Final snapshot saved: {self.end_filename}")
        
        print(f"📸 Snapshots saved: {self.output_dir} ({self.frame_count} frames processed)")