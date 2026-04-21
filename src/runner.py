import numpy as np
from src.source import Source
from src.detector import Detector
from src.renderer import Renderer
from src.metric import MetricsTracker
from typing import List, Optional

class Runner:
    """Оркестратор: управляет потоком данных между компонентами"""
    
    def __init__(self, 
                 source: Source, 
                 detector: Detector, 
                 renderers: List[Renderer],
                 tracker: Optional[MetricsTracker] = None):
        self.source = source
        self.detector = detector
        self.renderers = renderers
        self.tracker = tracker
        self._running = False

    def run(self) -> None:
        if not self.source.open():
            raise RuntimeError("❌ Не удалось открыть источник видео")
        
        self._running = True
        frame_count = 0
        print("✅ Источник открыт. Запуск цикла обработки...")
        
        try:
            while self._running:
                success, raw_frame = self.source.read()
                if not success or raw_frame is None:
                    print("⏹ Конец видео или ошибка чтения.")
                    break
                
                # 1. Предобработка
                frame = self.source.preprocess(raw_frame)
                
                # 2. Детекция
                det = self.detector.detect(frame)
                
                # 3. 📊 Расчёт метрик (если трекер подключен)
                if self.tracker:
                    metrics = self.tracker.update(det)
                    det.metrics.update(metrics)
                
                # 4. Визуализация и вывод
                for renderer in self.renderers:
                    renderer.render(frame, det)
                    if renderer.quit_requested:
                        self._running = False
                        print("👋 Запрошен выход из интерфейса.")
                        break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⛔ Прервано пользователем (Ctrl+C)")
        finally:
            self._cleanup()
            print(f"📊 Обработано кадров: {frame_count}")
            
            # Вывод сводки метрик
            if self.tracker:
                summary = self.tracker.get_summary()
                print("\n📈 Сводка метрик:")
                for key, value in summary.items():
                    print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    def stop(self) -> None:
        self._running = False

    def _cleanup(self) -> None:
        self._running = False
        self.source.release()
        for r in self.renderers:
            r.close()

# class Runner:
#     """Оркестратор: управляет потоком данных между компонентами"""
    
#     def __init__(self, source: Source, detector: Detector, renderers: List[Renderer]):
#         self.source = source
#         self.detector = detector
#         self.renderers = renderers
#         self._running = False

#     def run(self) -> None:
#         if not self.source.open():
#             raise RuntimeError("❌ Не удалось открыть источник видео")
        
#         self._running = True
#         frame_count = 0
#         print("✅ Источник открыт. Запуск цикла обработки...")
        
#         try:
#             while self._running:
#                 success, raw_frame = self.source.read()
#                 if not success or raw_frame.size == 0:
#                     print("⏹ Конец видео или ошибка чтения.")
#                     break
                
#                 # 1. Предобработка (обрезка ROI и т.д.)
#                 frame = self.source.preprocess(raw_frame)
                
#                 # 2. Детекция
#                 det = self.detector.detect(frame)
                
#                 # 3. Визуализация и вывод
#                 for renderer in self.renderers:
#                     renderer.render(frame, det)
#                     if renderer.quit_requested:
#                         self._running = False
#                         print("👋 Запрошен выход из интерфейса.")
#                         break
                
#                 frame_count += 1
                
#         except KeyboardInterrupt:
#             print("\n⛔ Прервано пользователем (Ctrl+C)")
#         finally:
#             self._cleanup()
#             print(f"📊 Обработано кадров: {frame_count}")

#     def stop(self) -> None:
#         self._running = False

#     def _cleanup(self) -> None:
#         self._running = False
#         self.source.release()
#         for r in self.renderers:
#             r.close()