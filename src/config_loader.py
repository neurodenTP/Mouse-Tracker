import os
import sys
import yaml
import cv2
from typing import Dict, Any, List, Optional

from src.source import Source, SourceCamera, SourceFile
from src.detector import (
    DetectorColorThreshold, DetectorStub, DetectorColorThresholdKalman
)
from src.metric import (
    MetricsTracker, Metric, SpeedMetric, DistanceMetric,
    PauseMetric, MetricHeatmap
)
from src.visualizer import (
    Visualizer, VisualizerCenterPoint, VisualizerContour, VisualizerHeatmap
)
from src.renderer import (
    Renderer, RendererDisplay, RendererFile, RendererCSV, RendererSnapshot
)
from src.setup import (
    CalibrationContext, run_setups, SetupROI, SetupSample, SetupSizeCalibration
)


class ConfigLoader:
    """Загрузка конфигурации и фабрики объектов"""
    
    def __init__(self, config_path: str = "config.yaml", config_dict: Optional[Dict[str, Any]] = None):
        if getattr(sys, 'frozen', False):
            base_path = sys.executable
        else:
            base_path = __file__
        
        # Ищем конфиг рядом с exe
        self.config_path = os.path.join(os.path.dirname(base_path), config_path)
        
        if not os.path.exists(self.config_path):
             # Фоллбэк: ищем в текущей рабочей директории
            self.config_path = config_path
        
        if config_dict is None:
            self.config_path = config_path
            self.config: Dict[str, Any] = {}
            self.load()
        else:
            self.config = config_dict

    def load(self) -> Dict[str, Any]:
        """Загружает конфиг из YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ Конфигурация не найдена: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"📂 Конфигурация загружена: {self.config_path}")
        return self.config

    def get(self, *keys, default: Any = None) -> Any:
        """Безопасное получение вложенных значений"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def create_source(self) -> Any:
        """Создаёт источник видео (камера или файл)"""
        src_cfg = self.get('source', default={})
        src_type = src_cfg.get('type', 'camera')
        fps = src_cfg.get('fps', 30.0)
        roi = src_cfg.get('roi', None)
        
        if src_type == 'camera':
            device_id = src_cfg.get('camera_id', 0)
            return SourceCamera(device_id=device_id, roi=roi, fps=fps)
        elif src_type == 'file':
            path = src_cfg.get('file_path', '')
            return SourceFile(path=path, roi=roi, fps=fps)
        else:
            raise ValueError(f"❌ Неизвестный тип источника: {src_type}")

    def create_context(self, source: Source) -> CalibrationContext:
        """
        Проводит калибровку на переданном источнике.
        :param source: Основной источник видео (уже созданный в main.py)
        """
        calib_cfg = self.get('calibration', default={})
        calib_path = calib_cfg.get('path', 'ExportData/calibration.json')
        
        context = CalibrationContext()
        
        # 🔹 Проверяем сохранённую калибровку
        if calib_cfg.get('auto_load', True) and os.path.exists(calib_path):
            print("📂 Найдена сохранённая калибровка. Загрузка...")
            context.load(calib_path)
        else:
            print("🛠 Запуск интерактивной калибровки...")
            
            # 🔹 Динамически собираем список этапов на основе конфига
            setups = []
            stages_cfg = calib_cfg.get('stages', {})
            
            if stages_cfg.get('roi', True):  # По умолчанию включено
                setups.append(SetupROI())
                print("   ✅ Этап ROI добавлен")
            
            if stages_cfg.get('size', False):  # По умолчанию выключено
                size_cfg = stages_cfg.get('size_params', {})
                width_mm = size_cfg.get('width_mm', 40.0)
                height_mm = size_cfg.get('height_mm', 40.0)
                setups.append(SetupSizeCalibration(width_mm=width_mm, height_mm=height_mm))
                print(f"   ✅ Этап калибровки размера добавлен ({width_mm}x{height_mm} мм)")
            
            if stages_cfg.get('sample', True):  # По умолчанию включено
                setups.append(SetupSample())
                print("   ✅ Этап образца добавлен")
            
            # 🔹 Запускаем калибровку
            if setups:
                context = run_setups(source, setups)
                context.save(calib_path)
            else:
                print("   ⚠️ Нет активных этапов калибровки. Создан пустой контекст.")
        
        return context

    def create_detector(self, context: CalibrationContext) -> Any:
        """Создаёт детектор на основе конфига"""
        det_cfg = self.get('detector', default={})
        det_type = det_cfg.get('type', 'color_threshold')
        
        # 🔹 Параметры порога (для всех детекторов кроме stub)
        threshold_params = det_cfg.get('threshold', {})
        
        if det_type == 'stub':
            return DetectorStub()
        
        elif det_type == 'color_threshold':
            detector = DetectorColorThreshold(
                h_mean=threshold_params.get('h_mean', 0),
                h_delta=threshold_params.get('h_delta', 60),
                s_mean=threshold_params.get('s_mean', 0),
                s_delta=threshold_params.get('s_delta', 60),
                v_mean=threshold_params.get('v_mean', 0),
                v_delta=threshold_params.get('v_delta', 60),
                size_mean=threshold_params.get('size_mean', 1000.0),
                size_ratio=threshold_params.get('size_ratio', 6.0),
                morph_kernel=threshold_params.get('morph_kernel', 9),
                morph_close_iterations=threshold_params.get('morph_close_iterations', 2),
                merge_gap_px=threshold_params.get('merge_gap_px', 0.0),
                merge_by_size=threshold_params.get('merge_by_size', True),
                merge_size_factor=threshold_params.get('merge_size_factor', 1.0)
            )
        
        elif det_type == 'color_threshold_kalman':
            # Параметры Калмана
            kalman_params = det_cfg.get('kalman', {})
            
            detector = DetectorColorThresholdKalman(
                h_mean=threshold_params.get('h_mean', 0),
                h_delta=threshold_params.get('h_delta', 60),
                s_mean=threshold_params.get('s_mean', 0),
                s_delta=threshold_params.get('s_delta', 60),
                v_mean=threshold_params.get('v_mean', 0),
                v_delta=threshold_params.get('v_delta', 60),
                size_mean=threshold_params.get('size_mean', 1000.0),
                size_ratio=threshold_params.get('size_ratio', 6.0),
                morph_kernel=threshold_params.get('morph_kernel', 9),
                morph_close_iterations=threshold_params.get('morph_close_iterations', 2),
                merge_gap_px=threshold_params.get('merge_gap_px', 0.0),
                merge_by_size=threshold_params.get('merge_by_size', True),
                merge_size_factor=threshold_params.get('merge_size_factor', 1.0),
                # Параметры Калмана
                dt=kalman_params.get('dt', 0.0333),
                process_noise=kalman_params.get('process_noise', 1e-5),
                measurement_noise=kalman_params.get('measurement_noise', 1e-2)
            )
        
        else:
            raise ValueError(f"❌ Неизвестный тип детектора: {det_type}")
        
        # 🔹 Применяем калибровку если есть
        if context.sample_color is not None:
            detector.auto_calibrate_color(
                context.sample_color, 
                context.background_color
            )
        if context.sample_size is not None:
            detector.auto_calibrate_size(context.sample_size)
        if context.init_position is not None:
            detector.set_initial_position(context.init_position)
        
        return detector

    def create_metrics(self) -> List[Metric]:
        """Создаёт список метрик на основе конфига (список items)"""
        metrics_cfg = self.get('metrics', default={})
        metrics: List[Metric] = []
        
        items = metrics_cfg.get('items', [])
        
        for item in items:
            name = item.get('name', '')
            params = {k: v for k, v in item.items() if k != 'name'}
            
            if name == 'speed':
                metrics.append(SpeedMetric(
                    smoothing_frames=params.get('smoothing_frames', 5)
                ))
            elif name == 'distance':
                metrics.append(DistanceMetric())
            elif name == 'pause':
                metrics.append(PauseMetric(
                    speed_threshold=params.get('speed_threshold', 3.0),
                    min_pause_s=params.get('min_pause_s', 0.5)
                ))
            elif name == 'heatmap':
                metrics.append(MetricHeatmap(
                    decay=params.get('decay', 0.97),
                    blur_kernel=params.get('blur_kernel', 25)
                ))
            else:
                print(f"⚠️ Неизвестная метрика: {name}")
        
        return metrics

    def create_tracker(self, metrics: List[Metric], fps: float) -> MetricsTracker:
        """Создаёт трекер метрик"""
        tracker_cfg = self.get('metrics', default={})
        return MetricsTracker(
            metrics=metrics,
            history_size=tracker_cfg.get('history_size', 30),
            fps=fps
        )

    def create_visualizers(self, metrics: List[Metric]) -> List[Visualizer]:
        """Создаёт список визуализаторов на основе конфига"""
        vis_cfg = self.get('visualizers', default={})
        visualizers: List[Visualizer] = []
        
        heatmap_metric = None
        for m in metrics:
            if isinstance(m, MetricHeatmap):
                heatmap_metric = m
                break
        
        if vis_cfg.get('heatmap', {}).get('enabled', True):
            params = vis_cfg['heatmap']
            if heatmap_metric:
                visualizers.append(VisualizerHeatmap(
                    metric_heatmap=heatmap_metric,
                    colormap=getattr(cv2, f"COLORMAP_{params.get('colormap', 'INFERNO').upper()}"),
                    opacity=params.get('opacity', 0.65)
                ))
        
        if vis_cfg.get('contour', {}).get('enabled', True):
            params = vis_cfg['contour']
            visualizers.append(VisualizerContour(
                color=tuple(params.get('color', [0, 255, 0])),
                thickness=params.get('thickness', 2)
            ))
        
        if vis_cfg.get('center_point', {}).get('enabled', True):
            params = vis_cfg['center_point']
            visualizers.append(VisualizerCenterPoint(
                color=tuple(params.get('color', [0, 0, 255])),
                radius=params.get('radius', 8)
            ))
        
        return visualizers
    
    
    def create_renderers(self, visualizers: List[Visualizer], fps: float) -> List[Renderer]:
        """
        Создаёт список рендереров из секции outputs.
        Каждый рендерер получает только свой поднабор визуализаторов.
        """
        ren_cfg = self.get('renderers', default={})
        renderers: List[Renderer] = []
        
        # 🔹 Словарь для быстрого доступа к созданным визуализаторам
        vis_dict = {
            'heatmap': None,
            'contour': None,
            'center_point': None
        }
        for v in visualizers:
            if isinstance(v, VisualizerHeatmap):
                vis_dict['heatmap'] = v
            elif isinstance(v, VisualizerContour):
                vis_dict['contour'] = v
            elif isinstance(v, VisualizerCenterPoint):
                vis_dict['center_point'] = v
        
        # 🔹 Display
        if ren_cfg.get('display', {}).get('enabled', True):
            params = ren_cfg['display']
            vis_list = params.get('visualizers', ['heatmap', 'contour', 'center_point'])
            selected_vis = [vis_dict[name] for name in vis_list if vis_dict.get(name)]
            
            renderers.append(RendererDisplay(
                visualizers=selected_vis,
                window_name=params.get('window_name', "Mouse Tracker"),
                fps=fps
            ))
        
        # 🔹 Video File
        if ren_cfg.get('video_file', {}).get('enabled', True):
            params = ren_cfg['video_file']
            vis_list = params.get('visualizers', ['contour', 'center_point'])
            selected_vis = [vis_dict[name] for name in vis_list if vis_dict.get(name)]
            
            renderers.append(RendererFile(
                visualizers=selected_vis,
                output_path=params.get('path', "ExportData/tracking_result.mp4"),
                fps=fps,
                codec=params.get('codec', "mp4v")
            ))
        
        # 🔹 CSV
        if ren_cfg.get('csv', {}).get('enabled', True):
            params = ren_cfg['csv']
            renderers.append(RendererCSV(
                visualizers=[],  # CSV не использует визуализаторы
                output_path=params.get('path', "ExportData/tracking_data.csv"),
                fps=fps
            ))
        
        # 🔹 Snapshots
        if ren_cfg.get('snapshots', {}).get('enabled', True):
            params = ren_cfg['snapshots']
            vis_list = params.get('visualizers', ['heatmap'])
            selected_vis = [vis_dict[name] for name in vis_list if vis_dict.get(name)]
            
            renderers.append(RendererSnapshot(
                visualizers=selected_vis,
                output_dir=params.get('output_dir', "ExportData/snapshots"),
                interval_frames=params.get('interval_frames', 100),
                save_at_end=params.get('save_at_end', "true"),
                end_filename=params.get('end_filename', "final_snapshot.png")
            ))
        
        return renderers