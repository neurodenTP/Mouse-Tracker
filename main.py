import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import ConfigLoader
from src.runner import Runner


def main():
    print("🐭 Запуск системы трекинга (ConfigLoader)")
    
    # 1️⃣ Загрузка конфигурации
    config = ConfigLoader("config.yaml")
    
    # 2️⃣ Источник видео
    source = config.create_source()
    
    # 3️⃣ Получаем FPS (нужен для трекера и рендереров)
    temp_open = source.open()
    target_fps = source.fps if temp_open else 30.0
    if temp_open:
        source.release()
    
    # 4️⃣ Калибровка
    context = config.create_context(source)
    if context.roi:
        source.roi = context.roi
    
    # 5️⃣ Фабрики объектов (всё из конфига!)
    detector = config.create_detector(context)
    metrics = config.create_metrics()
    tracker = config.create_tracker(metrics, target_fps)
    visualizers = config.create_visualizers(metrics)
    renderers = config.create_renderers(visualizers, target_fps)
    
    # 6️⃣ Запуск
    runner = Runner(
        source=source,
        detector=detector,
        renderers=renderers,
        tracker=tracker
    )
    
    # Засекаем время начала
    start_time = time.time()
    
    runner.run()
    
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*60)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
    print("="*60)
    print(f"⏱ Общее время выполнения: {total_time:.2f} секунд")
    print("="*60)
    
    # print("\n💡 Нажмите Enter для выхода...")
    # # Ждём нажатия Enter чтобы окно не закрывалось
    # try:
    #     input()
    # except:
    #     # Если input() не работает (например в IDE), ждём 10 секунд
    #     print("⏳ Окно закроется через 10 секунд...")
    #     time.sleep(10)

if __name__ == "__main__":
    main()