"""
Точка входу. Запускає консольне меню програми.

Для пакетного запуску всіх експериментів без меню:
    from menu import run_all_experiments
    run_all_experiments()
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from menu import run

if __name__ == "__main__":
    run()
