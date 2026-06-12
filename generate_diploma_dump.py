import os
from pathlib import Path
import logging
from typing import Set, List

# =========================
# КОНФИГУРАЦИЯ ДАМПА
# =========================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "diploma_full_dump.txt"

# Максимальный размер файла (100 КБ), чтобы не захватить огромные логи или веса моделей
MAX_FILE_SIZE_BYTES = 100 * 1024 

# Директории, которые мы игнорируем (виртуальные окружения, логи, БД, кэш)
IGNORE_DIRS: Set[str] = {
    ".git", ".venv", "venv", "__pycache__", "data", "Datasets", 
    "catboost_info", "logs", "reports", ".pytest_cache", "alembic", 
    ".idea", "ml_artifacts"
}

# Расширения файлов, которые пойдут в текстовый дамп
ALLOWED_EXTENSIONS: Set[str] = {
    ".py", ".json", ".yml", ".yaml", ".md", ".txt", ".conf", ".sql"
}

# Специфичные файлы, которые нужно пропустить
IGNORE_FILES: Set[str] = {
    ".env", "generate_diploma_dump.py", "dump_project.py", 
    "diploma_full_dump.txt", "project_context_dump.txt", "full_project_code.txt"
}

def generate_tree(dir_path: Path, prefix: str = "", is_last: bool = True) -> List[str]:
    """Рекурсивно строит красивое дерево директорий."""
    tree_lines = []
    folder_name = dir_path.name
    
    if folder_name in IGNORE_DIRS or folder_name.startswith("."):
        return tree_lines

    connector = "└── " if is_last else "├── "
    tree_lines.append(f"{prefix}{connector}{folder_name}/")
    
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    try:
        items = sorted(list(dir_path.iterdir()), key=lambda x: (x.is_file(), x.name))
        valid_items = [i for i in items if i.name not in IGNORE_DIRS and not i.name.startswith(".")]
        
        for index, item in enumerate(valid_items):
            is_item_last = index == (len(valid_items) - 1)
            if item.is_dir():
                tree_lines.extend(generate_tree(item, new_prefix, is_item_last))
            elif item.is_file():
                if item.suffix in ALLOWED_EXTENSIONS and item.name not in IGNORE_FILES:
                    item_connector = "└── " if is_item_last else "├── "
                    tree_lines.append(f"{new_prefix}{item_connector}{item.name}")
    except PermissionError:
        pass
        
    return tree_lines

def build_dump() -> None:
    """Собирает дерево структуры и весь исходный код в один файл."""
    logger.info("🔍 Строим дерево архитектуры проекта...")
    tree_output = generate_tree(BASE_DIR)
    
    dump_content = [
        "=========================================",
        "       ПОЛНЫЙ ДАМП ПРОЕКТА SENTINEL AI   ",
        "=========================================\n",
        "=== СТРУКТУРА ПРОЕКТА ==="
    ]
    dump_content.extend(tree_output)
    dump_content.append("\n=========================================\n")
    
    logger.info("📖 Собираем исходный код модулей...")
    file_count = 0
    
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        
        for file in sorted(files):
            if file in IGNORE_FILES or file.startswith("."):
                continue
                
            file_path = Path(root) / file
            
            if file_path.suffix in ALLOWED_EXTENSIONS:
                try:
                    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                        logger.warning(f"⚠️ Пропущен большой файл (>100KB): {file_path.name}")
                        continue
                except OSError:
                    continue

                relative_path = file_path.relative_to(BASE_DIR)
                dump_content.append(f"\n\n=== ФАЙЛ: {relative_path} ===")
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        dump_content.append(f.read())
                    file_count += 1
                except Exception as e:
                    dump_content.append(f"// ОШИБКА ЧТЕНИЯ ФАЙЛА: {e}")
                    logger.error(f"❌ Ошибка чтения {relative_path}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("\n".join(dump_content))
        
    logger.info(f"\n✅ УСПЕШНО! Дамп сохранен в файл: {OUTPUT_FILE}")
    logger.info(f"📊 Всего включено файлов с кодом: {file_count}")

if __name__ == "__main__":
    build_dump()
