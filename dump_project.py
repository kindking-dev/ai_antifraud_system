import os

# Что мы НЕ берем (мусор и тяжелые папки)
EXCLUDE_DIRS = {'venv', '.git', '__pycache__', 'pgdata', 'redisdata', '.ipynb_checkpoints'}
# Какие файлы нам НУЖНЫ
INCLUDE_EXTS = {'.py', '.sql', '.yml', '.yaml', '.env', '.md', '.ini'}

def generate_dump(output_file="full_project_code.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== PROJECT STRUCTURE ===\n")
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            for file in files:
                if any(file.endswith(ext) for ext in INCLUDE_EXTS):
                    f.write(f"{indent}    ├── {file}\n")
        
        f.write("\n" + "="*50 + "\n FILE CONTENTS \n" + "="*50 + "\n")
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if any(file.endswith(ext) for ext in INCLUDE_EXTS) and file != 'dump_project.py':
                    filepath = os.path.join(root, file)
                    f.write(f"\n\n--- FILE: {filepath} ---\n")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as code_file:
                            f.write(code_file.read())
                    except Exception as e:
                        f.write(f"ERROR READING FILE: {e}")

if __name__ == "__main__":
    generate_dump()
    print("✅ Готово! Весь твой проект теперь в файле: full_project_code.txt")