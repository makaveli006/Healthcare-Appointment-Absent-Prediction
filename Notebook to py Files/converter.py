import json
from pathlib import Path

def notebook_to_python(notebook_path):
    """
    Converts a Jupyter Notebook (.ipynb) into a Python script (.py)
    - Code cells become regular Python code.
    - Markdown cells become triple-quoted docstrings.
    - Output file keeps the same name as the notebook.
    """
    notebook_path = Path(notebook_path)
    output_path = notebook_path.with_suffix(".py")

    # Load the notebook JSON
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    lines = []
    for cell in nb_data.get("cells", []):
        cell_type = cell.get("cell_type")
        source = ''.join(cell.get("source", []))
        if not source.strip():
            continue  # skip empty cells
        if cell_type == "markdown":
            lines.append(f'"""\n{source.strip()}\n"""\n\n')
        elif cell_type == "code":
            lines.append(f"{source.strip()}\n\n")

    # Write everything to the .py file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✅ Converted: {notebook_path.name} → {output_path.name}")

def convert_all_notebooks_in_folder(folder="."):
    """
    Converts all .ipynb files in the given folder (default: current folder)
    """
    folder_path = Path(folder)
    ipynb_files = list(folder_path.glob("*.ipynb"))

    if not ipynb_files:
        print("No .ipynb files found in this folder.")
        return

    for notebook_file in ipynb_files:
        try:
            notebook_to_python(notebook_file)
        except Exception as e:
            print(f"❌ Error converting {notebook_file.name}: {e}")

# Example usage
if __name__ == "__main__":
    convert_all_notebooks_in_folder(".")
