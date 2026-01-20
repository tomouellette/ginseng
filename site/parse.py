#!/usr/bin/env python3
import ast
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class Parameter:
    name: str
    type_hint: str
    description: str
    default: str = None


@dataclass
class DocInfo:
    name: str
    summary: str
    parameters: List[Parameter]
    returns: str
    returns_description: str
    full_description: str
    examples: str = ""
    notes: str = ""


class PythonDocParser:
    def parse_numpy_docstring(self, docstring: str) -> DocInfo:
        """Parse NumPy-style docstring into structured data with robust section handling."""
        if not docstring:
            return DocInfo("", "", [], "", "", "", "", "")

        raw_lines = docstring.splitlines()
        # Keep original lines for content, but have a stripped version for header detection
        stripped_lines = [line.strip() for line in raw_lines]

        summary = stripped_lines[0] if stripped_lines else ""
        parameters = []
        returns = ""
        returns_description = ""
        full_description = []
        examples_content = []
        notes_content = []

        current_section = None
        i = 1

        headers = [
            "Parameters",
            "Returns",
            "Attributes",
            "Example",
            "Examples",
            "Notes",
            "Raises",
        ]

        while i < len(raw_lines):
            line_stripped = stripped_lines[i]

            # Identify a section header
            is_header = False
            if line_stripped in headers:
                if i + 1 < len(stripped_lines):
                    underline = stripped_lines[i + 1]
                    if len(underline) > 0 and set(underline) <= {"-", "="}:
                        is_header = True

            if is_header:
                current_section = line_stripped.lower()
                if current_section == "example":
                    current_section = "examples"
                # Skip header and underline
                i += 2
                continue

            # Section parsing logic
            if current_section in ["parameters", "attributes"]:
                if ":" in line_stripped:
                    parts = line_stripped.split(":", 1)
                    param_name = parts[0].strip()
                    param_type = parts[1].strip()
                    desc_lines = []
                    j = i + 1
                    while j < len(raw_lines):
                        # Break if we hit a new header
                        if stripped_lines[j] in headers and j + 1 < len(stripped_lines):
                            if (
                                set(stripped_lines[j + 1]) <= {"-", "="}
                                and len(stripped_lines[j + 1]) > 0
                            ):
                                break
                        # Break if we hit a new parameter (no leading space in raw line)
                        if ":" in stripped_lines[j] and not raw_lines[j].startswith(
                            " "
                        ):
                            break
                        desc_lines.append(stripped_lines[j])
                        j += 1
                    description = " ".join(desc_lines).strip()
                    parameters.append(Parameter(param_name, param_type, description))
                    i = j - 1

            elif current_section == "returns":
                if not returns:
                    if ":" in line_stripped:
                        parts = line_stripped.split(":", 1)
                        returns = parts[0].strip()
                        returns_description = parts[1].strip()
                    else:
                        returns = line_stripped
                else:
                    returns_description += f" {line_stripped}"

            elif current_section == "examples":
                # Crucial: append raw_lines[i] to preserve leading spaces (>>> and indentation)
                examples_content.append(raw_lines[i])

            elif current_section == "notes":
                notes_content.append(line_stripped)

            else:
                if line_stripped:
                    full_description.append(line_stripped)
            i += 1

        return DocInfo(
            name="",
            summary=summary,
            parameters=parameters,
            returns=returns.strip(),
            returns_description=returns_description.strip(),
            full_description=" ".join(full_description).strip(),
            examples="\n".join(examples_content).strip(),
            notes=" ".join(notes_content).strip(),
        )

    def extract_class_info(self, node: ast.ClassDef) -> DocInfo:
        docstring = ast.get_docstring(node)
        if not docstring:
            return None
        doc_info = self.parse_numpy_docstring(docstring)
        doc_info.name = node.name
        return doc_info

    def extract_function_info(self, node: ast.FunctionDef) -> DocInfo:
        docstring = ast.get_docstring(node)
        if not docstring:
            return None
        doc_info = self.parse_numpy_docstring(docstring)
        doc_info.name = node.name
        return doc_info

    def _get_anchor(self, text: str) -> str:
        return (
            text.lower()
            .replace("/", "")
            .replace(".", "")
            .replace(" ", "-")
            .replace("_", "-")
        )

    def generate_markdown(self, source_dirs: List[str], output_file: str):
        markdown_content = ["# API Reference\n"]
        py_files = []

        for src in source_dirs:
            path = Path(src)
            if path.exists():
                py_files.extend(
                    [f for f in path.glob("*.py") if f.name != "__init__.py"]
                )

        py_files = sorted(py_files, key=lambda f: str(f).lower())
        module_data = []

        for py_file in py_files:
            display_name = f"{py_file.parent.name}/{py_file.name}"
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            classes_in_module = []
            functions_in_module = []

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    if class_info:
                        methods = []
                        for m_node in node.body:
                            if isinstance(
                                m_node, ast.FunctionDef
                            ) and not m_node.name.startswith("__"):
                                m_info = self.extract_function_info(m_node)
                                if m_info:
                                    methods.append(m_info)
                        classes_in_module.append((class_info, methods))
                elif isinstance(node, ast.FunctionDef):
                    func_info = self.extract_function_info(node)
                    if func_info:
                        functions_in_module.append(func_info)

            if classes_in_module or functions_in_module:
                module_data.append(
                    (display_name, classes_in_module, functions_in_module)
                )

        # Table of contents
        markdown_content.append("## Table of Contents\n")
        for display_name, classes, functions in module_data:
            file_anchor = self._get_anchor(display_name)
            markdown_content.append(f"- [{display_name}](#{file_anchor})")

            for cls_info, methods in classes:
                cls_anchor = f"class-{self._get_anchor(cls_info.name)}"
                markdown_content.append(
                    f"    - [Class: {cls_info.name}](#{cls_anchor})"
                )
                for m in methods:
                    m_anchor = f"method-{self._get_anchor(m.name)}"
                    markdown_content.append(
                        f"        - [Method: {m.name}](#{m_anchor})"
                    )

            for func_info in functions:
                f_anchor = f"function-{self._get_anchor(func_info.name)}"
                markdown_content.append(
                    f"    - [Function: {func_info.name}](#{f_anchor})"
                )

        markdown_content.append("\n---\n")

        # Documentation body
        for display_name, classes, functions in module_data:
            file_anchor = self._get_anchor(display_name)
            markdown_content.append(
                f'## <a name="{file_anchor}"></a>File: `{display_name}`\n'
            )

            for cls_info, methods in classes:
                cls_anchor = f"class-{self._get_anchor(cls_info.name)}"
                markdown_content.append(
                    f'### <a name="{cls_anchor}"></a>Class `{cls_info.name}`'
                )
                if cls_info.summary:
                    markdown_content.append(f"\n{cls_info.summary}\n")
                if cls_info.full_description:
                    markdown_content.append(f"{cls_info.full_description}\n")

                if cls_info.parameters:
                    markdown_content.append("**Attributes:**\n")
                    for p in cls_info.parameters:
                        markdown_content.append(
                            f"  - **{p.name}** (`{p.type_hint}`): {p.description}"
                        )
                    markdown_content.append("")

                if cls_info.examples:
                    markdown_content.append("**Examples:**\n")
                    markdown_content.append(f"```python\n{cls_info.examples}\n```\n")

                for m in methods:
                    m_anchor = f"method-{self._get_anchor(m.name)}"
                    markdown_content.append(
                        f'#### <a name="{m_anchor}"></a>Method `{m.name}`'
                    )
                    if m.summary:
                        markdown_content.append(f"\n{m.summary}\n")
                    if m.parameters:
                        markdown_content.append("**Parameters:**\n")
                        for p in m.parameters:
                            markdown_content.append(
                                f"  - **{p.name}** (`{p.type_hint}`): {p.description}"
                            )
                        markdown_content.append("")
                    if m.returns:
                        markdown_content.append("**Returns:**\n")
                        markdown_content.append(
                            f"  - `{m.returns}`: {m.returns_description}\n"
                        )
                    if m.examples:
                        markdown_content.append("**Examples:**\n")
                        markdown_content.append(f"```python\n{m.examples}\n```\n")

            for f_info in functions:
                f_anchor = f"function-{self._get_anchor(f_info.name)}"
                markdown_content.append(
                    f'### <a name="{f_anchor}"></a>Function `{f_info.name}`'
                )
                if f_info.summary:
                    markdown_content.append(f"\n{f_info.summary}\n")
                if f_info.parameters:
                    markdown_content.append("**Parameters:**\n")
                    for p in f_info.parameters:
                        markdown_content.append(
                            f"  - **{p.name}** (`{p.type_hint}`): {p.description}"
                        )
                    markdown_content.append("")
                if f_info.returns:
                    markdown_content.append("**Returns:**\n")
                    markdown_content.append(
                        f"  - `{f_info.returns}`: {f_info.returns_description}\n"
                    )
                if f_info.examples:
                    markdown_content.append("**Examples:**\n")
                    markdown_content.append(f"```python\n{f_info.examples}\n```\n")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))
        print(f"Successfully generated: {output_file}")


def main():
    parser = PythonDocParser()
    sources = [
        "src/ginseng/model/",
        "src/ginseng/train/",
        "src/ginseng/data/",
        "src/ginseng/utils/",
    ]
    parser.generate_markdown(sources, "docs/api.md")


if __name__ == "__main__":
    main()
