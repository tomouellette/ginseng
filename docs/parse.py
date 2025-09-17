#!/usr/bin/env python3
import ast
import os
from pathlib import Path
from typing import Dict, List
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


class PythonDocParser:
    def parse_numpy_docstring(self, docstring: str) -> DocInfo:
        """Parse NumPy-style docstring into structured data."""
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]
        summary = lines[0] if lines else ""

        parameters = []
        returns = ""
        returns_description = ""
        full_description = []
        current_section = None
        i = 1
        while i < len(lines):
            line = lines[i]
            if line in ["Parameters", "Returns", "Attributes"]:
                current_section = line.lower()
                i += 1
                if i < len(lines) and set(lines[i]) <= {"-", "="}:
                    i += 1
                continue
            if current_section in ["parameters", "attributes"]:
                if ":" in line:
                    parts = line.split(":", 1)
                    param_name = parts[0].strip()
                    param_type = parts[1].strip()
                    desc_lines = []
                    j = i + 1
                    # Collect description lines until we hit a blank line or a new parameter
                    while j < len(lines):
                        next_line = lines[j]
                        # Stop if we hit a blank line, new section, or new parameter
                        if (
                            next_line == ""
                            or next_line in ["Parameters", "Returns", "Attributes"]
                            or (":" in next_line and not next_line.startswith(" "))
                        ):
                            break
                        desc_lines.append(next_line)
                        j += 1

                    description = " ".join(desc_lines).strip()
                    parameters.append(Parameter(param_name, param_type, description))
                    i = j - 1
            elif current_section == "returns":
                if not returns:
                    returns = line
                else:
                    returns_description += f" {line}"
            else:
                full_description.append(line)
            i += 1
        return DocInfo(
            name="",
            summary=summary,
            parameters=parameters,
            returns=returns,
            returns_description=returns_description.strip(),
            full_description=" ".join(full_description).strip(),
        )

    def extract_class_info(self, node: ast.ClassDef) -> DocInfo:
        """Extract class docstring info."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None
        doc_info = self.parse_numpy_docstring(docstring)
        doc_info.name = node.name
        return doc_info

    def extract_function_info(self, node: ast.FunctionDef) -> DocInfo:
        """Extract function docstring info."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None
        doc_info = self.parse_numpy_docstring(docstring)
        doc_info.name = node.name
        return doc_info

    def generate_markdown(self, source_dir: str, output_file: str):
        """Generate Markdown documentation for all Python files in a directory."""
        markdown_content = ["# API\n"]
        source_path = Path(source_dir)
        py_files = [f for f in source_path.glob("*.py") if f.name != "__init__.py"]

        # Sort files alphabetically by module name
        py_files = sorted(py_files, key=lambda f: f.stem.lower())

        # First pass: collect all modules, classes, and functions for TOC
        toc_items = []
        module_content = []

        for py_file in py_files:
            module_name = py_file.stem
            module_classes = []
            module_functions = []

            with open(py_file, "r") as f:
                content = f.read()

            tree = ast.parse(content)

            # Collect classes and their methods
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    if class_info:
                        class_methods = []
                        for method_node in node.body:
                            if isinstance(method_node, ast.FunctionDef):
                                method_info = self.extract_function_info(method_node)
                                if method_info:
                                    class_methods.append(method_info.name)
                        module_classes.append((node.name, class_methods))

            # Collect module-level functions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = self.extract_function_info(node)
                    if func_info:
                        module_functions.append(func_info.name)

            if module_classes or module_functions:
                toc_items.append((module_name, module_classes, module_functions))
                module_content.append((py_file, tree))

        # Generate Table of Contents
        if toc_items:
            markdown_content.append("## Table of Contents\n")
            for module_name, classes, functions in toc_items:
                # Convert module name to GitHub anchor format (lowercase, replace spaces/special chars with hyphens)
                module_anchor = module_name.lower().replace("_", "-")
                markdown_content.append(f"- [{module_name}](#{module_anchor})")

                # Add classes and their methods
                for class_name, methods in classes:
                    class_anchor = f"class-{class_name.lower()}"
                    markdown_content.append(f"  - [{class_name}](#{class_anchor})")
                    for method_name in methods:
                        method_anchor = f"method-{method_name.lower()}"
                        markdown_content.append(
                            f"    - [{method_name}](#{method_anchor})"
                        )

                # Add module-level functions
                for func_name in functions:
                    func_anchor = f"function-{func_name.lower()}"
                    markdown_content.append(f"  - [{func_name}](#{func_anchor})")

            markdown_content.append("")

        # Generate actual documentation content
        for py_file, tree in module_content:
            module_name = py_file.stem
            markdown_content.append(f"## `{module_name}`\n")

            # Classes first
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    if not class_info:
                        continue
                    markdown_content.append(f"### Class `{class_info.name}`\n")
                    if class_info.summary:
                        markdown_content.append(f"{class_info.summary}\n")
                    if class_info.full_description:
                        markdown_content.append(f"{class_info.full_description}\n")
                    if class_info.parameters:
                        markdown_content.append("**Attributes:**")
                        for param in class_info.parameters:
                            markdown_content.append(
                                f"- **{param.name}** (`{param.type_hint}`): {param.description}"
                            )
                        markdown_content.append("")

                    # Extract methods from the class
                    for method_node in node.body:
                        if isinstance(method_node, ast.FunctionDef):
                            method_info = self.extract_function_info(method_node)
                            if not method_info:
                                continue
                            markdown_content.append(
                                f"#### Method `{method_info.name}`\n"
                            )
                            if method_info.summary:
                                markdown_content.append(f"{method_info.summary}\n")
                            if method_info.full_description:
                                markdown_content.append(
                                    f"{method_info.full_description}\n"
                                )
                            if method_info.parameters:
                                markdown_content.append("**Parameters:**")
                                for param in method_info.parameters:
                                    markdown_content.append(
                                        f"- **{param.name}** (`{param.type_hint}`): {param.description}"
                                    )
                                markdown_content.append("")
                            if method_info.returns:
                                markdown_content.append("**Returns:**")
                                markdown_content.append(
                                    f"- `{method_info.returns}`: {method_info.returns_description}"
                                )
                                markdown_content.append("")

            # Module-level functions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = self.extract_function_info(node)
                    if not func_info:
                        continue
                    markdown_content.append(f"### Function `{func_info.name}`\n")
                    if func_info.summary:
                        markdown_content.append(f"{func_info.summary}\n")
                    if func_info.full_description:
                        markdown_content.append(f"{func_info.full_description}\n")
                    if func_info.parameters:
                        markdown_content.append("**Parameters:**")
                        for param in func_info.parameters:
                            markdown_content.append(
                                f"- **{param.name}** (`{param.type_hint}`): {param.description}"
                            )
                        markdown_content.append("")
                    if func_info.returns:
                        markdown_content.append("**Returns:**")
                        markdown_content.append(
                            f"- `{func_info.returns}`: {func_info.returns_description}"
                        )
                        markdown_content.append("")

        with open(output_file, "w") as f:
            f.write("\n".join(markdown_content))

        print(f"Documentation generated: {output_file}")


def main():
    parser = PythonDocParser()
    parser.generate_markdown("src/ginseng", "docs/api.md")


if __name__ == "__main__":
    main()
