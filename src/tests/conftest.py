# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import os
from datetime import datetime

_test_reports = []


def pytest_runtest_logreport(report):
    """Capture individual test results."""
    if report.when == "call":
        status = report.outcome.upper()
        _test_reports.append((report.nodeid, status))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write the terminal-style summary to README.md."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    readme_path = os.path.join(os.path.dirname(__file__), "README.md")

    with open(readme_path, "w") as f:
        f.write("# tests\n\n")
        f.write(f"**Last Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        f.write("```text\n")

        # Determine max width for the nodeid column
        id_column_width = max([len(r[0]) for r in _test_reports] + [80])

        for i, (nodeid, status) in enumerate(_test_reports, 1):
            percent = int((i / len(_test_reports)) * 100)
            f.write(f"{nodeid:<{id_column_width}} {status:<10} [{percent:>3}%]\n")

        f.write("\n" + "=" * (id_column_width + 20) + "\n")
        summary = f"{passed} passed, {failed} failed, {skipped} skipped"
        f.write(f"{summary:^{id_column_width + 20}}\n")
        f.write("```\n")
