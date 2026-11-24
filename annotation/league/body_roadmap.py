"""Read-only validation of the champion body recognition roadmap."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .media import file_hash

STATUS = {"planned", "ready", "running", "blocked", "complete", "rejected"}
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^### (M\d+) (.+)$", re.MULTILINE)


def validate(path, repository):
    path, repository = Path(path).resolve(), Path(repository).resolve()
    text = path.read_text()
    headings = list(HEADING.finditer(text))
    errors = []
    names = {heading[1] for heading in headings}
    if not headings or len(names) != len(headings):
        errors.append("Milestones must have unique IDs")
    for match in LINK.finditer(text):
        target = match[1].split("#", 1)[0]
        if not target or target.startswith(("https://", "http://")):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(repository) or not resolved.exists():
            errors.append(f"Invalid repository link {target}")
    help_by_module = {}
    for index, heading in enumerate(headings):
        tail = text[heading.end():headings[index + 1].start()] if index + 1 < len(headings) else text[heading.end():]
        body = re.split(r"^## ", tail, maxsplit=1, flags=re.MULTILINE)[0]
        fields = dict(re.findall(r"^- (Status|Depends on|Exit|Evidence|Receipt): (.+)$", body, re.MULTILINE))
        milestone = heading[1]
        for field in ("Status", "Depends on", "Exit", "Evidence"):
            if not fields.get(field):
                errors.append(f"{milestone} needs {field}")
        if fields.get("Status") not in STATUS:
            errors.append(f"{milestone} has an invalid status")
        dependencies = fields.get("Depends on", "none")
        if dependencies != "none" and not set(dependencies.split(", ")) <= names - {milestone}:
            errors.append(f"{milestone} has an invalid dependency")
        if not re.search(r"^- \[[ x]\]", body, re.MULTILINE):
            errors.append(f"{milestone} needs a checklist")
        commands = re.findall(r"^- Command \((available|planned)\): `([^`]+)`$", body, re.MULTILINE)
        if not commands:
            errors.append(f"{milestone} needs a command")
        for availability, command in commands:
            if availability == "planned":
                continue
            tokens = shlex.split(command)
            if len(tokens) < 3 or tokens[:2] != ["python", "-m"] or not tokens[2].startswith("annotation.league."):
                errors.append(f"{milestone} command must name a League module")
                continue
            module = tokens[2]
            if importlib.util.find_spec(module) is None:
                errors.append(f"{milestone} command module is missing")
                continue
            if module not in help_by_module:
                help_by_module[module] = subprocess.run([sys.executable, "-B", "-m", module, "--help"], cwd=repository,
                                                       capture_output=True, text=True, timeout=15)
            help_result = help_by_module[module]
            if help_result.returncode or (len(tokens) > 3 and tokens[3] not in help_result.stdout):
                errors.append(f"{milestone} command is unavailable")
        if fields.get("Status") == "complete":
            if "- [ ]" in body or not LINK.search(fields.get("Evidence", "")):
                errors.append(f"{milestone} needs completed checks and evidence")
            receipt = re.fullmatch(r"([0-9a-f]{64}) (\S+)", fields.get("Receipt", ""))
            target = (path.parent / receipt[2]).resolve() if receipt else path
            if not receipt or not target.is_relative_to(repository) or not target.is_file() or file_hash(target) != receipt[1]:
                errors.append(f"{milestone} completion receipt is missing or changed")
    return {"passed": not errors, "milestones": len(headings), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, default=Path("docs/league-body-rendering-roadmap.md"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate(args.roadmap, args.repo)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
