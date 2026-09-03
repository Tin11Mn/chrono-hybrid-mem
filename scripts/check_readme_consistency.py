"""Validate facts and local links shared by the multilingual README files."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

README_LABELS = {
    "README.md": "简体中文",
    "README.en.md": "English",
    "README.es.md": "Español",
    "README.ja.md": "日本語",
    "README.ko.md": "한국어",
}

FACT_MARKER = (
    "<!-- README_FACTS: main=stable-post-submission-research; "
    "p3=experimental; official-v020-mapping=CONFIRMED -->"
)

COMMON_LITERALS = (
    "Rank 5",
    "Overall 44.33",
    "CONFIRMED",
    "7cf45c76ea7998554a13386b924627b83aeb3134",
    "1,977",
    "0.5761",
    "0.7157",
    "0.7618",
    "0.6479",
    "v0.1.0",
    "v0.2.0",
    "research-v0.3.0",
    "research-v0.4.0",
    "research-p1-20260816",
    "research/p3-evidence-graph",
    "MEMORY_STRUCTURED_QUERY_PLAN=true",
    "--structured-query-plan",
    "requirements.txt",
    "requirements-test.txt",
    "requirements-local.txt",
    "MIT",
    "POST /add",
    "POST /search",
)

TAG_TARGETS = {
    "v0.1.0": "5fd77045c74a5b17876abca30812888587628eaa",
    "v0.2.0": "7cf45c76ea7998554a13386b924627b83aeb3134",
    "research-v0.3.0": "3a0ba8c06722fe53b07d8d251ada1729c390bcdc",
    "research-v0.4.0": "1b013b93a0f3e6e12366208f20eae1d245889909",
    "research-p1-20260816": "0691afafe4cede21f973efb996b86a29d441ff88",
}

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=\"([^\"]+)\"")
FENCED_CODE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
NUMBER_TOKEN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")


def _git(*args: str, root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _local_targets(text: str) -> Iterable[str]:
    for match in MARKDOWN_LINK.finditer(text):
        yield match.group(1)
    for match in HTML_LINK.finditer(text):
        yield match.group(1)


def validate_readmes(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    texts: dict[str, str] = {}

    for name in README_LABELS:
        path = root / name
        if not path.is_file():
            errors.append(f"missing README: {name}")
            continue
        texts[name] = path.read_text(encoding="utf-8")

    logo = root / "assets" / "chronohybridmem-logo.png"
    if not logo.is_file():
        errors.append("missing shared logo: assets/chronohybridmem-logo.png")

    canonical = texts.get("README.en.md", "")
    canonical_without_fences = FENCED_CODE.sub("", canonical)
    expected_code_blocks = FENCED_CODE.findall(canonical)
    expected_inline_code = Counter(INLINE_CODE.findall(canonical_without_fences))
    expected_numbers = Counter(NUMBER_TOKEN.findall(canonical_without_fences))
    language_files = set(README_LABELS)
    expected_content_links = sorted(
        target
        for target in _local_targets(canonical)
        if target.split("#", 1)[0] not in language_files
    )

    for name, text in texts.items():
        if FACT_MARKER not in text:
            errors.append(f"{name}: missing machine-readable fact marker")
        if "assets/chronohybridmem-logo.png" not in text:
            errors.append(f"{name}: missing shared logo link")
        if re.search(r"[A-Za-z]:[\\/]Users[\\/]", text):
            errors.append(f"{name}: contains an absolute local path")

        text_without_fences = FENCED_CODE.sub("", text)
        if FENCED_CODE.findall(text) != expected_code_blocks:
            errors.append(f"{name}: fenced code blocks differ from README.en.md")
        if Counter(INLINE_CODE.findall(text_without_fences)) != expected_inline_code:
            errors.append(f"{name}: inline code tokens differ from README.en.md")
        if Counter(NUMBER_TOKEN.findall(text_without_fences)) != expected_numbers:
            errors.append(f"{name}: numeric tokens differ from README.en.md")
        content_links = sorted(
            target
            for target in _local_targets(text)
            if target.split("#", 1)[0] not in language_files
        )
        if content_links != expected_content_links:
            errors.append(f"{name}: content link targets differ from README.en.md")

        for literal in COMMON_LITERALS:
            if literal not in text:
                errors.append(f"{name}: missing shared literal {literal!r}")

        current_label = README_LABELS[name]
        if f"<strong>{current_label}</strong>" not in text:
            errors.append(f"{name}: current language is not bold in navigation")
        for target_name, label in README_LABELS.items():
            if target_name == name:
                continue
            expected = f'<a href="{target_name}">{label}</a>'
            if expected not in text:
                errors.append(f"{name}: missing language navigation {expected}")

        for target in _local_targets(text):
            clean_target = target.split("#", 1)[0]
            if not clean_target or clean_target.startswith(
                ("http://", "https://", "mailto:")
            ):
                continue
            resolved = (root / clean_target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{name}: local link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{name}: broken local link: {target}")

    for tag, expected_target in TAG_TARGETS.items():
        try:
            if _git("cat-file", "-t", tag, root=root) != "tag":
                errors.append(f"{tag}: must remain an annotated tag")
            actual_target = _git("rev-list", "-n", "1", tag, root=root)
            if actual_target != expected_target:
                errors.append(
                    f"{tag}: points to {actual_target}, expected {expected_target}"
                )
        except RuntimeError as error:
            errors.append(f"{tag}: cannot verify tag ({error})")

    return errors


def main() -> int:
    errors = validate_readmes()
    if errors:
        print("README consistency check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("README consistency check passed for 5 languages and 5 annotated tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
