"""Bounded model references to immutable, contiguous source witnesses."""
import re
from typing import Any, Dict, List, Tuple

PAGE_CHARS = 3600
MAX_PAGES = 4
RECORDS_PER_PAGE = 32


def source_pages(content: str) -> Tuple[List[str], bool]:
    """Prefer sentence boundaries; every page remains a literal source slice."""
    pages = []
    start = 0
    while start < len(content) and len(pages) < MAX_PAGES:
        end = min(start + PAGE_CHARS, len(content))
        if end < len(content):
            breaks = list(re.finditer(r'(?<=[.!?。！？;])\s+|\n+', content[start:end]))
            if breaks and breaks[-1].end() >= PAGE_CHARS // 2:
                end = start + breaks[-1].end()
        pages.append(content[start:end])
        start = end
    return pages or [''], start == len(content)


def source_units(content: str) -> List[Dict[str, Any]]:
    boundaries = [0] + [match.end() for match in re.finditer(r'(?<=[.!?。！？;])\s+|\n+', content)] + [len(content)]
    units = []
    for start, end in zip(boundaries, boundaries[1:]):
        text = content[start:end].strip()
        if text:
            at = content.index(text, start, end)
            units.append({'id': 'u' + str(len(units) + 1), 'text': text, 'start': at, 'end': at + len(text)})
    return units


def decode_annotations(payload: Any, content: str) -> Dict[str, Any]:
    """Copy witnesses by ID. Invalid references cannot certify completeness."""
    if not isinstance(payload, dict):
        return {'records': [], 'facts': [], 'complete': False}
    result = dict(payload)
    units = source_units(content)
    by_id = {unit['id']: i for i, unit in enumerate(units)}
    records = payload.get('records')
    if not isinstance(records, list):
        return {**result, 'records': [], 'complete': False}
    complete = payload.get('complete') is True and len(records) <= RECORDS_PER_PAGE
    decoded = []
    for item in records[:RECORDS_PER_PAGE]:
        if not isinstance(item, dict):
            complete = False
            continue
        item = dict(item)
        refs = item.pop('unit_ids', None)
        if refs is not None:
            if refs == ['all']:
                span = content
            elif (isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in by_id for ref in refs)
                  and len(set(refs)) == len(refs)):
                indices = [by_id[ref] for ref in refs]
                if indices != list(range(indices[0], indices[-1] + 1)):
                    complete = False
                    continue
                span = content[units[indices[0]]['start']:units[indices[-1]]['end']]
            else:
                complete = False
                continue
            item['span'] = span
            item.setdefault('scope', '')
            item.setdefault('value', span)
            # Symbolic premises come only from source literals, never generated
            # translations of natural-language policy. Execution stays conservative.
            from .structured_memory import CONDITION
            item['conditions'] = ([match.group() for match in CONDITION.finditer(span)]
                                  if item.get('kind') == 'rule' else [])
        # Compatibility with older providers is still subject to normal grounding.
        span = item.get('span', '')
        if item.get('kind') in {'update', 'correction', 'retraction'}:
            item.setdefault('modality', item['kind'])
            item['kind'] = 'state'
        value = item.get('value')
        if isinstance(span, str) and isinstance(value, str) and value and span.count(value) == 1:
            point = span.index(value)
            left = 0
            right = len(span)
            for boundary in re.finditer(r'(?<=[.!?。！？;])\s+|\n+', span):
                if boundary.end() <= point:
                    left = boundary.end()
                elif boundary.start() >= point:
                    right = boundary.start()
                    break
            clause = span[left:right]
            if item.get('kind') in {'fact', 'state'} and item.get('modality', 'assert') == 'assert':
                if re.search(r'\b(correction|corrected|actually|instead)\b|更正|纠正|改为', clause, re.I):
                    item['modality'] = 'correction'
                elif re.search(r'\b(now|changed|updated|effective)\b|现在|更新|改为|生效', clause, re.I):
                    item['modality'] = 'update'
            dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', clause)
            if len(set(dates)) == 1 and item.get('effective_date') is None:
                if item.get('kind') == 'event' or re.search(
                        r'\beffective\b|\bwas\b.*\bon\b|生效|曾于', clause, re.I):
                    item['effective_date'] = dates[0]
        decoded.append(item)
    result.update(records=decoded, complete=complete)
    return result
