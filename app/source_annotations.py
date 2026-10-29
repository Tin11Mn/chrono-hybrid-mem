"""Bounded model references to immutable, contiguous source witnesses."""
import json
import re
from typing import Any, Dict, List, Tuple

PAGE_CHARS = 3600
MAX_PAGES = 4
RECORDS_PER_PAGE = 32

# CJK punctuation ends a sentence without requiring following whitespace.
CLAUSE_BOUNDARY = r'(?<=[.!?;])\s+|(?<=[。！？；])|\n+'


def first_person_value_witness(span: str, value: str) -> bool:
    """Require a direct self statement in this value's own clause."""
    if not value or value not in span:
        return False
    boundaries = list(re.finditer(CLAUSE_BOUNDARY, span))
    for occurrence in re.finditer(re.escape(value), span):
        left, right = 0, len(span)
        for boundary in boundaries:
            if boundary.end() <= occurrence.start():
                left = boundary.end()
            elif boundary.start() >= occurrence.end():
                right = boundary.start()
                break
        clause = span[left:right].strip()
        if re.match(r'^I\b', clause, re.I):
            if re.match(r'^I\s+(?:know|think|believe|heard|hear|read|was told|wonder|hope|suspect|said|reported|learned|understand)\b', clause, re.I):
                return False
            if re.search(r'\b(?:that|who|whose|whom)\b', clause[:clause.find(value)], re.I):
                return False
        elif re.match(r'^my\b', clause, re.I):
            if re.match(r'^my\s+(?:(?:best|close|former|current|old)\s+)?(?:colleague|co-worker|coworker|friend|boss|manager|parent|mother|father|brother|sister|partner|spouse|child|son|daughter|teacher|doctor|team|company|employer|organization)\b', clause, re.I):
                return False
        elif re.match(r'^我(?:的)?', clause):
            if re.match(r'^我(?:们|国|方|知道|听说|认为|觉得|得知|相信|听到|了解到|记得|想知道|希望)', clause):
                return False
            if re.match(r'^我(?:的)?(?:同事|朋友|老板|经理|父亲|母亲|爸爸|妈妈|哥哥|弟弟|姐姐|妹妹|丈夫|妻子|孩子|儿子|女儿|老师|医生|团队|公司|单位)', clause):
                return False
        else:
            return False
    return True


def compact_repair_feedback(packet: Dict[str, Any], page: str) -> Dict[str, Any]:
    """Bound advisory feedback; immutable page text is supplied separately."""
    feedback = {'complete': packet.get('complete') is True, 'records': [], 'omitted_records': 0}
    records = packet.get('records', [])
    if not isinstance(records, list):
        return feedback
    budget = min(8192, max(2048, len(page) * 2))
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        compact = {key: record[key] for key in
                   ('kind', 'subject', 'attribute', 'scope', 'modality', 'effective_date') if key in record}
        value = record.get('value', '')
        compact['value'] = value[:512] if isinstance(value, str) else ''
        if isinstance(value, str) and len(value) > 512:
            compact['value_truncated'] = True
        compact['unit_ids'] = ['all']
        proposed = {**feedback, 'records': feedback['records'] + [compact]}
        if len(json.dumps(proposed, ensure_ascii=False)) > budget - 64:
            feedback['omitted_records'] = len(records) - index
            break
        feedback['records'].append(compact)
    return feedback


def annotation_schema(content: str, whole_page: bool = False) -> Dict[str, Any]:
    ids = ['all'] + [unit['id'] for unit in source_units(content)]
    properties = {
        'kind': {'type':'string','enum':['fact','state','rule','observation','event','link','privacy']},
        'subject': {'type':'string','minLength':1},
        'attribute': {'type':'string','minLength':1},
        'value': {'type':'string','minLength':1},
        'unit_ids': {'type':'array','items':{'type':'string','enum':['all'] if whole_page else ids},'minItems':1,
                     **({'maxItems':1} if whole_page else {})},
        'scope': {'type':'string'},
        'modality': {'type':'string','enum':['assert','update','correction','retraction','future','hypothetical','exception']},
        'effective_date': {'type':['string','null']},
    }
    return {'type':'object','additionalProperties':False,
        'properties':{'facts':{'type':'array','items':{'type':'string'},'maxItems':32},
            'complete':{'type':'boolean'},
            'records':{'type':'array','maxItems':32,'items':{'type':'object',
                'additionalProperties':False,'properties':properties,'required':list(properties)}},
            'ignored_unit_ids':{'type':'array','items':{'type':'string','enum':ids[1:] or ['all']}}},
        'required':['facts','complete','records','ignored_unit_ids']}


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
    # Keep semicolon-attached qualifications with their assertion, e.g.
    # "might happen; this is only a plan". Numeric rows are parsed separately.
    boundaries = [0] + [match.end() for match in re.finditer(r'(?<=[.!?])\s+|(?<=[。！？])|\n+', content)] + [len(content)]
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
    uses_units = 'ignored_unit_ids' in payload or any(isinstance(r, dict) and 'unit_ids' in r for r in records)
    covered_units = set()
    for item in records[:RECORDS_PER_PAGE]:
        if not isinstance(item, dict):
            complete = False
            continue
        item = dict(item)
        refs = item.pop('unit_ids', None)
        if refs is not None:
            if refs == ['all']:
                span = content
                covered_units.update(by_id)
            elif (isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in by_id for ref in refs)
                  and len(set(refs)) == len(refs)):
                indices = [by_id[ref] for ref in refs]
                if indices != list(range(indices[0], indices[-1] + 1)):
                    complete = False
                    continue
                span = content[units[indices[0]]['start']:units[indices[-1]]['end']]
                covered_units.update(refs)
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
        if item.get('effective_date') == '':
            item['effective_date'] = None  # Empty optional field asserts no date.
        date_scope = re.fullmatch(r'(?:on|as of|effective(?: on)?)\s+(\d{4}-\d{2}-\d{2})', str(item.get('scope', '')), re.I)
        if date_scope and isinstance(span, str) and date_scope[1] in span:
            item['scope'] = ''
            item.setdefault('effective_date', date_scope[1])
            if item['effective_date'] is None:
                item['effective_date'] = date_scope[1]
        if item.get('kind') == 'rule' and isinstance(span, str) and str(item.get('subject', '')) not in span:
            header = re.match(r'^\s*For\s+([^,:\n]{1,100})[,,:]', span, re.I)
            subject = str(item.get('subject', ''))
            if header and subject.casefold().startswith(header[1].casefold() + ' '):
                suffix = subject[len(header[1]):].strip()
                if re.fullmatch(r'(?:(?:pricing|shipping|handling)\s+)?(?:policy|rules|procedure)', suffix, re.I):
                    item['subject'] = header[1]
        if item.get('kind') in {'update', 'correction', 'retraction'}:
            item.setdefault('modality', item['kind'])
            item['kind'] = 'state'
        value = item.get('value')
        # Captured provider grammar variant: "Device status" / "is now".
        # Canonicalize only an explicit literal attribute, never a guessed alias.
        status_subject = re.fullmatch(r'(.+?)\s+(status|state)', str(item.get('subject', '')), re.I)
        if status_subject and re.fullmatch(r'(?:is|was|is now|was previously)', str(item.get('attribute', '')), re.I):
            item['subject'], item['attribute'] = status_subject[1], status_subject[2].casefold()
        # A categorical label cannot turn a nonnumeric attribute into a numeric
        # observation. Downgrade to a generic assertion; do not invent measures.
        if item.get('kind') == 'observation' and isinstance(value, str) and not re.search(r'\d', value):
            item['kind'] = 'fact'
        if item.get('kind') == 'event' and str(item.get('attribute', '')).casefold() in {
                'status', 'state', 'preference', 'location', 'affiliation'}:
            item['kind'] = 'fact'
        if isinstance(span, str) and isinstance(value, str) and value and span.count(value) == 1:
            point = span.index(value)
            left = 0
            right = len(span)
            for boundary in re.finditer(CLAUSE_BOUNDARY, span):
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
        if item.get('kind') in {'fact', 'event', 'observation'} and item.get('modality') in {
                'update', 'correction', 'retraction'}:
            item['kind'] = 'state'
        decoded.append(item)
    ignored = payload.get('ignored_unit_ids', [])
    if uses_units:
        if (not isinstance(ignored, list) or any(not isinstance(i, str) or i not in by_id for i in ignored)
                or len(set(ignored)) != len(ignored)):
            complete = False
            ignored = []
        safe_ignored = []
        for ref in ignored:
            # A redundant "ignored" label cannot erase a supplied witness.
            # The record still has to pass subject/value/span grounding below.
            if ref in covered_units:
                continue
            text = units[by_id[ref]]['text'].strip()
            if text.endswith(('?', '？')) or re.fullmatch(
                    r'(?:hi|hello|hey|thanks|thank you|ok|okay|你好|谢谢|好的)[.!。！\s]*', text, re.I):
                safe_ignored.append(ref)
            else:
                complete = False
        ignored = safe_ignored
        uncovered = [i for i in by_id if i not in covered_units and i not in ignored]
        if uncovered:
            complete = False
        result['uncovered_unit_ids'] = uncovered
    # Literal measurement tables often arrive as separate x/y annotations.
    # Reconstruct only explicitly labelled observed rows, never an induced law.
    if any(r.get('kind') == 'observation' for r in decoded):
        header = re.match(r'^\s*([\w .-]{1,100}?)\s+(?:observations|measurements|readings|data)\s*:\s*', content, re.I)
        if header and len(re.findall(r'\b(?:observations|measurements|readings|data)\s*:', content, re.I)) == 1:
            from .structured_memory import CONDITION
            measured = []
            for chunk in re.split(r';\s*|(?<=[.!?。！？])\s+|\n+', content[header.end():]):
                matches = [m for m in CONDITION.finditer(chunk) if m[2] in {'=', '=='} and re.fullmatch(r'-?\d+(?:\.\d+)?', m[3])]
                if len(matches) < 2 or len({m[1] for m in matches}) != len(matches):
                    continue
                if re.search(r'\b(if|unless|suppose|might|could)\b|假设|如果', chunk, re.I):
                    continue
                literal = chunk[matches[0].start():matches[-1].end()]
                measured.append({'kind':'observation','subject':header[1], 'attribute':'measurements',
                    'value':literal,'span':content,'scope':'','modality':'assert','conditions':[]})
            if measured:
                decoded = [r for r in decoded if r.get('kind') != 'observation'] + measured
    result.update(records=decoded, complete=complete)
    return result
