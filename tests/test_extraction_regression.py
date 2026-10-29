"""Extraction counterexamples independent of leaderboard questions and answers."""
from copy import deepcopy
import json

import pytest

from app.model import MemoryModel
from app.source_annotations import decode_annotations
from app.structured_memory import grounded_records


def annotation(subject, value, *, complete=True):
    return {
        'complete': complete, 'facts': [], 'ignored_unit_ids': [],
        'records': [{
            'kind': 'fact', 'subject': subject, 'attribute': 'office',
            'value': value, 'unit_ids': ['all'], 'scope': '',
            'modality': 'assert', 'effective_date': None,
        }],
    }


@pytest.mark.parametrize('subject', ['Morrow office', '@speaker'])
def test_third_party_clause_cannot_borrow_first_person_from_another_sentence(subject):
    text = 'I visited Morrow. Its office is in Bergen.'
    rows, complete = grounded_records(
        decode_annotations(annotation(subject, 'Bergen'), text), text, 'request-user', 100,
    )
    assert not complete
    assert not rows


@pytest.mark.parametrize('text,value', [
    ('My colleague Luis works in Bergen.', 'Bergen'),
    ('I know that Luis works in Bergen.', 'Bergen'),
    ('我同事小林住在青岛。', '青岛'),
])
def test_possession_or_reporting_does_not_make_another_person_the_speaker(text, value):
    rows, complete = grounded_records(
        decode_annotations(annotation('@speaker', value), text), text, 'request-user', 100,
    )
    assert not complete and not rows


@pytest.mark.parametrize('text,value', [
    ('My office is in Bergen.', 'Bergen'),
    ('I work in Bergen.', 'Bergen'),
    ('I prefer quiet rooms and decaf tea.', 'quiet rooms'),
    ('我的办公室在青岛。', '青岛'),
])
def test_direct_first_person_claim_remains_indexable(text, value):
    rows, complete = grounded_records(
        decode_annotations(annotation('@speaker', value), text), text, 'request-user', 100,
    )
    assert complete and rows[0]['subject'] == 'request-user'


def test_repair_feedback_does_not_repeat_literal_page_for_every_annotation(monkeypatch):
    text = 'Morrow office is in Bergen. ' + 'Literal background detail. ' * 90
    packet = annotation('Morrow', 'Bergen', complete=False)
    packet['records'] *= 24
    calls = []
    model = object.__new__(MemoryModel)

    def response(prompt, payload, **kwargs):
        calls.append(deepcopy(payload))
        return deepcopy(packet)

    monkeypatch.setattr(model, '_json_response', response)
    model.extract_structured_memory(text, 'request-user', 100)
    assert len(calls) == 2
    feedback = calls[1]['previous_annotation']
    assert all('span' not in record for record in feedback['records'])
    # Feedback is annotations and references, not 24 duplicate source pages.
    assert len(json.dumps(feedback)) < len(text) * 3
    assert all(record['unit_ids'] == ['all'] for record in feedback['records'])


def test_cjk_sentence_boundary_does_not_make_current_state_a_future_hypothesis():
    source = '雾杉项目现在状态改为已批准，2024-06-10生效。明年也许会暂停，但尚未决定。'
    data = annotation('雾杉项目', '已批准')
    data['records'][0].update(attribute='status', modality='update', effective_date='2024-06-10')
    rows, complete = grounded_records(decode_annotations(data, source), source, 'user', 1717977600)
    assert complete and rows[0]['modality'] == 'update'


def test_another_subjects_transition_cue_cannot_authorize_supersession():
    source = 'Morrow status is pending. Kestrel status is now approved.'
    data = annotation('Morrow', 'pending')
    data['records'][0].update(attribute='status', modality='update')
    rows, complete = grounded_records(decode_annotations(data, source), source, 'user', 100)
    assert complete and rows[0]['modality'] == 'assert'


def test_full_page_values_cannot_bypass_feedback_size_bound():
    from app.source_annotations import compact_repair_feedback
    page = 'Literal page text. ' * 180
    data = annotation('Literal', page, complete=False)
    data['records'] *= 24
    feedback = compact_repair_feedback(data, page)
    assert len(json.dumps(feedback, ensure_ascii=False)) <= 8192
    assert feedback['omitted_records'] > 0
    assert all(record['value_truncated'] for record in feedback['records'])
