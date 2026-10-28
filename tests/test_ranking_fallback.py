"""Ranking confidence and evidence completeness are independent contracts."""
import json
import pytest

from app.evidence_selection import validated_ranking
from app.model import MemoryModel
from app.schemas import AddRequest, Message
from app.storage import MemoryStore


class PartialEvidenceModel:
    def __init__(self, *, mode='incomplete', explicit=True):
        self.mode, self.explicit = mode, explicit
        self.calls = 0
        self.expected = []

    def extract_facts(self, content, **kwargs):
        return []

    def plan_query_structured(self, query, options):
        return {'intent': 'temporal', 'core_terms': ['Project'], 'entities': [],
                'expansion_terms': [], 'temporal_cues': [], 'evidence_needs': ['Project chronology']}

    def select_evidence(self, query, options, candidates, plan):
        self.calls += 1
        self.expected = [item['id'] for item in reversed(candidates)]
        payload = {'complete': self.mode != 'incomplete',
                   'selected_ids': self.expected[:1], 'excluded_ids': self.expected[1:],
                   'covered_need_indices': [], 'state_updates': []}
        if self.explicit:
            payload['ordered_ids'] = self.expected
        else:
            self.expected = self.expected[:1]
        return payload


@pytest.mark.parametrize('mode', ['incomplete', 'missing_coverage'])
@pytest.mark.parametrize('explicit', [True, False])
def test_rejected_selection_preserves_ranking_and_all_legacy_sources(tmp_path, mode, explicit):
    model = PartialEvidenceModel(mode=mode, explicit=explicit)
    store = MemoryStore(str(tmp_path/'fallback.db'), model=model,
                        structured_query_plan=True, evidence_selection=True)
    store.initialize()
    for i in range(4):
        store.add(AddRequest(request_id=str(i), user_id='u', session_id='s',
                            messages=[Message(role='user', content=f'Project phase {i} happened on day {i}.')]))
    results = store.search(user_id='u', query='Project chronology', top_k=100)
    assert [r.id for r in results][:len(model.expected)] == model.expected
    assert {r.id for r in results} == {'mem_1', 'mem_2', 'mem_3', 'mem_4'}
    assert model.calls == 1
    assert store.last_retrieval_trace['evidence_selection']['applied'] is False
    assert store.last_retrieval_trace['evidence_selection']['fallback_order_applied'] is True
    results = store.search(user_id='u', query='Project chronology', top_k=1)
    assert [r.id for r in results] == model.expected[:1]


@pytest.mark.parametrize('payload', [None, [], {'ordered_ids': 'mem_1'},
    {'ordered_ids': ['mem_1', 'other-user']}, {'ordered_ids': ['mem_1', 'mem_1']},
    {'ordered_ids': [1]}, {'ordered_ids': [True]},
    {'ordered_ids': None, 'selected_ids': ['mem_1']}])
def test_malformed_order_cannot_gain_ranking_authority(payload):
    assert validated_ranking(payload, ['mem_1', 'mem_2']) == []


def test_partial_order_is_valid_without_coverage_or_state_authority():
    payload = {'complete': False, 'ordered_ids': ['mem_2', 'mem_1'],
               'excluded_ids': ['mem_1'], 'state_updates': [{'invented': 'claim'}]}
    assert validated_ranking(payload, ['mem_1', 'mem_2', 'mem_3']) == ['mem_2', 'mem_1']
    assert validated_ranking({'selected_ids': ['mem_2']}, ['mem_1', 'mem_2']) == ['mem_2']


def test_selector_requests_ranking_even_without_complete_coverage(monkeypatch):
    model = object.__new__(MemoryModel)
    calls = []
    def response(prompt, payload, **kwargs):
        calls.append((prompt, payload, kwargs))
        return {'complete': False, 'ordered_ids': ['mem_1']}
    monkeypatch.setattr(model, '_json_response', response)
    assert model.select_evidence('when', [], [{'id':'mem_1','content':'original'}], {})['ordered_ids'] == ['mem_1']
    assert len(calls) == 1
    assert 'EVEN when complete:false' in calls[0][0]
    assert 'Do NOT preserve candidate input order' in calls[0][0]


@pytest.mark.parametrize('mode', ['specialist', 'coverage_only', 'invalid_order', 'error'])
def test_ranking_fallback_does_not_publish_sidecars_or_remove_sources(tmp_path, monkeypatch, mode):
    class Model(PartialEvidenceModel):
        def select_evidence(self, query, options, candidates, plan):
            payload = super().select_evidence(query, options, candidates, plan)
            if mode == 'error':
                raise RuntimeError('private source error')
            if mode == 'invalid_order':
                payload['ordered_ids'] = ['other-user-source']
            if mode == 'specialist':
                payload.update(complete=True, covered_need_indices=[0])
            if mode == 'coverage_only':
                assert 'mem_3' in payload['ordered_ids']
            return payload
    model = Model()
    trace = tmp_path/'redacted.jsonl'
    store = MemoryStore(str(tmp_path/'sidecar.db'), model=model, structured_query_plan=True,
                        evidence_selection=True, coverage_retrieval=mode == 'coverage_only',
                        structured_memory=mode == 'specialist', retrieval_trace_path=str(trace))
    store.initialize()
    for i, text in enumerate(['Project first original.', 'Project second original.', 'Hidden unrelated archive.']):
        store.add(AddRequest(request_id=str(i), user_id='u', session_id='archive' if i == 2 else 's',
                            messages=[Message(role='user',content=text)]))
    # Isolate coverage-only membership from ordinary lexical retrieval.
    if mode == 'coverage_only':
        monkeypatch.setattr(store, '_coverage_candidates', lambda *args: [dict(
            id=3,content='Hidden unrelated archive.',created_at='2024-01-01T00:00:00+00:00',event_ts=None)])
    if mode == 'specialist':
        from app.structured_memory import StructuredMemoryIndex
        monkeypatch.setattr(StructuredMemoryIndex,'retrieve', lambda *args: {
            'hints': [], 'required_groups': [['mem_1','mem_2']], 'coverage_limited':False,
            'index_complete':False, 'blocked_ids':[], 'diagnostics':{}, 'rows':[]})
    results = store.search(user_id='u',query='Project chronology',top_k=100)
    ids = [r.id for r in results]
    assert set(ids) == {'mem_1','mem_2'}
    decision = store.last_retrieval_trace['evidence_selection']
    assert decision['applied'] is False
    if mode == 'specialist':
        assert decision['reason'] == 'specialist_obligation_missing'
    if mode in {'specialist','coverage_only'}:
        assert ids == [i for i in model.expected if i in {'mem_1','mem_2'}]
        assert decision['fallback_order_applied'] is True
    else:
        assert not decision['fallback_order_applied']
    record = json.loads(trace.read_text())
    assert 'Hidden' not in json.dumps(record) and 'private' not in json.dumps(record)
    assert record['final_ids'] == ids
