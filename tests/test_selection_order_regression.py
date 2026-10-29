"""Sufficient membership must not discard independently validated relevance order."""
from app.schemas import AddRequest, Message
from app.storage import MemoryStore


class CompleteModel:
    def extract_facts(self, content, **kwargs):
        return []
    def plan_query_structured(self, query, options):
        return {'intent':'temporal','core_terms':['Juniper'],'evidence_needs':['Juniper update history']}
    def select_evidence(self, query, options, candidates, plan):
        return {'complete':True,'ordered_ids':['mem_2','mem_1'],
                'selected_ids':['mem_1','mem_2'],'excluded_ids':[],
                'covered_need_indices':[0],'state_updates':[]}


def test_complete_selection_keeps_latest_evidence_first_before_top_k(tmp_path):
    store=MemoryStore(str(tmp_path/'selection.db'),model=CompleteModel(),
                      structured_query_plan=True,evidence_selection=True)
    store.initialize()
    for i,text in enumerate(['Juniper status was pending.','Juniper status is now approved.']):
        store.add(AddRequest(request_id=str(i),user_id='u',session_id='s',
                             messages=[Message(role='user',content=text)]))
    results=store.search(user_id='u',query='What is the latest Juniper status?',top_k=1)
    assert store.last_retrieval_trace['evidence_selection']['applied']
    assert [r.id for r in results]==['mem_2']


def test_diagnostics_surface_empty_candidates_without_free_query_text(tmp_path):
    import json
    from app.evidence_selection import append_redacted_trace
    target=tmp_path/'trace.jsonl'
    append_redacted_trace(str(target),{'plan':{'intent':'summary','query':'private secret'},
        'rerank_pool_ids':['mem_1'],'final_ids':[]})
    row=json.loads(target.read_text())
    assert row['query_intent']=='summary' and row['empty_after_candidates']
    assert 'private' not in target.read_text() and 'secret' not in target.read_text()
    append_redacted_trace(str(target),{'plan':{'intent':['private secret']}})
    rows=target.read_text().splitlines()
    assert 'query_intent' not in json.loads(rows[-1])
