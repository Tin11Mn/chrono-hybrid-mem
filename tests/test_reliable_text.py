"""Independent source-witness fixtures for the reliable text release."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import pytest
from app.model import MemoryModel
from app.source_annotations import decode_annotations, source_pages, source_units
from app.structured_memory import grounded_records, observation_summary, resolve_states
from app.storage import MemoryStore
from app.schemas import AddRequest, Message


def packet(subject, value, *, kind='fact', units=None, **kwargs):
    return {'complete':True,'facts':[], 'records':[{'kind':kind,'subject':subject,
        'attribute':'status','value':value,'unit_ids':units or ['all'],'modality':'assert',**kwargs}]}


def test_unit_references_copy_literal_context_and_reject_invented_spans():
    source='Probe input=1, output=7; input=2, output=11; input=3, output=15.'
    assert all(unit['text'] == source[unit['start']:unit['end']] for unit in source_units(source))
    p=packet('Probe','input=2, output=11',kind='observation')
    p['records'][0]['span']='Probe input=2, output=11'  # Generated text is ignored.
    decoded=decode_annotations(p,source)
    records, complete=grounded_records(decoded,source,'user',100)
    assert complete and records[0]['span'] == source
    assert records[0]['value'] == 'input=2, output=11'
    for ids in [['u1','u3'],['foreign'],['all','u1'],['u1','u1'],[True]]:
        p['records'][0]['unit_ids']=ids
        assert not decode_annotations(p,source)['complete']


def test_first_person_witness_and_independent_modalities_in_one_page():
    source='My name is Mara. I prefer quiet rooms and decaf tea. I might move next year.'
    p=packet('@speaker','quiet rooms',units=['u2'])
    records,complete=grounded_records(decode_annotations(p,source),source,'user',100)
    assert not complete and records[0]['subject']=='user' and records[0]['modality']=='assert'
    # Full-page context must not turn an unrelated literal preference into a plan.
    p['records'][0]['unit_ids']=['all']
    records,complete=grounded_records(decode_annotations(p,source),source,'user',100)
    assert complete and records[0]['modality']=='assert'


def test_explicit_transition_and_dates_survive_shared_source():
    source='AsterCase was pending on 2024-04-01. Correction: AsterCase status is now approved, effective 2024-04-12. AsterCase might be archived next year.'
    p=packet('AsterCase','pending')
    p['records'].append(deepcopy(p['records'][0]))
    p['records'][1]['value']='approved'
    records,complete=grounded_records(decode_annotations(p,source),source,'user',1714000000)
    assert complete and [r['modality'] for r in records]==['assert','correction']
    assert [r['effective_ts'] for r in records]==[
        int(datetime(2024,4,1,tzinfo=timezone.utc).timestamp()),
        int(datetime(2024,4,12,tzinfo=timezone.utc).timestamp())]
    for r in records:r.update(source_id=1)
    assert resolve_states(records)['status']=='resolved'


def test_natural_language_rules_are_evidence_without_invented_executable_form():
    source='ParcelDock packages at most 12 kg cost 8 credits. Fragile packages add 3 credits, except emergency medical packages.'
    p=packet('ParcelDock',source,kind='rule',conditions=['weight<=12'])
    records,complete=grounded_records(decode_annotations(p,source),source,'user',100)
    assert complete and records[0]['conditions']==[]
    assert 'except emergency medical' in records[0]['span']


def test_measurement_rows_under_shared_heading_keep_counterexample():
    source='ProbeRig input=1, output=7; input=2, output=11; input=3, output=15; input=4, output=20.'
    p=packet('ProbeRig','input=1, output=7',kind='observation')
    for value in ['input=2, output=11','input=3, output=15','input=4, output=20']:
        p['records'].append({**p['records'][0],'value':value})
    rows,complete=grounded_records(decode_annotations(p,source),source,'user',100)
    assert complete and len(rows)==4
    for row in rows:row['source_id']=1
    summary=observation_summary(rows)
    assert summary['statistics']['input']['count']==4
    assert summary['statistics']['output']['max']=='20'
    assert summary['exact_affine_fit'] is None


def test_literal_observation_table_recovers_split_input_output_without_a_law():
    text='MeterB observations: u=1, v=8; u=2, v=12; u=3, v=17.'
    p=packet('MeterB','1',kind='observation')
    rows,complete=grounded_records(decode_annotations(p,text),text,'user',100)
    assert complete and len(rows)==3
    for row in rows:row['source_id']=1
    summary=observation_summary(rows)
    assert summary['statistics']['u']['count']==3 and summary['exact_affine_fit'] is None


def test_pages_are_bounded_literal_slices_and_never_claim_truncated_coverage():
    source=('Aster notes are literal. '*1000)
    pages,complete=source_pages(source)
    assert len(pages)==4 and not complete
    assert ''.join(pages)==source[:sum(map(len,pages))]
    assert all(len(page)<=3600 for page in pages)


def test_extractor_repairs_partial_page_once_without_weakening_validation(monkeypatch):
    model=object.__new__(MemoryModel)
    calls=[]
    bad=packet('UnknownName','approved')
    good=packet('AsterCase','approved')
    def response(prompt,payload,**kwargs):
        calls.append(payload)
        return deepcopy(bad if len(calls)==1 else good)
    monkeypatch.setattr(model,'_json_response',response)
    result=model.extract_structured_memory('AsterCase status approved.','user',100)
    assert result['complete'] and result['repair_calls']==1 and len(calls)==2
    assert calls[1]['repair_attempt'] is True
    calls.clear()
    monkeypatch.setattr(model,'_json_response',lambda *a,**k:bad)
    assert not model.extract_structured_memory('AsterCase status approved.','user',100)['complete']


def test_no_relative_or_deadline_date_is_invented():
    source='AsterCase deadline is 2030-02-01. AsterCase status is now approved.'
    p=packet('AsterCase','approved',units=['u2'])
    row=decode_annotations(p,source)['records'][0]
    assert row['modality']=='update' and 'effective_date' not in row


@pytest.mark.parametrize('kind',['event','observation'])
def test_misclassified_status_weakens_to_fact_and_witnessed_transition(kind):
    old='AsterCase status is pending.'
    new='AsterCase status is now approved.'
    first=decode_annotations(packet('AsterCase','pending',kind=kind),old)
    second=decode_annotations(packet('AsterCase','approved',kind=kind),new)
    rows1,valid1=grounded_records(first,old,'user',100)
    rows2,valid2=grounded_records(second,new,'user',200)
    assert valid1 and valid2 and rows1[0]['kind']=='fact' and rows2[0]['kind']=='state'
    assert rows1[0]['group_id']==rows2[0]['group_id']
    rows1[0]['source_id']=1;rows2[0]['source_id']=2
    assert resolve_states(rows1+rows2)['required']==[2]


def test_whole_quoted_witness_is_not_an_effective_update():
    source='"AsterCase status is now approved."'
    rows,valid=grounded_records(decode_annotations(packet('AsterCase','approved'),source),source,'user',100)
    assert valid and rows[0]['modality']=='hypothetical'


@pytest.mark.parametrize('count',[40,100])
def test_summary_history_bundles_expand_but_never_claim_truncated_coverage(tmp_path,count):
    class Model:
        def plan_query_structured(self,*args):
            raise AssertionError('indexed planner must be used')
        def extract_structured_memory(self,content,**kwargs):
            return {'complete':True,'facts':[], 'records':[{
                'kind':'fact','subject':'Aster','attribute':'milestone','value':content,
                'span':content,'scope':'','modality':'assert','conditions':[]}]}
        def plan_indexed_query(self,query,options,catalog):
            return {'intent':'summary','core_terms':['Aster'],'entities':['Aster'],
                'expansion_terms':[],'evidence_needs':[], 'temporal_cues':[],
                'memory_requests':{'summary_topics':['aster'],'temporal_mode':'history'}}
        def select_evidence(self,query,options,candidates,plan):
            self.pool=candidates
            ids=list(dict.fromkeys(i for group in plan['required_source_groups'] for i in group))
            return {'complete':True,'selected_ids':ids,'covered_need_indices':[]}
    model=Model()
    store=MemoryStore(str(tmp_path/'summary.db'),model=model,structured_query_plan=True,
                      evidence_selection=True,coverage_retrieval=True,structured_memory=True)
    store.initialize()
    store.add(AddRequest(request_id='batch',user_id='u',session_id='s',messages=[
        Message(role='user',content=f'Aster milestone step {i}.',timestamp=1700000000+i) for i in range(count)]))
    rows=store.search(user_id='u',query='Summarize Aster milestones',top_k=100)
    trace=store.last_retrieval_trace
    assert len(model.pool)==min(count,96)
    assert trace['model_pool_limit']==min(count,96)
    assert len(rows)==count
    assert trace['evidence_selection']['applied'] is (count<=96)
    assert trace['structured_memory']['pool_coverage_complete'] is (count<=96)
    if count>96:
        assert trace['evidence_selection']['reason']=='specialist_pool_incomplete'
