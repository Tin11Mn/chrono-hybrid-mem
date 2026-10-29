"""Evidence validity is distinct from a source asserting a current state."""
import sqlite3
import pytest
from app.structured_memory import StructuredMemoryIndex, grounded_records, resolve_states, route_memory_requests


def packet(text, subject, value, *, mode='assert', complete=True, kind='state', attribute='status'):
    return {'complete': complete, 'records': [{'kind':kind, 'subject':subject, 'attribute':attribute,
        'value':value, 'span':text, 'scope':'', 'modality':mode, 'conditions':[]}]}


def database():
    db=sqlite3.connect(':memory:'); db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE raw_messages(id INTEGER PRIMARY KEY,user_id TEXT,session_id TEXT,role TEXT,content TEXT,created_at TEXT,event_ts INTEGER)')
    StructuredMemoryIndex.initialize(db)
    return db


def add(db, sid, text, payload=None):
    db.execute('INSERT INTO raw_messages VALUES(?,?,?,?,?,?,?)',(sid,'u','s','user',text,'2024-01-01',sid))
    if payload is not None:
        StructuredMemoryIndex.store(db,user_id='u',session_id='s',source_id=sid,content=text,speaker='user',timestamp=sid,payload=payload)


def retrieve(db):
    catalog=StructuredMemoryIndex.catalog(db,'u','Atlas status')
    slot=next(r['group_id'] for r in catalog['groups'] if r['subject']=='atlas' and r['attribute']=='status')
    return StructuredMemoryIndex.retrieve(db,'u','"Atlas"','What is Atlas current status?',
        {'intent':'governance','memory_requests':{'state_slots':[slot],'temporal_mode':'current'}})


@pytest.mark.parametrize('mode,text', [('hypothetical','Maybe Atlas status is approved.'),('future','Atlas status will be approved next year.')])
def test_unestablished_state_keeps_original_qualifications_as_evidence(mode,text):
    records, complete=grounded_records(packet(text,'Atlas','approved',mode=mode),text,'user',100)
    assert complete
    records[0].update(source_id=1)
    result=resolve_states(records)
    assert result['status']=='unresolved' and result['required']==[1] and result['blocked']==[]


def test_partial_unrelated_source_does_not_force_stale_slot_values():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    text='Borealis color is blue.'
    add(db,3,text,packet(text,'Borealis','blue',complete=False,kind='fact',attribute='color'))
    result=retrieve(db)
    assert result['required_groups']==[['mem_2']] and result['blocked_ids']==['mem_1']


@pytest.mark.parametrize('uncertain', ['unindexed','empty','same_subject','pronoun'])
def test_uncertain_source_cannot_certify_slot_complete(uncertain):
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    text={'unindexed':'A later report.','empty':'A later report.','same_subject':'Atlas color is blue.','pronoun':'Borealis is blue. It may be ready.'}[uncertain]
    payload=None if uncertain=='unindexed' else ({'complete':False,'records':[]} if uncertain=='empty' else
        packet(text,'Atlas' if uncertain=='same_subject' else 'Borealis','blue',complete=False,kind='fact',attribute='color'))
    add(db,3,text,payload)
    result=retrieve(db)
    assert set(result['required_groups'][0])=={'mem_1','mem_2'} and not result['blocked_ids']


def test_superseded_slot_cannot_remove_other_facts_in_same_original():
    db=database(); text='Atlas status is pending. Beacon room is Zurich.'
    data=packet(text,'Atlas','pending')
    data['records']+=packet(text,'Beacon','Zurich',kind='fact',attribute='room')['records']
    add(db,1,text,data)
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    assert 'mem_1' not in retrieve(db)['blocked_ids']


def test_latest_status_is_current_but_ordering_is_history():
    assert route_memory_requests({'groups':[]},{},'What is the latest Atlas status?')['temporal_mode']=='current'
    assert route_memory_requests({'groups':[]},{},'List Atlas in chronological order.')['temporal_mode']=='order'
    assert route_memory_requests({'groups':[]},{'temporal_mode':'order'},'Which happened latest?')['temporal_mode']=='order'


def test_an_active_value_does_not_make_future_plan_ineligible_as_source_evidence():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status will be approved next year.',packet('Atlas status will be approved next year.','Atlas','approved',mode='future'))
    result=retrieve(db)
    assert result['required_groups']==[['mem_1']]
    assert 'mem_2' not in result['blocked_ids']


def test_summary_keeps_literal_named_raw_when_extraction_has_no_records():
    db=database()
    add(db,1,'Atlas design milestone.',packet('Atlas design milestone.','Atlas','design',kind='fact',attribute='milestone'))
    add(db,2,'Atlas trial milestone.',{'complete':False,'records':[]})
    add(db,3,'Atlas password is secret.',packet('Atlas password is secret.','Atlas','secret',kind='privacy',attribute='password'))
    result=StructuredMemoryIndex.retrieve(db,'u','"Atlas"','Summarize all Atlas milestones',{'intent':'summary','entities':['Atlas']})
    required={sid for group in result['required_groups'] for sid in group}
    assert required=={'mem_1','mem_2'}
