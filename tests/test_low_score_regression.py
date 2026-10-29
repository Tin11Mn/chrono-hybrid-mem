from app.structured_memory import grounded_records, route_memory_requests
from test_state_scope_regression import database, add, packet
from app.structured_memory import StructuredMemoryIndex


def test_summary_preserves_prior_stages_even_when_planner_defaults_current():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    slot=StructuredMemoryIndex.catalog(db,'u')['groups'][0]['group_id']
    result=StructuredMemoryIndex.retrieve(db,'u','"Atlas"','Summarize Atlas status changes.',
        {'intent':'summary','entities':['Atlas'],'memory_requests':{'state_slots':[slot],'temporal_mode':'current'}})
    assert result['blocked_ids']==[]
    assert result['diagnostics'][slot]['status']=='history'


def test_summary_without_history_keyword_preserves_stages():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    slot=StructuredMemoryIndex.catalog(db,'u')['groups'][0]['group_id']
    result=StructuredMemoryIndex.retrieve(db,'u','"Atlas"','Summarize Atlas milestones.',
        {'intent':'summary','entities':['Atlas'],'memory_requests':{'state_slots':[slot],'temporal_mode':'current'}})
    assert result['blocked_ids']==[]
    assert result['diagnostics'][slot]['status']=='history'


def test_effective_date_cannot_be_borrowed_from_another_event():
    text='On 2024-01-10 Atlas launched. On 2024-04-20 Beacon closed.'
    payload=packet(text,'Atlas','launched',kind='event')
    payload['records'][0]['effective_date']='2024-04-20'
    rows,complete=grounded_records(payload,text,'user',100)
    assert not complete
    assert not rows


def test_before_question_does_not_default_to_current_state():
    result=route_memory_requests({'groups':[]},{'temporal_mode':'current'},'Which Atlas stage happened before launch?')
    assert result['temporal_mode']=='order'


def test_chinese_order_question_preserves_versions():
    result=route_memory_requests({'groups':[]},{'temporal_mode':'current'},'青禾项目哪个阶段最先发生？')
    assert result['temporal_mode']=='order'


def test_partial_index_keeps_current_candidate_first_without_deleting_history():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    add(db,3,'Unindexed context.')
    slot=StructuredMemoryIndex.catalog(db,'u')['groups'][0]['group_id']
    result=StructuredMemoryIndex.retrieve(db,'u','"Atlas"','What is Atlas current status?',
        {'intent':'governance','memory_requests':{'state_slots':[slot]}})
    assert result['required_groups']==[['mem_2','mem_1']]
    assert result['blocked_ids']==[]


def test_correct_event_date_is_retained():
    text='On 2024-01-10 Atlas launched. On 2024-04-20 Beacon closed.'
    payload=packet(text,'Atlas','launched',kind='event')
    payload['records'][0]['effective_date']='2024-01-10'
    rows,complete=grounded_records(payload,text,'user',100)
    assert complete and rows[0]['effective_ts']==1704844800


def test_current_summary_does_not_restore_obsolete_status():
    db=database()
    add(db,1,'Atlas status is pending.',packet('Atlas status is pending.','Atlas','pending'))
    add(db,2,'Atlas status is now approved.',packet('Atlas status is now approved.','Atlas','approved',mode='update'))
    slot=StructuredMemoryIndex.catalog(db,'u')['groups'][0]['group_id']
    result=StructuredMemoryIndex.retrieve(db,'u','"Atlas"','Summarize Atlas current status.',
        {'intent':'summary','entities':['Atlas'],'memory_requests':{'state_slots':[slot]}})
    assert result['diagnostics'][slot]['status']=='resolved'
