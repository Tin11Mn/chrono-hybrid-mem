"""Literal request routing supplements, rather than guesses, model intent."""
from app.query_routing import recover_summary_intent


def test_explicit_summary_request_recovers_omitted_model_route():
    for query in ['Summarize all Juniper milestones.', 'Please give me an overview of the Vega project.',
                  '请梳理青禾项目的完整历程。', '概括全部阶段的进展']:
        plan={'intent':'fact','entities':['Juniper'],'evidence_needs':['milestones']}
        fixed=recover_summary_intent(query,plan)
        assert fixed['intent']=='summary'
        assert fixed['entities']==plan['entities']
        assert plan['intent']=='fact'


def test_summary_word_in_a_fact_question_cannot_trigger_full_history():
    for query in ['What did Mara summarize yesterday?', 'Who wrote the project summary?',
                  'What does overview mean?', '总结报告的作者是谁？', 'How many milestones were completed?']:
        plan={'intent':'fact'}
        assert recover_summary_intent(query,plan)==plan
