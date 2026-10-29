"""Small literal routing guards; no facts or semantic aliases are invented."""
import re
from typing import Any, Dict


def recover_summary_intent(query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Honor explicit summary imperatives if the planner chooses a narrow intent.

    Merely mentioning a summary/report in a fact question does not expand history.
    Existing as-of/current/history constraints and entities remain unchanged.
    """
    english = re.match(r'^\s*(?:please\s+)?(?:summari[sz]e\b|recap\b|'
                       r'(?:give(?:\s+me)?|provide)\s+(?:an?\s+)?(?:overview|summary)\b)', query, re.I)
    chinese = re.match(r'^\s*(?:请|请帮我|帮我)?(?:总结|概括|归纳|梳理)(?!报告|文档|一词|这个词)', query)
    if not (english or chinese):
        return plan
    return {**plan, 'intent': 'summary'}
