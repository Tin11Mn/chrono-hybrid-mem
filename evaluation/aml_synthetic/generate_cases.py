"""Generate a public, deterministic AML-like evidence-retrieval suite.

The cases are templates, not reconstructions of hidden benchmark examples.
Regenerate with: python evaluation/aml_synthetic/generate_cases.py
"""

import json
from pathlib import Path


NAMES = [
    "Mina", "Ravi", "Noah", "Ava", "Lina", "Omar", "Iris", "Theo", "Nora", "Kai",
]
ITEMS = [
    "oolong tea", "jazz records", "trail maps", "ceramic mugs", "violet ink",
    "chess books", "linen notebooks", "herbal soap", "film cameras", "red scarves",
]
CITIES = [
    "Oslo", "Kyoto", "Lima", "Accra", "Tallinn", "Dakar", "Bern", "Pune", "Riga", "Quito",
]


def memory(memory_id, content, timestamp=None, user_id="user-main"):
    item = {
        "id": memory_id,
        "user_id": user_id,
        "session_id": "session-main",
        "role": "user",
        "content": content,
    }
    if timestamp is not None:
        item["timestamp"] = timestamp
    return item


def plan(intent, core, expansion=None, entities=None, temporal=None, needs=None):
    return {
        "intent": intent,
        "core_terms": core,
        "expansion_terms": expansion or [],
        "entities": entities or [],
        "temporal_cues": temporal or [],
        "evidence_needs": needs or [],
    }


def case(case_id, category, memories, query, required, forbidden=None, options=None,
         structured_plan=None):
    return {
        "id": case_id,
        "memories": memories,
        "query": query,
        "options": options or [],
        "required_evidence_ids": required,
        "forbidden_evidence_ids": forbidden or [],
        "category": category,
        "structured_plan": structured_plan or plan("other", []),
    }


def generate():
    cases = []
    for index in range(30):
        name = NAMES[index % len(NAMES)]
        other = NAMES[(index + 3) % len(NAMES)]
        item = ITEMS[index % len(ITEMS)]
        city = CITIES[index % len(CITIES)]
        prior_city = CITIES[(index + 4) % len(CITIES)]
        base = 1_700_000_000 + index * 10_000

        cases.append(case(
            f"A-{index:02d}", "A",
            [
                memory("mem_1", f"{name}'s archive code is ZX-{index:02d}-{city}."),
                memory("mem_2", f"{name} talked about archives and travel."),
                memory("mem_3", f"{other}'s archive code is ZX-{(index + 1):02d}-{city}."),
            ],
            f"What exact archive code belongs to {name}?", ["mem_1"], ["mem_3"],
            structured_plan=plan(
                "fact", [name, "archive code"], ["archive", "travel", city],
                [name], needs=[f"{name} exact archive code"],
            ),
        ))

        cases.append(case(
            f"B-{index:02d}", "B",
            [
                memory("mem_1", f"{name} works at the Northwind {item} studio."),
                memory("mem_2", f"The Northwind {item} studio is located in {city}."),
                memory("mem_3", f"{name} once visited {prior_city}."),
                memory("mem_4", f"{other} works at a Northwind branch."),
            ],
            f"In which city is the studio where {name} works?", ["mem_1", "mem_2"], ["mem_3"],
            structured_plan=plan(
                "multi_hop", [name, "works", "studio"], ["visited", "branch"], [name],
                needs=[f"studio where {name} works", "studio location city"],
            ),
        ))

        cases.append(case(
            f"C-{index:02d}", "C",
            [
                memory("mem_1", f"{name}'s travel destination was {prior_city}.", base),
                memory("mem_2", f"{name}'s current travel destination is {city}.", base + 500),
                memory("mem_3", f"{other}'s current travel destination is {prior_city}.", base + 600),
            ],
            f"What is {name}'s current travel destination?", ["mem_2"], ["mem_1", "mem_3"],
            structured_plan=plan(
                "temporal", [name, "travel destination", "current"], ["visited", prior_city],
                [name], ["current", "now"], [f"latest destination for {name}"],
            ),
        ))

        governance_historical = index % 2 == 1
        cases.append(case(
            f"D-{index:02d}", "D",
            [
                memory("mem_1", f"{name}'s notification channel is email.", base),
                memory("mem_2", f"Correction: {name}'s notification channel is now SMS, not email.", base + 500),
                memory("mem_3", f"{other}'s notification channel is email.", base + 600),
            ],
            (
                f"Which notification channel did {name} use before the correction?"
                if governance_historical else
                f"Which notification channel is valid now for {name}?"
            ),
            ["mem_1"] if governance_historical else ["mem_2"],
            ["mem_2"] if governance_historical else ["mem_1", "mem_3"],
            structured_plan=plan(
                "governance", [name, "notification channel"], ["email", "message"], [name],
                ["before", "previous"] if governance_historical else ["current", "now"],
                [
                    f"previous notification state for {name}"
                    if governance_historical else f"valid corrected state for {name}"
                ],
            ),
        ))

        cases.append(case(
            f"E-{index:02d}", "E",
            [
                memory("mem_1", f"{name} prefers gifts related to {item}."),
                memory("mem_2", f"{other} dislikes {item}."),
                memory("mem_3", f"{name} discussed a generic gift guide."),
            ],
            f"Which memory should guide a personalized gift for {name}?", ["mem_1"], ["mem_2"],
            structured_plan=plan(
                "personalization", [name, item, "prefers"], ["generic gift guide", "dislikes"],
                [name], needs=[f"{name} gift preference"],
            ),
        ))

        cases.append(case(
            f"G-{index:02d}", "G",
            [
                memory("mem_1", f"Rule: {name}'s reports must never include raw account numbers."),
                memory("mem_2", f"Example: {name} prepared a report about account activity."),
                memory("mem_3", f"Old suggestion: reports may include the last four digits."),
            ],
            f"What authoritative constraint applies when preparing {name}'s reports?", ["mem_1"], ["mem_2", "mem_3"],
            structured_plan=plan(
                "rule", [name, "reports", "must never"], ["example", "suggestion"], [name],
                needs=[f"authoritative report rule for {name}"],
            ),
        ))

        cases.append(case(
            f"H-{index:02d}", "H",
            [
                memory("mem_1", f"Privacy boundary: do not disclose {name}'s {item} purchase history."),
                memory("mem_2", f"{name} bought {item} last month."),
                memory("mem_3", f"{other} publicly reviewed {item}."),
                memory("mem_4", f"Private record for another user: {city}.", user_id="user-other"),
            ],
            f"Which memory governs whether {name}'s purchase history may be disclosed?", ["mem_1"], ["mem_2", "mem_4"],
            structured_plan=plan(
                "safety", [name, "privacy boundary", "disclose", "purchase history"],
                ["bought", "public review"], [name], needs=[f"privacy consent boundary for {name}"],
            ),
        ))
    return cases


def validate(cases):
    categories = {category: 0 for category in "ABCDEGH"}
    required_keys = {
        "memories", "query", "options", "required_evidence_ids",
        "forbidden_evidence_ids", "category",
    }
    ids = set()
    for item in cases:
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"{item.get('id')} missing {sorted(missing)}")
        if item["id"] in ids:
            raise ValueError(f"duplicate case id {item['id']}")
        ids.add(item["id"])
        categories[item["category"]] += 1
        memory_ids = {memory_item["id"] for memory_item in item["memories"]}
        if not set(item["required_evidence_ids"]) <= memory_ids:
            raise ValueError(f"{item['id']} has unknown required evidence")
        if set(item["required_evidence_ids"]) & set(item["forbidden_evidence_ids"]):
            raise ValueError(f"{item['id']} overlaps required and forbidden evidence")
    if any(count < 30 for count in categories.values()):
        raise ValueError(f"insufficient category counts: {categories}")
    return categories


if __name__ == "__main__":
    generated = generate()
    validate(generated)
    output = Path(__file__).with_name("cases.json")
    output.write_text(json.dumps(generated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(generated)} cases to {output}")
