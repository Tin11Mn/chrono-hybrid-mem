from scripts.check_readme_consistency import validate_readmes


def test_multilingual_readmes_share_facts_links_and_versions():
    assert validate_readmes() == []
