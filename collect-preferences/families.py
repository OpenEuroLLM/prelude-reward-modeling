"""Model family of each responder in the dataset, for the own-family exclusion: a judge does not judge a pair
that contains an answer from a model of its own family (`family` in judges.yaml). Responders not listed share
no family with any judge."""
RESPONDER_FAMILY = {"qwen3.5-9b": "qwen", "qwen2.5-7b": "qwen", "gemma-4-12b": "gemma", "mistral-7b-v0.1": "mistral",
                    "zephyr-7b-beta": "mistral", "apertus-8b-2509": "apertus"}


def own_family(judge_family, pair):
    """True if either answer of the pair (a row of a pairs file) comes from the judge's family"""
    return judge_family in (RESPONDER_FAMILY.get(pair["model_x"]), RESPONDER_FAMILY.get(pair["model_y"]))
