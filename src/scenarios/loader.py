from pathlib import Path
import yaml
from .models import ScenarioConfig, TriggerPolicy, InteractionPolicy, PlatformPolicy

def _trigger(data):
    return TriggerPolicy(**(data or {}))

def load_scenario(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["scenario"]
    i = dict(raw.get("interaction", {}))
    p = dict(raw.get("platform", {}))
    interaction = InteractionPolicy(trigger=_trigger(i.pop("trigger", None)), **i)
    platform = PlatformPolicy(trigger=_trigger(p.pop("trigger", None)), **p)
    s = ScenarioConfig(
        id=raw["id"], name=raw["name"], description=raw["description"],
        expected_mechanism=raw["expected_mechanism"],
        interaction=interaction, platform=platform,
        diagnosis_enabled=raw.get("diagnosis_enabled", False),
        adaptive_actions=raw.get("adaptive_actions", {}),
        tags=raw.get("tags", []),
    )
    s.validate()
    return s

def load_scenarios(directory):
    return [load_scenario(p) for p in sorted(Path(directory).glob("S*.yaml"))]
