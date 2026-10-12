"""Verify the five materialized phases and publish a registry for RQ1/RQ3."""
from pathlib import Path
import argparse
import json
import yaml
from common.artifacts import atomic_write_json
from scripts.aria.verify_training_checkpoint import verify


def register(spec):
    root = Path(spec['output_root']).resolve()
    checkpoints = {}
    for stage in spec['stages']:
        accepted = verify({
            'name': f"{spec['family']}/{stage}", 'output_dir': str(root / stage),
            'stages': [stage], 'foundation_model': spec['foundation_model'],
            'lora_rank': spec['lora_rank'],
            'expected_grpo_updates': spec['expected_grpo_updates'],
        })
        checkpoints[stage] = accepted['stages'][0]
    gate_report = json.loads((root / 'grpo_gate/checkpoints/grpo_gate/training_report.json').read_text())
    if gate_report.get('stage3', {}).get('gate_policy') != 'contextual':
        raise ValueError('The released Full recipe requires contextual Gate GRPO')
    registry = {'seed': spec['seed'], 'status': 'materialized_training_verified',
                'source_corpus': str(Path('data/training/full').resolve()),
                'checkpoints': checkpoints}
    atomic_write_json(root / 'checkpoint_registry.json', registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    register(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
