"""Archive verified S1/S2 and freeze a dated, native 16+16 training release."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from common.artifacts import atomic_write_json
from masfm.config import load_config
from scripts.aria.verify_training_checkpoint import file_sha256, verify

ROOT = Path(__file__).resolve().parents[2]


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def set_option(command, name, value):
    flag = '--' + name
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def build(spec):
    release = ROOT / 'outputs/aria' / ('aria-training-' + spec['timestamp'])
    release.mkdir(parents=True, exist_ok=False)
    snapshot = release / 'code'
    snapshot.mkdir()
    # Freeze the current workspace, including intentional uncommitted fixes.
    for name in ('src', 'baselines', 'scripts', 'configs'):
        shutil.copytree(ROOT / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('main.py', 'pyproject.toml', 'uv.lock', '.python-version'):
        shutil.copy2(ROOT / name, snapshot / name)
    hashes = {str(p): file_sha256(p) for p in snapshot.rglob('*') if p.is_file()}
    (release / 'workspace.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    views = release / 'views'
    views.mkdir()
    view_manifest = {}
    for key, source in spec['views'].items():
        source = ROOT / source
        target = views / (key + '.jsonl')
        shutil.copy2(source, target)
        hashes[str(target)] = file_sha256(target)
        count = sum(1 for line in target.open() if line.strip())
        view_manifest[key] = {'path': str(target), 'source': str(source), 'rows': count, 'sha256': hashes[str(target)]}
    atomic_write_json(release / 'views.json', view_manifest)
    shutil.copy2(spec['figure_readme'], release / 'figure_design.md')
    jobs = []
    for model in spec['models']:
        name = 'aria-' + model['backbone'] + '-' + spec['timestamp']
        root = ROOT / 'outputs/aria' / name
        root.mkdir(exist_ok=False)
        (root / 'logs').mkdir()
        (root / 'eval_logs').mkdir()
        (root / 'eval_logs/cases.jsonl').touch()
        registry = {'name': name, 'source_version': 'rq1-three-stage-20260913', 'seed': 42,
                    'status': 'stage12_archived_stage3_pending', 'checkpoints': {},
                    'scope': 'in_domain full model; S1+3, held-out training and seeds43/44 are separate jobs'}
        stats = []
        for stage, source in model['stage12'].items():
            source = ROOT / source
            report = json.loads((source.parent / 'training_report.json').read_text())
            phase_root = source.parents[2]
            manifest = json.loads((phase_root / 'post_training_manifest.json').read_text())
            acceptance = verify({'name': name + '-' + stage, 'output_dir': str(phase_root),
                                 'stages': manifest['stages'], 'foundation_model': model['foundation_model'], 'lora_rank': 32})
            target = root / 'checkpoints' / stage
            shutil.copytree(source, target / 'checkpoint')
            shutil.copy2(source.parent / 'training_report.json', target / 'training_report.json')
            entry = next(v for v in acceptance['stages'] if v['stage'] == stage)
            registry['checkpoints'][stage] = {**entry, 'source_checkpoint': str(source), 'checkpoint': str(target / 'checkpoint'), 'reused': True}
            for p in target.rglob('*'):
                if p.is_file(): hashes[str(p)] = file_sha256(p)
            evidence = report.get('context_budget', {}).get('sft_evidence', {})
            rows = evidence.get('kept') if stage == 'sft' else report.get('transfer_diagnosis_rows') if stage == 'transfer_diagnosis' else report.get('context_budget', {}).get('dpo_evidence', {}).get('kept')
            if stage == 'dpo' and rows is None:
                count = sum(1 for line in Path(report['datasets'][0]).open() if line.strip())
                rows = f'{count} input pairs (filtered unique row count unrecorded)'
            stats.append((stage, rows, report.get('optimizer_steps'), str(source)))
        atomic_write_json(root / 'checkpoint_registry.json', registry)
        source_path = root / 'source_corpus'
        shutil.copytree(ROOT / model['source_corpus'], source_path)
        for p in source_path.rglob('*'):
            if p.is_file(): hashes[str(p)] = file_sha256(p)
        stage2 = registry['checkpoints']['transfer_diagnosis']['checkpoint']
        previous = stage2
        phases = []
        for stage, template in spec['templates'].items():
            config = yaml.safe_load((ROOT / template).read_text())
            post = config['aria']['post_training']
            phase_root = root / 'training' / stage
            post.update(base_model=previous, source_path=str(source_path), output_dir=str(phase_root),
                        resume=False, environment_path=view_manifest['repair' if stage == 'grpo_repair' else 'gate']['path'])
            post['gate_validation_path'] = view_manifest['gate_validation']['path']
            command = post['backend']['command']
            command[:] = [f"CUDA_VISIBLE_DEVICES={model['gpu_uuid']}" if arg.startswith('CUDA_VISIBLE_DEVICES=') else arg for arg in command]
            for flag, value in {'reference-model': stage2, 'grpo-optimizer-steps':16, 'grpo-budget-passes':6,
                                'grpo-checkpoint-interval':1, 'grpo-throughput-after-updates':1,
                                'cuda-memory-fraction':0.80, 'grpo-gate-anchor-path':str(source_path / 'sft.jsonl')}.items():
                set_option(command, flag, value)
            if stage == 'grpo_gate':
                set_option(command, 'grpo-incidents', 192)
            env = post['environment']['options']
            env['host_api']['base_url'] = spec['host_url']
            env['host_api']['model'] = spec['host_name']
            # Keep each source phase's native budgets; disclose differences.
            env['control_output_root'] = str(root / 'controls' / stage)
            config['api'] = copy.deepcopy(env['host_api'])
            config['output_dir'] = str(root / 'output')
            config['workspace_dir'] = str(root / 'workspace')
            path = ROOT / 'configs/post_training' / name / (stage + '.yml')
            write_yaml(path, config)
            load_config(path)
            frozen = release / 'configs' / name / (stage + '.yml')
            write_yaml(frozen, config)
            hashes[str(frozen)] = file_sha256(frozen)
            phases.append({'stage':stage, 'config':str(frozen), 'output_dir':str(phase_root), 'log':str(root / 'logs' / (stage + '.log'))})
            previous = str(phase_root / 'checkpoints' / stage / 'checkpoint')
        contract = {'schema_version':'aria-figure-log-reservation-v1', 'status':'reserved_no_evaluation_run',
                    'case_schema':'aria-figure-case-v1', 'writer':'aria.figure_logging.append_case',
                    'pair_key':['pool_id','task_id','benchmark','mas','host_config_sha256','execution_seed','training_seed'],
                    'case_fields':sorted(__import__('aria.figure_logging', fromlist=['REQUIRED']).REQUIRED),
                    'training_seeds':[42,43,44], 'trained_seed_in_this_release':42,
                    'gate_grid':[0.90,0.75,0.60,0.45,0.30], 'default_gate':0.60,
                    'grid_status':'design_proposal_requires_validation_freeze_before_test',
                    'budget_grid':None, 'stop_rule':None, 'output_selection_rule':None,
                    'budget_status':'must_preregister_common_caps_stop_and_selection_rules',
                    'eb':'measured full executions; posthoc excludes source acquisition; runtime includes initial; count diagnostic/replay/additional separately',
                    'rq1':['SR','HIR and harm reasons relative to matched no-intervention control','EB','tokens','calls','wall_seconds','trigger_rate'],
                    'rq3_variants':['S1','S1+2','S1+3','Full'],
                    'rq3_edges':[['S1','S1+2'],['S1+3','Full'],['S1','S1+3'],['S1+2','Full']],
                    'rq3_denominator':'all four variants jointly resolved matched task pool, same N for every edge within domain and seed',
                    'rq3_checks':['right-left=SR(B)-SR(A)','both paths S1->Full have identical net change'],
                    'coverage':'report expected, missing, infrastructure-invalid, unresolved, jointly resolved counts',
                    'uncertainty':'compute each seed first then equal-weight mean; cluster bootstrap by task ID preserving variant/size/seed pairing',
                    'figure03':'reserve paired controls for conditional recovery; RQ2 held-out checkpoints not trained by this release',
                    'artifacts':'retain raw gate probabilities, model calls, interventions, execution ledger, native/evaluator results and config hashes'}
        atomic_write_json(root / 'eval_logs/contract.json', contract)
        doc = ROOT / 'docs' / (name + '.md')
        text = f'# {name}\n\n'
        text += f'Foundation model: `{model["foundation_model"]}`. Seed 42; LoRA rank 32; BF16; SDPA; gradient checkpointing.\n\n'
        text += 'Verified Stage 1/2 weights and tokenizers are archived with SHA-256 hashes. Stage 3 performs 16 Repair updates followed by 16 Gate updates.\n\n'
        text += '| Phase | Eligible rows / mined pairs | Optimizer updates | Source checkpoint |\n|---|---:|---:|---|\n'
        for stage, rows, steps, source in stats:
            text += f'| {stage} | {rows} | {steps if steps is not None else "unrecorded"} | `{source}` |\n'
        text += '\nRow exposures are not unique tasks. Supervised budgets may differ across backbones.\n\n'
        text += f'Output: `{root}`. Repair environments: {view_manifest["repair"]["rows"]}; Gate environments: {view_manifest["gate"]["rows"]}.\n\n'
        text += 'Acceptance requires consecutive native-reward updates and finite changed LoRA weights. Holdout overlap is checked before each phase. Intermediate weights support evaluation and warm starts; optimizer state is not retained.\n\n'
        text += 'Training reports, update journals and native artifacts record actual usage. Evaluation metrics require separate task-level evaluation. S1+3, held-out training and additional seeds require independent runs.\n\n'
        text += 'Training commands:\n\n```bash\n'
        for phase in phases:
            text += f'UV_PROJECT_ENVIRONMENT=.venv-aria-training uv run --no-sync python main.py -c configs/post_training/{name}/{phase["stage"]}.yml\n'
        text += '```\n\n<!-- LIVE_TRAINING_STATUS -->\n\nStage 3 is pending. Completion is determined by `status.json` and checkpoint acceptance.\n'
        doc.write_text(text)
        jobs.append({'name':name, 'root':str(root), 'gpu_uuid':model['gpu_uuid'], 'foundation_model':model['foundation_model'], 'documentation':str(doc), 'phases':phases})
    atomic_write_json(release / 'freeze.json', {'sha256':hashes, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    launch = {'snapshot':str(snapshot), 'training_environment':str(ROOT / '.venv-aria-training'), 'freeze_manifest':str(release / 'freeze.json'), 'terminal_path':str(release / 'terminal.json'), 'jobs':jobs}
    write_yaml(release / 'launch.yml', launch)
    print(json.dumps({'release':str(release), 'jobs':[job['name'] for job in jobs]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c','--config',type=Path,required=True)
    build(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
