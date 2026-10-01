import ast, hashlib, subprocess
from pathlib import Path
root=Path('/Users/jzales/.codex/worktrees/crest-proposal-contract/fall-2026-nesl-crest')
base='0f870764b6cf83d0024bc98139276a993d141871'
checks={
'src/nas_model_client.py':['objective','_constraints_func','_direction_penalty_values','_family_trial_params','train_best_trial','evaluate_checkpoint'],
'src/crest/model.py':['log_trial','build_trial_outcome','evaluate_score_config','evaluate_prune_rules','evaluate_feasibility_rules'],
}
for path,names in checks.items():
    old=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=root,text=True)
    new=(root/path).read_text()
    old_nodes={n.name:n for n in ast.walk(ast.parse(old)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_nodes={n.name:n for n in ast.walk(ast.parse(new)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for name in names:
        assert ast.dump(old_nodes[name],include_attributes=False)==ast.dump(new_nodes[name],include_attributes=False),(path,name)
        print(f'UNCHANGED AST {path}:{name}')
for path in ['src/crest/pareto_replay.py','src/pareto_hil_replay.py','src/crest/model_families/audio_dscnn.py','src/crest/model_families/odom_tcn.py','analysis_scripts/llm_token_cost/estimate.py']:
    old=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=root)
    assert old==(root/path).read_bytes(),path
    print(f'UNCHANGED FILE {path} sha256={hashlib.sha256(old).hexdigest()}')
old=subprocess.check_output(['git','show',f'{base}:src/crest/optimizers/llm/semantic_context.py'],cwd=root,text=True)
new=(root/'src/crest/semantic_context.py').read_text()
old_nodes={n.name:n for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef)}
new_nodes={n.name:n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.FunctionDef)}
for name,node in old_nodes.items():
    assert ast.dump(node,include_attributes=False)==ast.dump(new_nodes[name],include_attributes=False),name
print(f'UNCHANGED AST all {len(old_nodes)} relocated semantic-context functions')
