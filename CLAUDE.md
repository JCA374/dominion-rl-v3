# CLAUDE.md

rl-v3: rl-v1's champion trained on (helper inputs, curriculum vs the champion, card school), plus the chain exam and the
turn planner that answered "does the network need a chain model?" (no: see README). Code is rl-v2's plus `c/rl.c`
(`rl_plan`, `rl_chain_rule`, `rl_forced_buy`), `train.py` (`STAGES`, `SCHOOL`, `widen_ckpt`) and the experiments.

- Run: `.venv/bin/python -u train.py --run c1` (detached with setsid nohup; log in `logs/`). Resume: `--continue`.
- Stop a run by pid: `pgrep -f "^.venv/bin/python -u train.py --run c[1]"`. Never `pkill -f` with a pattern that also
  matches your own shell command.
- Evaluation seeds are rl-v1's (comparable numbers); the champion games come after them. `experiments/confirm.py`: once, at the end.
- The engine submodule is newer than az-v1's (adds `de_resume_action`, no rules); `tests/test_core.py` checks that.
- Site: ml-project-site (`publish.py`, `site.json`, `site_hooks.py`, `docs/`), loop `publish_loop.sh`. Chains and Cards
  pages read `results/chain_exam.json`, `plan_tests.json`, `card_audit_*.json` (site_hooks.audits() adds the newest net every 100 iterations).
- Never put personal names or home paths on the site, in commits or podcasts.
