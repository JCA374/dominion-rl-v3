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
- Run c2 (card drill, from c1 iteration 2363): `systemd-run --user --unit=rl-v3-c2 --working-directory=$PWD bash -c
  'exec .venv/bin/python -u train.py --run c2 --continue --set drill_share=0.3 lr=0.0002 workers=5 >> logs/c2.log 2>&1'`.
  Run training and `publish_loop.sh` as systemd user units, not in a terminal: on 20 Nov systemd-oomd killed the terminal
  (and c1 with it). Stop: `systemctl --user stop rl-v3-c2`.
- Card losses: `experiments/card_losses.py <net> --name X` (64,000 games, 35 s per 16,000) then
  `experiments/card_losses_report.py X --ref champion_ref` (same games with the champion; report to `docs/reports/`).
- Podcast: project-podcast skill, `podcast.conf`. The voice studio's venv runs on uv's Python 3.12 (the OS moved to 3.14).
- Never put personal names or home paths on the site, in commits or podcasts.
- PPO fixes (2026-11-24, tests/test_ppo_fixes.py): the update divides the logits by cfg temp (as the C sampler does);
  errored or record-overflowed games give no records (stats: rejected_error/overflow, capped_games, points_uncapped);
  cfg seed = stream seed independent of the run name, ordered pool.imap, seeded minibatch shuffles (same seed ->
  identical nets); cfg frozen ("c2:0.1,c1:0.1", FROZEN), eval_vs, exclude (shared kingdom exclusion manifest);
  --init_from needs --stage and runs init_check (weights equal source, first PPO ratio 1). Logs games/updates/CPU hours.
