// Project formatters: how this project's run events read in the log.
window.eventText = function (e) {
  const p = (x) => (x == null ? '–' : (100 * x).toFixed(1) + '%');
  switch (e.kind) {
    case 'run_start': return [`Run started from rl-v1's checkpoint (iteration 2,219)`, `With the 115 helper inputs at zero weight: ${e.config.games.toLocaleString()} games per PPO iteration.`];
    case 'note': return ['Note', e.text];
    case 'init_eval': return ['Iteration 0 evaluated (rl-v1 checkpoint, widened)', e.eval];
    case 'goal_best': return [`New best against the GA base: iteration ${e.iter}`, `${p(e.check.score)} on ${e.check.n} evaluation games. ${e.eval}\nBest of many evaluations: flattered by luck (winner's curse). The goal checks are the real measure.`];
    case 'stage_start': return [`Stage started: ${e.stage} (iteration ${e.iter})`, e.expect ? 'Expected: ' + e.expect : ''];
    case 'stage_pass': return [`✅ Stage done: ${e.stage} (iteration ${e.iter})`, `${e.why}. ${e.eval || ''}`];
    case 'stage_cap': return [`Stage ${e.stage} used its budget (iteration ${e.iter}); moving on`, `${e.why}. ${e.eval || ''}`];
    case 'run_resume': return [`Run resumed at iteration ${e.iter}`, e.stage ? `Stage: ${e.stage}.` : ''];
  }
  return null;
};
