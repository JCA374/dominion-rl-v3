// The Chains page: data/results.json -> chain_exam (experiments/chain_exam.py) and plan_tests.
(async function () {
  const R = await get('results.json');
  if (!R) return;
  const pct = (x) => (x == null ? '–' : (100 * x).toFixed(1) + '%');
  const ex = R.chain_exam, pt = R.plan_tests;
  if (ex) {
    const types = Object.keys(ex.types);
    const head = '<tr><th>Model</th><th>Right</th><th>Regret</th>' + types.map((t) => `<th>${t}<br><span class="muted small">${ex.types[t]} positions</span></th>`).join('') + '</tr>';
    const rows = Object.entries(ex.models).map(([name, m]) => `<tr><td><b>${name}</b><br><span class="muted small">${m.n.toLocaleString()} positions</span></td>
      <td>${pct(m.right)}</td><td>$${m.coin_regret.toFixed(3)}</td>` + types.map((t) => {
        const b = m.by_type[t];
        return b ? `<td>${pct(b.right)}<br><span class="muted small">$${b.coin_regret.toFixed(2)}</span></td>` : '<td>–</td>';
      }).join('') + '</tr>').join('');
    document.getElementById('exam').innerHTML = `<div class="tablewrap"><table class="t">${head}${rows}</table></div>
      <p class="muted small">The GA genome is scored only on its own positions. It plays every +Actions card before any terminal (a fixed
      rule), and its decks are simpler. The networks are scored on all positions. Measured ${ex.t}.</p>`;
    const champ = ex.models['rl-v1 champion (r2 net 2,200)'];
    const best = (pt && pt.tests) ? Math.max(...pt.tests.map((t) => t.score)) : null;
    document.getElementById('answer').innerHTML = `
      <div class="card"><div class="k">Champion picks the best action</div><div class="v">${pct(champ.right)}</div><div class="s">of ${champ.n.toLocaleString()} exam positions</div></div>
      <div class="card"><div class="k">Money lost per action decision</div><div class="v">$${champ.coin_regret.toFixed(2)}</div><div class="s">against the planner's best</div></div>
      <div class="card"><div class="k">Best planner vs plain net</div><div class="v">${pct(best)}</div><div class="s">head to head; 50% = no gain</div></div>`;
    const m = (ex.champion_misses || []).map((e) => `<tr><td>${e.hand.join(', ')}</td><td>${e.actions_left}</td><td>${e.net}</td><td><b>${e.planner}</b></td>
      <td class="small">${Object.entries(e.coins).map(([k, v]) => `${k} $${v}`).join(' · ')}</td></tr>`).join('');
    document.getElementById('misses').innerHTML = `<div class="tablewrap"><table class="t"><tr><th>Hand</th><th>Actions</th><th>Net played</th><th>Planner</th><th>Expected coins per option</th></tr>${m}</table></div>`;
  }
  if (pt) {
    const rows = pt.tests.map((t) => `<tr><td>${t.what}${t.note ? `<br><span class="muted small">${t.note}</span>` : ''}</td><td>${t.kingdoms}</td>
      <td>${t.games.toLocaleString()}</td><td><b>${pct(t.score)}</b> ± ${(100 * t.se).toFixed(1)}</td></tr>`).join('');
    document.getElementById('tests').innerHTML = `<div class="tablewrap"><table class="t"><tr><th>Change</th><th>Kingdoms</th><th>Games</th><th>Score</th></tr>${rows}</table></div>
      <p>${pt.conclusion}</p>`;
  }
})();
