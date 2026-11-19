// The Cards page: data/results.json -> card_audit_<name> (experiments/card_audit.py), one table per audited model.
(async function () {
  const R = await get('results.json');
  if (!R) return;
  const audits = Object.entries(R).filter(([k]) => k.startsWith('card_audit_')).map(([, v]) => v);
  if (!audits.length) return;
  const label = (a) => a.name === 'champion' ? "rl-v1's champion" : `rl-v3 ${a.name.replace('_', ' iteration ')}`;
  const pct = (x) => Math.round(100 * x) + '%';
  function show(a) {
    const rows = Object.entries(a.cards).map(([c, r]) => {
      const q = {...r.asks_answered, ...r.as_victim_answered};
      const qs = Object.entries(q).map(([k, ok]) => ok ? `${k}: yes` : `<span class="muted">${k}: not asked${r.plays_per_game === 0 && !(k in r.as_victim_answered) ? ' (never played)' : ''}</span>`).join('<br>');
      const f = r.forced;
      const fa = f ? Object.entries(f.answers).map(([k, v]) => `<b>${k}</b>: ` + Object.entries(v).slice(0, 5).map(([a, n]) => `${a} ${n}`).join(', ')).join('<br>') : '';
      const fq = f ? Object.entries(f.asks_answered).map(([k, ok]) => `<span class="${ok ? '' : 'bad'}">${k}: ${ok ? 'yes' : 'never'}</span>`).join('<br>') : '';
      return `<tr><td><b>${c}</b></td><td>${pct(r.win)}</td><td>${pct(r.bought_in)}</td><td>${r.copies_bought.toFixed(2)}</td>
        <td>${r.plays_per_game.toFixed(2)}</td><td class="small">${qs || '<span class="muted">none</span>'}</td>
        <td>${f ? f.plays_per_game.toFixed(2) : '–'}</td><td class="small">${fq || '<span class="muted">none</span>'}</td><td class="small">${fa}</td>
        <td>${r.errors + (f ? f.errors : 0)}</td></tr>`;
    }).join('');
    document.getElementById('audit').innerHTML = `<p class="muted small">${label(a)}: ${a.games.toLocaleString()} games, ${a.t}.
      Problems found: ${a.problems.length ? a.problems.join('; ') : 'none'}.</p>
      <div class="tablewrap"><table class="t"><tr><th rowspan="2">Card</th><th colspan="5">Its own choices</th><th colspan="3">With 2 forced copies</th><th rowspan="2">Errors</th></tr>
      <tr><th>Win</th><th>Takes it</th><th>Copies</th><th>Plays</th><th>Questions</th><th>Plays</th><th>Questions</th><th>What it answers (counts)</th></tr>${rows}</table></div>`;
  }
  const sel = audits.map((a, i) => `<button data-i="${i}">${label(a)}</button>`).join(' ');
  document.getElementById('pick').innerHTML = `<p>${sel}</p>`;
  document.querySelectorAll('#pick button').forEach((b) => b.onclick = () => show(audits[+b.dataset.i]));
  show(audits[audits.length - 1]);
})();
