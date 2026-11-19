// "What it plays right now": data/favorites.json from site_hooks.favorites() (the newest net against the GA base).
(async function () {
  const el = document.getElementById('favorites');
  if (!el) return;
  let d;
  try { d = await (await fetch('data/favorites.json', {cache: 'no-store'})).json(); } catch (e) { el.textContent = 'Not measured yet.'; return; }
  const pct = (x) => (x == null ? '–' : Math.round(100 * x) + '%');
  const bar = (x, cls) => `<span class="favbar ${cls || ''}" style="width:${Math.round(100 * Math.min(1, x))}%"></span>`;
  const top = d.cards.filter((c) => c.games >= 20);
  const rowsCards = top.map((c) => `<tr><td><b>${c.card}</b></td><td class="favcell">${bar(c.take)}<span>${pct(c.take)}</span></td>
      <td>${c.copies.toFixed(1)}</td><td>${pct(c.score_took)}</td><td>${pct(c.score_left)}</td><td class="muted">${c.games}</td></tr>`).join('');
  const maxBase = Math.max(...d.base.map((b) => b.per_game));
  const rowsBase = d.base.map((b) => `<tr><td><b>${b.card}</b></td><td class="favcell">${bar(b.per_game / maxBase, 'base')}<span>${b.per_game.toFixed(1)}</span></td></tr>`).join('');
  const rowsCombo = d.combos.map((c) => `<tr><td><b>${c.cards[0]} + ${c.cards[1]}</b></td><td>${pct(c.both)}</td>
      <td>${c.lift.toFixed(1)}×</td><td>${c.copies[0].toFixed(1)} + ${c.copies[1].toFixed(1)}</td><td>${pct(c.score)}</td><td class="muted">${c.games}</td></tr>`).join('');
  const rowsOpen = d.openings.map((o) => `<tr><td><b>${o.opening}</b></td><td class="favcell">${bar(o.share / d.openings[0].share, 'open')}<span>${pct(o.share)}</span></td></tr>`).join('');
  el.innerHTML = `
  <p class="muted small">Measured on iteration ${d.iter}: ${d.games} games (${d.games / 2} fresh kingdoms, both seats) against the GA
  base genome, playing its top choice; it scored ${pct(d.score)} in them. Redrawn every time a new network is saved (every 25 iterations).</p>
  <div class="favgrid">
  <div><h3>Kingdom cards it goes for</h3>
  <p class="muted small">"Takes it": the share of the games with the card on the table in which it bought at least one.
  "Score" columns: its result when it took the card and when it left it (a big gap means the card matters, not that it causes the win).</p>
  <div class="tablewrap"><table class="t fav"><tr><th>Card</th><th>Takes it</th><th>Copies</th><th>Score took</th><th>Score left</th><th>Games</th></tr>${rowsCards}</table></div></div>
  <div><h3>Money and victory cards</h3>
  <p class="muted small">Copies bought per game, on average.</p>
  <div class="tablewrap"><table class="t fav"><tr><th>Card</th><th>Per game</th></tr>${rowsBase}</table></div>
  <h3>Its openings</h3>
  <p class="muted small">What it buys on its first two turns ($3/$4 or $5/$2 hands).</p>
  <div class="tablewrap"><table class="t fav"><tr><th>Opening</th><th>Share</th></tr>${rowsOpen}</table></div></div>
  </div>
  <h3>Combos it likes</h3>
  <p class="muted small">Pairs of kingdom cards it buys together, when both are on the table. "Together": the share of those games in which it
  bought both. "Lift": how much more often than if it chose each card independently (1× = no link, 2× = twice as often).
  Ranked by how often it builds the pair, weighted by the lift.</p>
  <div class="tablewrap"><table class="t fav"><tr><th>Pair</th><th>Together</th><th>Lift</th><th>Copies</th><th>Score</th><th>Games</th></tr>${rowsCombo}</table></div>`;
})();

// How it uses its turns (+Actions, +Cards, +Buys), next to the GA genome and rl-v1.
(async function () {
  const el = document.getElementById('turnuse');
  if (!el) return;
  let d;
  try { d = await (await fetch('data/favorites.json', {cache: 'no-store'})).json(); } catch (e) { return; }
  if (!d.turns) { el.textContent = 'Not measured yet.'; return; }
  const who = Object.keys(d.turns);
  const rows = [['Actions played per turn', 'actions per turn', (x) => x.toFixed(2)],
                ['Turns with 2 or more actions (needs +Actions)', 'turns with 2+ actions', (x) => Math.round(100 * x) + '%'],
                ['Cards drawn by its actions, per turn (+Cards)', 'cards drawn by actions per turn', (x) => x.toFixed(2)],
                ['Turns in which it played a +Buy card', 'turns with a +Buy card played', (x) => Math.round(100 * x) + '%'],
                ['… and used the extra buy (bought 2+ cards)', '... of those, bought 2+ cards', (x) => Math.round(100 * x) + '%']];
  const head = '<tr><th></th>' + who.map((w) => `<th>${w === 'this net' ? 'rl-v2 (iteration ' + d.iter + ')' : w}</th>`).join('') + '</tr>';
  const body = rows.map(([label, key, f]) => `<tr><td>${label}</td>` + who.map((w) => `<td>${f(d.turns[w][key])}</td>`).join('') + '</tr>').join('');
  const most = who.map((w) => `<tr><td>${w === 'this net' ? 'rl-v2' : w}</td><td>${d.turns[w]['most played'].slice(0, 6).map((p) => p[0]).join(', ')}</td></tr>`).join('');
  el.innerHTML = `<p class="muted small">${d.turns[who[0]].turns.toLocaleString()} turns each, on the same 300 games against the GA base genome
  (150 kingdoms that have Village, Festival, Market, Laboratory or Council Room). A strategy that chains actions ("an engine") plays
  2+ actions in many turns and draws many cards; a money strategy plays about one terminal action per turn.</p>
  <div class="tablewrap"><table class="t fav">${head}${body}</table></div>
  <h3>Actions it plays most</h3><div class="tablewrap"><table class="t fav">${most}</table></div>`;
})();
