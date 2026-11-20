// The Learn page: lessons and a quiz from data/tactics.json (experiments/tactics.py: the strongest network's own games).
(function () {
  const h = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const P = (x) => Math.round(100 * x) + '%';
  const bars = (rows) => { const m = Math.max(...rows.map(([, x]) => x), 1e-9); return `<div class="hb">${rows.map(([l, x]) => `<span>${h(l)}</span><span><span class="bar2" style="display:block;width:${(100 * x / m).toFixed(1)}%"></span></span><span class="v">${P(x)}</span>`).join('')}</div>`; };
  const share = (band, card) => (band && (band.top.find(([n]) => n === card) || [0, 0])[1]) || 0;

  function lesson(title, rule, body) {
    return `<div class="lesson2"><h3>${title}</h3><p class="rule">${rule}</p>${body}</div>`;
  }

  function lessons(T) {
    const out = [];
    const op = T.openings;
    const o43 = op['4/3'], o52 = op['5/2'];
    out.push(lesson('Openings',
      `With $4 and $3 its favourite opening is <b>${h(o43.top[0][0])}</b>; with $5 and $2, <b>${h(o52 ? o52.top[0][0] : '–')}</b>.
       Most of its favourite openings include an attack or a card that trashes, and with a $2 it often buys nothing.`,
      `<p class="muted small">$4/$3, ${o43.n} openings</p>${bars(o43.top)}${o52 ? `<p class="muted small">$5/$2, ${o52.n} openings</p>${bars(o52.top)}` : ''}`));

    const th = T.thresholds, pls = ['8', '7', '6', '5', '4', '3', '2', '1'];
    const row = (band, cards) => `<table class="mini"><tr><th>Provinces left</th>${pls.map((p) => `<th>${p}</th>`).join('')}</tr>
      ${cards.map((c) => `<tr><td>${h(c)}</td>${pls.map((p) => `<td>${th[band][p] ? P(share(th[band][p], c)) : '–'}</td>`).join('')}</tr>`).join('')}</table>`;
    const first = (band, card, x) => pls.find((p) => th[band][p] && share(th[band][p], card) >= x);
    const p8 = th['8'];
    out.push(lesson('$8: Province, or build first?',
      `With 8 Provinces left it buys Province only ${P(share(p8['8'], 'Province'))} of the time at $8: early it takes Gold or a
       strong action instead. From ${first('8', 'Province', 0.5) || '–'} Provinces left it takes the Province most of the time.`,
      `<div class="tablewrap">${row('8', ['Province', 'Gold', 'Duchy'])}</div>`));
    const switchAt = pls.find((p) => th['6-7'][p] && share(th['6-7'][p], 'Duchy') > share(th['6-7'][p], 'Gold'));
    out.push(lesson('$6–$7: Gold or Duchy?',
      `It buys Gold early and switches to Duchy at about <b>${switchAt || '–'} Provinces left</b>: from then on, points beat money.`,
      `<div class="tablewrap">${row('6-7', ['Gold', 'Duchy'])}</div>`));
    out.push(lesson('$5: when Duchy beats an action',
      `At $5 it buys actions while the game is young; from about ${first('5', 'Duchy', 0.3) || '–'} Provinces left Duchy becomes its top buy.`,
      `<div class="tablewrap">${row('5', ['Duchy'])}</div>`));
    out.push(lesson('$2–$4 late: the Estate',
      `With little money late in the game it takes an Estate (${P(share(th['2-4']['1'], 'Estate'))} with one Province left)
       rather than nothing: a single point decides many games.`,
      `<div class="tablewrap">${row('2-4', ['Estate', 'Silver', 'nothing'])}</div>`));

    const pp = T.penultimate, g = (k) => pp[k] || 0;
    const winTake = g('wins|1') / Math.max(1, g('wins|1') + g('wins|0'));
    const loseHold = g('loses|0') / Math.max(1, g('loses|1') + g('loses|0'));
    out.push(lesson('The last Province',
      `With one Province left and $8: if taking it wins the game, it takes it ${P(winTake)} of the time. If taking it would
       end the game behind, it holds off ${P(loseHold)} of the time and buys something else to catch up first.`,
      `<p class="muted small">${g('wins|1') + g('wins|0')} positions where it would win, ${g('loses|1') + g('loses|0')} where it would lose.</p>`));

    const cards = Object.entries(T.cards).filter(([, c]) => c.present >= 50).sort((a, b) => b[1].bought - a[1].bought);
    const crow = ([n, c]) => `<tr><td>${h(n)}</td><td>$${c.cost}</td><td>${P(c.bought)}</td><td>${c.copies ? c.copies.toFixed(1) : '–'}</td><td>${c.first_turn ?? '–'}</td></tr>`;
    out.push(lesson('Cards it wants, cards it skips',
      `Top of its list when they are on the board: ${cards.slice(0, 4).map(([n]) => `<b>${h(n)}</b>`).join(', ')}. It mostly skips
       ${cards.slice(-3).map(([n]) => h(n)).join(', ')}.`,
      `<div class="tablewrap"><table class="mini"><tr><th>Card</th><th>Cost</th><th>Bought (when on the board)</th><th>Copies</th><th>First turn</th></tr>
       ${cards.map(crow).join('')}</table></div><p class="muted small">Copies: how many it ends with when it buys any; first turn: median turn of its first one.</p>`));
    return out.join('');
  }

  // ── quiz ──
  let Q = [], qi = 0, right = 0, done = 0;
  function deck(o) { return Object.entries(o).map(([n, c]) => `${h(n)} ×${c}`).join(', '); }
  function showQuiz() {
    const q = Q[qi % Q.length], s = q.state;
    const opts = q.options.map(([n]) => n).sort();
    document.getElementById('quiz').innerHTML = `
      <div class="row" style="justify-content:space-between;margin-top:0"><b>Position ${qi % Q.length + 1} of ${Q.length}</b><span class="muted small">${done ? `${right} of ${done} right` : ''}</span></div>
      <div class="ctx"><p><b>Turn ${s.turn}</b> · <b>$${s.coins}</b>, ${s.buys} buy${s.buys > 1 ? 's' : ''} · VP: you ${s.vp}, them ${s.opp_vp} · ${s.prov_left} Province${s.prov_left === 1 ? '' : 's'} left</p>
        <p class="small"><b>You</b>: ${deck(s.mine)}</p><p class="small"><b>Them</b>: ${deck(s.theirs)}</p>
        <p class="small muted">Kingdom: ${q.kingdom.map(h).join(', ')}</p></div>
      <p><b>What does the model buy?</b></p>
      <div class="btns">${opts.map((n) => `<button data-n="${h(n)}">${h(n)}</button>`).join('')}</div><div id="qres"></div>`;
    document.querySelectorAll('#quiz .btns button').forEach((b) => { b.onclick = () => reveal(q, b.dataset.n); });
  }
  function reveal(q, pick) {
    if (document.querySelector('#quiz .btns button.right')) return;
    const best = q.options[0][0];
    done++; if (pick === best) right++;
    document.querySelectorAll('#quiz .btns button').forEach((b) => {
      if (b.dataset.n === best) b.classList.add('right'); else if (b.dataset.n === pick) b.classList.add('wrong');
    });
    document.getElementById('qres').innerHTML = `<p>${pick === best ? '✓ Same as the model.' : `✗ The model buys <b>${h(best)}</b>.`}
      Its win estimate here: ${P((q.value + 1) / 2)}.</p>${bars(q.options.slice(0, 5))}
      <div class="row"><button class="big" id="qnext">Next position →</button></div>`;
    document.getElementById('qnext').onclick = () => { qi++; showQuiz(); window.scrollTo({ top: document.getElementById('quiz').offsetTop - 60, behavior: 'smooth' }); };
  }

  fetch('data/tactics.json?t=' + Date.now()).then((r) => r.json()).then((T) => {
    const sc = T.score || {};
    document.getElementById('src').textContent = `From ${T.games.toLocaleString()} games of ${T.net} (as on the Play page: always its
      favourite move) against itself, the GA base and rl-v1's champion, on random kingdoms. It scored ${P(sc['GA base'] || 0)} against the GA base and ${P(sc['rl-v1'] || 0)} against rl-v1.`;
    document.getElementById('lessons').innerHTML = lessons(T);
    Q = T.quiz; qi = Math.floor(Math.random() * Q.length);
    showQuiz();
  }).catch((e) => { document.getElementById('lessons').innerHTML = `<p class="muted">No lesson data yet (${h(e)}).</p>`; });
})();
