// Play against a model: the Python reference engine and the network (play/rlplay.py) in Pyodide. Very plain on purpose.
(function () {
  const root = document.getElementById('vs');
  if (!root) return;
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  const PY_FILES = ['dominion_engine/__init__.py', 'dominion_engine/cards.py', 'dominion_engine/rules.py',
    'dominion_engine/rng.py', 'dominion_engine/game.py', 'dominion_engine/bots.py', 'rlplay.py', 'session.py'];
  const TEXT = {
    Copper: '$1', Silver: '$2', Gold: '$3', Estate: '1 VP', Duchy: '3 VP', Province: '6 VP', Curse: '−1 VP',
    Village: '+1 Card, +2 Actions', Smithy: '+3 Cards', Market: '+1 Card, +1 Action, +1 Buy, +$1',
    Laboratory: '+2 Cards, +1 Action', Festival: '+2 Actions, +1 Buy, +$2', Chapel: 'Trash up to 4 cards from your hand',
    'Throne Room': 'Play an action from your hand twice', 'Council Room': '+4 Cards, +1 Buy; each other player draws a card',
    Moneylender: 'You may trash a Copper from your hand for +$3', Gardens: '1 VP per 10 cards you have',
    Mine: 'Trash a treasure from your hand; gain one costing up to $3 more, to your hand',
    Merchant: '+1 Card, +1 Action; the first time you play a Silver this turn, +$1',
    Militia: '+$2; each other player discards down to 3 cards', Witch: '+2 Cards; each other player gains a Curse',
    Moat: '+2 Cards; reaction: blocks attacks', Cellar: '+1 Action; discard any number of cards, then draw that many',
    Harbinger: '+1 Card, +1 Action; you may put a card from your discard pile onto your deck',
    Vassal: '+$2; discard the top card of your deck; if it is an action, you may play it',
    Workshop: 'Gain a card costing up to $4', Bureaucrat: 'Gain a Silver onto your deck; others put a Victory card from hand onto their deck',
    Poacher: '+1 Card, +1 Action, +$1; discard a card per empty Supply pile', Remodel: 'Trash a card from your hand; gain one costing up to $2 more',
    Library: 'Draw until you have 7 cards; you may set aside action cards', Sentry: '+1 Card, +1 Action; look at the top 2 cards: trash, discard or put back',
    Artisan: 'Gain a card costing up to $5 to your hand; put a card from your hand onto your deck',
    Bandit: 'Gain a Gold; others reveal 2 cards and trash a treasure other than Copper',
  };
  const ASK = {
    play_action: ['Play an action card?', 'No more actions'],
    buy: ['Buy a card', 'End turn'],
    throne: ['Throne Room: which action do you play twice?', 'None'],
    chapel_trash: ['Chapel: trash a card?', 'Stop trashing'],
    moneylender_trash: ['Moneylender: trash a Copper for +$3?', 'No'],
    mine_trash: ['Mine: trash a treasure to upgrade it?', 'No'],
    mine_gain: ['Mine: gain a treasure to your hand (you trashed {s})', ''],
    remodel_trash: ['Remodel: trash a card', ''],
    remodel_gain: ['Remodel: gain a card costing up to $2 more than {s}', ''],
    workshop_gain: ['Workshop: gain a card costing up to $4', ''],
    artisan_gain: ['Artisan: gain a card costing up to $5, to your hand', ''],
    artisan_topdeck: ['Artisan: put a card from your hand onto your deck', ''],
    cellar_discard: ['Cellar: discard a card (you draw one for each)?', 'Done'],
    harbinger_topdeck: ['Harbinger: put a card from your discard pile onto your deck?', 'No'],
    vassal_play: ['Vassal discarded {s}: play it?', 'No'],
    library_skip: ['Library drew {s}: set it aside instead of keeping it?', 'Keep it'],
    sentry_card: ['Sentry revealed {s}:', ''],
    sentry_top: ['Sentry: which card goes on top?', ''],
    poacher_discard: ['Poacher: discard a card ({n} to go)', ''],
    militia_discard: ['Militia attack: discard down to 3 ({n} to go)', ''],
    bureaucrat_topdeck: ['Bureaucrat attack: put a Victory card from your hand onto your deck', ''],
    bandit_trash: ['Bandit attack: which of your treasures is trashed?', ''],
  };
  let py = null, models = [], S = null, busy = false;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const card = (n, extra) => `<span class="vcard" title="${esc(TEXT[n] || '')}">${esc(n)}${extra ? ' <b>' + extra + '</b>' : ''}</span>`;
  const list = (o) => Object.entries(o).map(([n, c]) => card(n, c > 1 ? '×' + c : '')).join(' ') || '<span class="muted">none</span>';

  async function setup() {
    try { models = await (await fetch('playvs/models.json', {cache: 'no-store'})).json(); } catch (e) { models = []; }
    root.innerHTML = `<div class="vsetup">
      <label>Opponent <select id="vs-opp">${models.map((m, i) => `<option value="${i}">${esc(m.label)}</option>`).join('')}</select></label>
      <label>You play <select id="vs-seat"><option value="0">first</option><option value="1">second</option></select></label>
      <label>Kingdom number <input id="vs-seed" type="number" min="1" value="${1 + Math.floor(Math.random() * 99999)}"></label>
      <button id="vs-start">Start a game</button></div>
      <p class="muted small" id="vs-note">${models[0] ? esc(models[0].note) : ''}</p><div id="vs-board"></div>`;
    document.getElementById('vs-opp').onchange = (e) => { document.getElementById('vs-note').textContent = models[e.target.value].note; };
    document.getElementById('vs-start').onclick = start;
  }

  async function boot(status) {
    if (py) return;                                   // set only once everything below has worked
    status('Loading Python for the browser…');
    if (!window.loadPyodide) await new Promise((ok, bad) => { const s = document.createElement('script'); s.src = PYODIDE + 'pyodide.js'; s.onload = ok; s.onerror = () => bad(new Error('could not load Pyodide')); document.head.appendChild(s); });
    const p = await window.loadPyodide({indexURL: PYODIDE});
    status('Loading numpy…');
    await p.loadPackage('numpy');
    status('Loading the rules engine…');
    p.FS.mkdirTree('/home/pyodide/py/dominion_engine');
    for (const f of PY_FILES) {
      const r = await fetch('playvs/py/' + f, {cache: 'no-store'});
      if (!r.ok) throw new Error(`could not load playvs/py/${f} (HTTP ${r.status})`);
      p.FS.writeFile('/home/pyodide/py/' + f, await r.text());
    }
    p.runPython('import sys; sys.path.insert(0, "/home/pyodide/py"); import session');
    py = p;
  }

  async function start() {
    if (busy) return;
    busy = true;
    const board = document.getElementById('vs-board');
    const status = (t) => { board.innerHTML = `<p class="muted">${esc(t)}</p>`; };
    try {
      await boot(status);
      const m = models[document.getElementById('vs-opp').value];
      const seat = +document.getElementById('vs-seat').value;
      const seed = +document.getElementById('vs-seed').value || 1;
      let flat = 'None';
      if (m.file) {
        status('Loading the network…');
        py.FS.writeFile('/home/pyodide/m.npy', new Uint8Array(await (await fetch('playvs/models/' + m.file)).arrayBuffer()));
        flat = '__import__("numpy").load("/home/pyodide/m.npy")';
      }
      py.runPython(`import session\nS = session.Session(session.kingdom_random(${seed}), ${seed}, ${JSON.stringify(m.name)}, ${flat}, human=${seat})`);
      S = py.globals.get('S');
      draw(JSON.parse(S.view()));
    } catch (e) { status('Could not start: ' + e); }
    busy = false;
  }

  function answer(a) {
    if (!S || busy) return;
    busy = true;
    setTimeout(() => { try { draw(JSON.parse(S.answer(a))); } finally { busy = false; } }, 10);
  }

  function draw(v) {
    const b = document.getElementById('vs-board');
    const q = v.question;
    const opts = q ? new Set(q.options.map((o) => o.id)) : new Set();
    const supply = v.supply.map((s) => `<div class="vpile${q && q.kind.indexOf('gain') >= 0 || (q && q.kind === 'buy') ? (opts.has(s.id) ? ' can' : ' dim') : ''}${s.left === 0 ? ' empty' : ''}" title="${esc(TEXT[s.card] || '')}">
        <div><b>${esc(s.card)}</b></div><div class="muted small">$${s.cost} · ${s.left} left</div></div>`).join('');
    let ask = '';
    if (q) {
      const [txt, passTxt] = ASK[q.kind] || [q.kind, 'Pass'];
      const head = txt.replace('{s}', q.subject_name || '').replace('{n}', q.subject);
      let buttons = q.options.slice();
      if (q.kind === 'buy') buttons.sort((x, y) => (v.supply.find((s) => s.id === y.id).cost - v.supply.find((s) => s.id === x.id).cost));
      ask = `<div class="vask"><b>${esc(head)}</b>${q.kind === 'buy' ? ` <span class="muted">($${v.coins}, ${v.buys} buy${v.buys > 1 ? 's' : ''})</span>` : ''}
        <div class="vbtns">${buttons.map((o) => `<button data-a="${o.id}" title="${esc(TEXT[o.label] || '')}">${esc(o.label)}</button>`).join('')}
        ${q.may_pass ? `<button class="pass" data-a="-1">${esc(passTxt || 'Pass')}</button>` : ''}</div></div>`;
    }
    let end = '';
    if (v.result) {
      const r = v.result;
      end = `<div class="vask"><b>${r.won ? 'You win' : r.shared ? 'Shared win' : 'You lose'}: ${r.scores[0]} – ${r.scores[1]} VP</b>
        <div class="muted small">Its final deck: ${list(r.opp_deck)}</div>
        <div class="vbtns"><button id="vs-again">Play again</button></div></div>`;
    }
    const th = v.thinking ? `<p class="muted small">Its last choice (${esc(v.thinking.kind)}): ${v.thinking.top.map(([l, p]) => `${esc(l)} ${Math.round(100 * p)}%`).join(' · ')}.
      Its win estimate: ${Math.round(50 * (v.thinking.value + 1))}%.</p>` : '';
    const log = v.log.slice().reverse().slice(0, 14).map((l) => `<div><span class="muted">T${l.turn}</span> <b>${l.who}</b> ${esc(l.text)}</div>`).join('');
    b.innerHTML = `
      <div class="vrow"><b>${esc(v.opp.name)}</b> · ${v.opp.vp} VP · turn ${v.opp.turn} · hand ${v.opp.hand} · deck ${v.opp.deck} · discard ${v.opp.discard}
        ${v.opp.play.length ? '<br>In play: ' + v.opp.play.map((n) => card(n)).join(' ') : ''}
        <details><summary class="muted small">Its cards</summary>${list(v.opp.owned)}</details></div>
      <div class="vsupply">${supply}</div>
      <div class="vrow"><b>You</b> · ${v.me.vp} VP · turn ${v.me.turn}${v.actions != null ? ` · <b>${v.actions}</b> action${v.actions === 1 ? "" : "s"} · <b>${v.buys}</b> buy${v.buys === 1 ? "" : "s"} · <b>$${v.coins}</b>` : ''}
        · deck ${v.me.deck} · discard ${v.me.discard}${v.me.discard_top ? ' (top: ' + esc(v.me.discard_top) + ')' : ''}</div>
      <div class="vrow">Hand: ${v.me.hand.map((n) => card(n)).join(' ') || '<span class="muted">empty</span>'}
        ${v.me.play.length ? '<br>In play: ' + v.me.play.map((n) => card(n)).join(' ') : ''}</div>
      ${ask}${end}${th}
      <details><summary class="muted small">Your cards</summary>${list(v.me.owned)}</details>
      <div class="vlog">${log}</div>`;
    b.querySelectorAll('.vbtns button[data-a]').forEach((x) => { x.onclick = () => answer(+x.dataset.a); });
    const again = document.getElementById('vs-again');
    if (again) again.onclick = () => { document.getElementById('vs-seed').value = 1 + Math.floor(Math.random() * 99999); start(); };
  }

  // #vsauto: start a game and make 30 random moves (for a headless test of the whole page)
  setup().then(async () => {
    if (location.hash.indexOf('vsauto') < 0) return;
    await start();
    for (let i = 0; i < 30 && S; i++) {
      const v = JSON.parse(S.view());
      if (!v.question) break;
      const o = v.question.options.map((x) => x.id).concat(v.question.may_pass ? [-1] : []);
      draw(JSON.parse(S.answer(o[i % o.length])));
    }
  });
})();
