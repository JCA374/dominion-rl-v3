// The game page: you against a model, phone first. The Python reference engine and the networks run in the browser
// (Pyodide): play/session.py (copied to playvs/py/) runs the game; an advisor net (playvs/spec.json "advisor") gives
// your win probability and suggestions; it never plays. Everything the page does is set in playvs/spec.json.
(function () {
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
    play_action: ['Play an action?', 'No more actions'], buy: ['Buy a card', 'End turn'],
    throne: ['Throne Room: play which action twice?', 'None'], chapel_trash: ['Chapel: trash a card?', 'Stop trashing'],
    moneylender_trash: ['Moneylender: trash a Copper for +$3?', 'No'], mine_trash: ['Mine: trash a treasure to upgrade it?', 'No'],
    mine_gain: ['Mine: gain a treasure to your hand (you trashed {s})', ''], remodel_trash: ['Remodel: trash a card', ''],
    remodel_gain: ['Remodel: gain a card costing up to $2 more than {s}', ''], workshop_gain: ['Workshop: gain a card costing up to $4', ''],
    artisan_gain: ['Artisan: gain a card costing up to $5, to your hand', ''], artisan_topdeck: ['Artisan: put a card from your hand onto your deck', ''],
    cellar_discard: ['Cellar: discard a card (you draw one for each)?', 'Done'],
    harbinger_topdeck: ['Harbinger: put a card from your discard pile onto your deck?', 'No'],
    vassal_play: ['Vassal discarded {s}: play it?', 'No'], library_skip: ['Library drew {s}: set it aside?', 'Keep it'],
    sentry_card: ['Sentry revealed {s}:', ''], sentry_top: ['Sentry: which card goes on top?', ''],
    poacher_discard: ['Poacher: discard a card ({n} to go)', ''], militia_discard: ['Militia attack: discard down to 3 ({n} to go)', ''],
    bureaucrat_topdeck: ['Bureaucrat attack: put a Victory card from your hand onto your deck', ''],
    bandit_trash: ['Bandit attack: which of your treasures is trashed?', ''],
  };
  const SUPPLY_KINDS = new Set(['buy', 'mine_gain', 'remodel_gain', 'workshop_gain', 'artisan_gain']);
  const HAND_KINDS = new Set(['play_action', 'throne', 'chapel_trash', 'moneylender_trash', 'mine_trash', 'remodel_trash',
    'artisan_topdeck', 'cellar_discard', 'poacher_discard', 'militia_discard', 'bureaucrat_topdeck']);
  const TYPE = (n) => ['Copper', 'Silver', 'Gold'].includes(n) ? 't-tre' : ['Estate', 'Duchy', 'Province', 'Gardens'].includes(n) ? 't-vic'
    : n === 'Curse' ? 't-cur' : ['Militia', 'Witch', 'Bureaucrat', 'Bandit'].includes(n) ? 't-atk' : '';
  const BASE = new Set(['Copper', 'Silver', 'Gold', 'Estate', 'Duchy', 'Province', 'Curse']);
  const DEFAULT_SPEC = {
    advisor: null, toggles: { win_probability: true, suggestion: false, opponent_thinking: false },
    suggestion: { show_top: 3, mark_on_board: true }, start: { opponent: 0, seat: 'random', kingdom: 'random' },
    turn: { confirm_end_turn_from_coins: 3, show_their_turn: true }, take_back: { enabled: true },
    review: { enabled: true, min_gap: 0.25, show: 8 }, log_lines: 40, about: {},
  };
  const PREFS_KEY = 'dominion-rl-v3-play';
  const h = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const el = (id) => document.getElementById(id);
  const P = (x) => (x > 0 && x < 0.005 ? '<1' : Math.round(100 * x)) + '%';

  let spec = DEFAULT_SPEC, models = [], py = null, S = null, v = null, busy = false, booting = null;
  let prefs = {}, choice = { opp: 0, seat: 'random', seed: 0 }, lastWin = null, game = null;

  function loadPrefs() { try { prefs = JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') || {}; } catch (e) { prefs = {}; } }
  function savePrefs() { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (e) { /* private mode */ } }
  const on = (k) => (prefs[k] != null ? prefs[k] : spec.toggles[k]);
  function merge(a, b) { const o = { ...a }; for (const k in b) o[k] = b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) ? { ...(a[k] || {}), ...b[k] } : b[k]; return o; }

  function toast(t) {
    let x = document.querySelector('.toast');
    if (!x) { x = document.createElement('div'); x.className = 'toast'; document.body.appendChild(x); }
    x.textContent = t; x.hidden = false;
    clearTimeout(toast.t); toast.t = setTimeout(() => { x.hidden = true; }, 2600);
  }

  // ── setup ──────────────────────────────────────────────────────────────────────────────────────
  async function init() {
    loadPrefs();
    if (location.hash.indexOf('learn') >= 0) prefs.suggestion = true;   // from the Learn page: suggestions on (not saved)
    try { spec = merge(DEFAULT_SPEC, await (await fetch('playvs/spec.json', { cache: 'no-store' })).json()); } catch (e) { spec = DEFAULT_SPEC; }
    try { models = await (await fetch('playvs/models.json', { cache: 'no-store' })).json(); } catch (e) { models = []; }
    choice.opp = Math.min(prefs.opp != null ? prefs.opp : spec.start.opponent || 0, Math.max(0, models.length - 1));
    choice.seat = prefs.seat || spec.start.seat || 'random';
    choice.seed = 1 + Math.floor(Math.random() * 99999);
    el('g-gear').onclick = openDrawer;
    el('g-drawer').onclick = (e) => { if (e.target.id === 'g-drawer') closeDrawer(); };
    renderSetup();
    const slow = navigator.connection && (navigator.connection.saveData || /2g/.test(navigator.connection.effectiveType || ''));
    if (!slow) setTimeout(() => { boot(() => {}).catch(() => {}); }, 1200);   // warm up while you choose
  }

  function renderSetup() {
    S = null; v = null; game = null; lastWin = null;
    el('g-sheet').hidden = true;
    el('g-title').textContent = 'Play against a model';
    const fixed = Array.isArray(spec.start.kingdom);
    el('g-main').innerHTML = `<div class="g-setup">
      <h1>Play Dominion</h1>
      <p class="muted">The real rules engine and the real networks, in your browser. Treasures are played for you.
        New to the page? <a href="learn.html">Learn</a> shows how to read it and what the model has learned.</p>
      <h2>Opponent</h2>
      <div class="optlist">${models.map((m, i) => `<label class="opt${i === choice.opp ? ' on' : ''}"><input type="radio" name="opp" value="${i}"${i === choice.opp ? ' checked' : ''}>
        <b>${h(m.label)}</b><span class="d">${h(m.note)}</span></label>`).join('')}</div>
      <div class="row"><span>You play</span><span class="seg" id="seat">${['first', 'second', 'random'].map((s) =>
        `<button data-s="${s}" class="${choice.seat === s ? 'on' : ''}">${s}</button>`).join('')}</span></div>
      ${fixed ? `<p class="muted small">Kingdom fixed by the game spec: ${h(spec.start.kingdom.join(', '))}</p>` :
        `<div class="row"><label>Kingdom number <input id="seed" type="number" min="1" value="${choice.seed}"></label>
        <button class="iconbtn" id="reseed" aria-label="New random kingdom">🎲</button></div>`}
      <div class="row"><button class="big" id="start">Start a game</button></div>
      <p class="muted small" id="bootmsg"></p>
    </div>`;
    el('g-main').querySelectorAll('input[name=opp]').forEach((r) => { r.onchange = () => { choice.opp = +r.value; prefs.opp = choice.opp; savePrefs(); renderSetup(); }; });
    el('seat').querySelectorAll('button').forEach((b) => { b.onclick = () => { choice.seat = b.dataset.s; prefs.seat = choice.seat; savePrefs(); renderSetup(); }; });
    if (el('reseed')) el('reseed').onclick = () => { choice.seed = 1 + Math.floor(Math.random() * 99999); el('seed').value = choice.seed; };
    if (el('seed')) el('seed').onchange = () => { choice.seed = +el('seed').value || 1; };
    el('start').onclick = start;
  }

  async function boot(status) {
    if (py) return;
    if (booting) return booting;
    booting = (async () => {
      status('Loading Python for the browser (about 10 MB, once)…');
      if (!window.loadPyodide) await new Promise((ok, bad) => { const s = document.createElement('script'); s.src = PYODIDE + 'pyodide.js'; s.onload = ok; s.onerror = () => bad(new Error('could not load Pyodide')); document.head.appendChild(s); });
      const p = await window.loadPyodide({ indexURL: PYODIDE });
      status('Loading numpy…');
      await p.loadPackage('numpy');
      status('Loading the rules engine…');
      p.FS.mkdirTree('/home/pyodide/py/dominion_engine');
      for (const f of PY_FILES) {
        const r = await fetch('playvs/py/' + f, { cache: 'no-store' });
        if (!r.ok) throw new Error(`could not load playvs/py/${f} (HTTP ${r.status})`);
        p.FS.writeFile('/home/pyodide/py/' + f, await r.text());
      }
      p.runPython('import sys; sys.path.insert(0, "/home/pyodide/py"); import session');
      py = p;
    })();
    try { await booting; } finally { booting = null; }
  }

  async function net(file, path) {
    if (py.FS.analyzePath(path).exists) return;
    const r = await fetch('playvs/models/' + file);
    if (!r.ok) throw new Error('could not load ' + file);
    py.FS.writeFile(path, new Uint8Array(await r.arrayBuffer()));
  }

  async function start() {
    if (busy) return;
    busy = true;
    const msg = (t) => { const m = el('bootmsg'); if (m) m.textContent = t; };
    try {
      await boot(msg);
      const m = models[choice.opp];
      const seat = choice.seat === 'random' ? Math.floor(Math.random() * 2) : choice.seat === 'second' ? 1 : 0;
      const seed = choice.seed || 1;
      let flat = 'None', adv = 'None';
      if (m.file) { msg('Loading the opponent…'); await net(m.file, '/home/pyodide/opp_' + m.file); flat = `np.load("/home/pyodide/opp_${m.file}")`; }
      const af = (models.find((x) => x.file === spec.advisor) || models.find((x) => x.file) || {}).file;
      if (af) { msg('Loading the advisor…'); await net(af, '/home/pyodide/adv_' + af); adv = `np.load("/home/pyodide/adv_${af}")`; }
      const k = Array.isArray(spec.start.kingdom) ? JSON.stringify(spec.start.kingdom) : `session.kingdom_random(${seed})`;
      py.runPython(`import session, numpy as np\nS = session.Session(${k}, ${seed}, ${JSON.stringify(m.name)}, ${flat}, human=${seat}, advisor=${adv})`);
      S = py.globals.get('S');
      game = { opp: m.label, seat, seed, advisor: !!af };
      el('g-title').textContent = `vs ${m.label.split(' (')[0]} · kingdom ${seed}`;
      draw(JSON.parse(S.view()));
    } catch (e) { msg('Could not start: ' + e); }
    busy = false;
  }

  // ── the game ───────────────────────────────────────────────────────────────────────────────────
  function act(fn) {
    if (!S || busy) return;
    busy = true;
    el('g-sheet-in').innerHTML = '<div class="wait">The opponent is playing…</div>';
    setTimeout(() => { try { draw(JSON.parse(fn())); } catch (e) { toast('Error: ' + e); } finally { busy = false; } }, 15);
  }

  function answer(a) {
    if (!v || !v.question) return;
    const q = v.question, need = spec.turn.confirm_end_turn_from_coins;
    if (q.kind === 'buy' && a === -1 && need > 0 && v.coins >= need && !confirm(`End your turn with $${v.coins} unspent?`)) return;
    act(() => S.answer(a));
  }

  function label(q, id) { if (id === -1) return (ASK[q.kind] || [])[1] || 'Pass'; const o = q.options.find((x) => x.id === id); return o ? o.label : '?'; }

  function spark(wp, result) {
    const pts = wp.map(([t, w]) => [t, w]);
    if (result) pts.push([pts.length ? pts[pts.length - 1][0] + 1 : 1, result.won ? 1 : result.shared ? 0.5 : 0]);
    if (pts.length < 2) return '';
    const W = 300, H = 38, t0 = pts[0][0], t1 = pts[pts.length - 1][0] || 1;
    const X = (t) => (t1 === t0 ? 0 : (t - t0) / (t1 - t0)) * (W - 4) + 2, Y = (w) => H - 3 - w * (H - 6);
    return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Your win probability over the game">
      <line x1="0" x2="${W}" y1="${Y(0.5)}" y2="${Y(0.5)}"/><path d="${pts.map(([t, w], i) => (i ? 'L' : 'M') + X(t).toFixed(1) + ' ' + Y(w).toFixed(1)).join(' ')}"/></svg>`;
  }

  // a log line, said as a sentence: "buy: Witch" -> "bought Witch", "play action: Militia → +$2" -> "played Militia → +$2"
  const said = (s) => s.replace(/^library skip \(([^)]*)\): no$/, 'Library: kept $1').replace(/^library skip \(([^)]*)\): (.*)$/, 'Library: set aside $1').replace(/^buy: ends turn$/, 'bought nothing').replace(/^buy: /, 'bought ').replace(/^play action: /, 'played ')
    .replace(/^throne: /, 'Throne Room on ').replace(/^(\w+) (trash|gain|discard|topdeck)( \(([^)]*)\))?: /, (m, c, k, x, sub) =>
      `${c[0].toUpperCase() + c.slice(1)} ${{ trash: 'trashed', gain: 'gained', discard: 'discarded', topdeck: 'put on top' }[k]}${sub ? ' (' + sub + ')' : ''}: `);

  function theirTurn(log) {
    let i = log.length; const out = [];
    while (i > 0 && log[i - 1].who !== 'You') out.unshift(log[--i]);
    return out;
  }

  function draw(view) {
    v = view;
    const q = v.question, adv = q && v.advice, res = v.result;
    const opts = q ? new Set(q.options.map((o) => o.id)) : new Set();
    const best = adv && on('suggestion') ? adv.probs.slice(0, spec.suggestion.show_top) : [];
    const star = new Map(spec.suggestion.mark_on_board ? best.slice(0, 1).map(([id, p]) => [id, p]) : []);
    if (adv) lastWin = adv.win;
    if (res) lastWin = res.won ? 1 : res.shared ? 0.5 : 0;

    const supplyTile = (s) => {
      const can = q && SUPPLY_KINDS.has(q.kind) && opts.has(s.id);
      const cls = ['tile', TYPE(s.card), q && SUPPLY_KINDS.has(q.kind) ? (can ? 'can' : 'dim') : '', s.left === 0 ? 'empty' : ''].join(' ');
      return `<button class="${cls}" data-card="${h(s.card)}" ${can ? `data-a="${s.id}"` : ''}>
        <span class="n">${h(s.card)}</span><span class="m">$${s.cost} · ${s.left}</span>${can && star.has(s.id) ? `<span class="star">★ ${P(star.get(s.id))}</span>` : ''}</button>`;
    };
    const idOf = (name) => (v.supply.find((s) => s.card === name) || {}).id;
    const handTile = (n) => {
      const id = idOf(n), can = q && HAND_KINDS.has(q.kind) && opts.has(id);
      return `<button class="tile ${TYPE(n)} ${q && HAND_KINDS.has(q.kind) ? (can ? 'can' : 'dim') : ''}" data-card="${h(n)}" ${can ? `data-a="${id}"` : ''}>
        <span class="n">${h(n)}</span><span class="m">${h(TEXT[n] || '')}</span>${can && star.has(id) ? `<span class="star">★ ${P(star.get(id))}</span>` : ''}</button>`;
    };
    const ORDER = ['Copper', 'Silver', 'Gold', 'Estate', 'Duchy', 'Province', 'Curse'];
    const base = ORDER.map((n) => v.supply.find((s) => s.card === n)).filter(Boolean), kingdom = v.supply.filter((s) => !BASE.has(s.card));
    const prov = (v.supply.find((s) => s.card === 'Province') || {}).left;
    const tt = spec.turn.show_their_turn ? theirTurn(v.log) : [];
    const think = on('opponent_thinking') && v.thinking ? `<div class="muted small">It considered (${h(v.thinking.kind)}): ${v.thinking.top.map(([l, p]) => `${h(l)} ${P(p)}`).join(' · ')}; its own win estimate ${P((v.thinking.value + 1) / 2)}.</div>` : '';
    let html = `
      <div class="g-score"><span>You <b>${v.me.vp}</b></span><span class="vs">vs</span><span>${h(v.opp.name)} <b>${v.opp.vp}</b></span>
        <span class="muted small">turn ${v.me.turn} · ${prov} Province${prov === 1 ? '' : 's'} left</span></div>
      ${on('win_probability') && game && game.advisor && lastWin != null ? `<div class="wbar" title="The advisor's estimate"><div class="f" style="width:${(100 * lastWin).toFixed(1)}%"></div>
        <div class="l"><span>You ${P(lastWin)}</span><span>${P(1 - lastWin)}</span></div></div>${spark(v.wp, res)}` : ''}
      ${tt.length || think ? `<div class="pane their"><h3>Their last turn</h3>${tt.slice(-10).map((l) => `<div>${h(said(l.text))}</div>`).join('')}${tt.length > 10 ? `<div class="muted small">and ${tt.length - 10} earlier moves (game log)</div>` : ''}${think}</div>` : ''}
      <div class="pane"><h3>Supply</h3><div class="tiles base">${base.map(supplyTile).join('')}</div><div class="tiles">${kingdom.map(supplyTile).join('')}</div></div>
      <div class="pane"><h3>Your hand${v.actions != null ? ` · ${v.actions} action${v.actions === 1 ? '' : 's'} · ${v.buys} buy${v.buys === 1 ? '' : 's'} · $${v.coins}` : ''}</h3>
        <div class="hand">${v.me.hand.map(handTile).join('') || '<span class="muted">empty</span>'}</div>
        ${v.me.play.length ? `<div class="chips" style="margin-top:6px"><span class="muted small">In play:</span>${v.me.play.map((n) => `<span class="chip">${h(n)}</span>`).join('')}</div>` : ''}
        <div class="muted small" style="margin-top:4px">Deck ${v.me.deck} · discard ${v.me.discard}${v.me.discard_top ? ' (top: ' + h(v.me.discard_top) + ')' : ''}</div></div>
      <details class="pane"><summary class="muted small">Cards: yours and theirs</summary>
        <p class="small"><b>You</b> ${Object.entries(v.me.owned).map(([n, c]) => `${h(n)} ×${c}`).join(', ')}</p>
        <p class="small"><b>${h(v.opp.name)}</b> ${Object.entries(v.opp.owned).map(([n, c]) => `${h(n)} ×${c}`).join(', ')} · hand ${v.opp.hand}, deck ${v.opp.deck}, discard ${v.opp.discard}</p></details>`;
    if (res) html = resultPane(res) + html;
    html += `<details class="pane"><summary class="muted small">Game log</summary><div class="glog">${v.log.slice().reverse().slice(0, spec.log_lines).map((l) =>
      `<div><span class="muted">T${l.turn}</span> <b>${h(l.who)}</b> ${h(said(l.text))}</div>`).join('')}</div></details>`;
    el('g-main').innerHTML = html;
    el('g-main').querySelectorAll('.tile').forEach((b) => {
      b.onclick = () => { if (b.dataset.a != null) answer(+b.dataset.a); else toast(`${b.dataset.card}: ${TEXT[b.dataset.card] || ''}`); };
    });
    const again = el('again'); if (again) again.onclick = () => { choice.seed = 1 + Math.floor(Math.random() * 99999); start(); };
    const setup = el('tosetup'); if (setup) setup.onclick = renderSetup;
    sheet(q, adv, best);
  }

  function sheet(q, adv, best) {
    const sh = el('g-sheet');
    if (!q) { sh.hidden = true; return; }
    sh.hidden = false;
    const [txt, passTxt] = ASK[q.kind] || [q.kind, 'Pass'];
    const head = txt.replace('{s}', q.subject_name || '').replace('{n}', q.subject);
    const bestIds = new Set(best.slice(0, 1).map(([id]) => id));
    const pOf = new Map(adv && on('suggestion') ? adv.probs : []);
    let buttons;
    if (SUPPLY_KINDS.has(q.kind)) {          // the supply tiles are the answers; here only the suggestions + pass
      buttons = best.filter(([id]) => id !== -1).map(([id]) => ({ id, label: label(q, id) }));
    } else {
      buttons = q.options.slice();
    }
    const btn = (o) => `<button data-a="${o.id}" class="${bestIds.has(o.id) ? 'best' : ''}">${h(o.label)}${pOf.has(o.id) ? `<span class="p">${P(pOf.get(o.id))}</span>` : ''}</button>`;
    const pass = q.may_pass ? `<button class="pass ${bestIds.has(-1) ? 'best' : ''}" data-a="-1">${h(passTxt || 'Pass')}${pOf.has(-1) ? `<span class="p">${P(pOf.get(-1))}</span>` : ''}</button>` : '';
    const hint = SUPPLY_KINDS.has(q.kind) ? 'Tap a green card above' : HAND_KINDS.has(q.kind) ? 'or tap a green card in your hand' : '';
    el('g-sheet-in').innerHTML = `<div class="q"><span>${h(head)}</span>${q.kind === 'buy' ? `<span class="res"><b>$${v.coins}</b> · ${v.buys} buy${v.buys > 1 ? 's' : ''}</span>` : ''}
        <span class="muted small">${hint}</span></div>
      ${best.length ? `<div class="sugg">The model suggests: ${best.map(([id, p], i) => `${i ? '' : '<b>'}${h(label(q, id))} ${P(p)}${i ? '' : '</b>'}`).join(' · ')}</div>` : ''}
      <div class="btns">${buttons.map(btn).join('')}${pass}</div>`;
    el('g-sheet-in').querySelectorAll('button[data-a]').forEach((b) => { b.onclick = () => answer(+b.dataset.a); });
  }

  function resultPane(r) {
    let rev = '';
    if (spec.review.enabled && v.review && game.advisor) {
      const bad = v.review.map((x) => ({ ...x, gap: x.p_best - x.p_chose })).filter((x) => x.gap >= spec.review.min_gap && x.chose !== x.best)
        .sort((a, b) => b.gap - a.gap).slice(0, spec.review.show);
      rev = `<h3>Review: where the model would have played differently</h3>` + (bad.length ? `<ul class="review">${bad.map((x) =>
        `<li><b>Turn ${x.turn}</b>, ${h(x.kind.replace(/_/g, ' '))}${x.subject ? ` (${h(x.subject)})` : ''}: you chose <b>${h(x.chose)}</b> (${P(x.p_chose)}),
          the model <b>${h(x.best)}</b> (${P(x.p_best)}). Your win chance then: ${P(x.win)}.</li>`).join('')}</ul>`
        : `<p class="muted">None: every move you made was close to the model's choice.</p>`);
    }
    return `<div class="pane result"><h2>${r.won ? 'You win' : r.shared ? 'Shared win' : 'You lose'}: ${r.scores[0]} – ${r.scores[1]} VP</h2>
      <p class="muted small">${Array.isArray(r.turns) ? Math.max(...r.turns) : r.turns} turns. Its final deck: ${Object.entries(r.opp_deck).map(([n, c]) => `${h(n)} ×${c}`).join(', ')}</p>
      ${rev}<div class="row"><button class="big" id="again">Play again (new kingdom)</button><button class="big ghost" id="tosetup">Change opponent</button></div></div>`;
  }

  // ── settings drawer ────────────────────────────────────────────────────────────────────────────
  function openDrawer() {
    const sw = (k, t, d) => `<label class="switch"><span>${t}<br><span class="d">${d}</span></span><input type="checkbox" class="tg" data-k="${k}"${on(k) ? ' checked' : ''}></label>`;
    const flat = [];
    const walk = (o, p) => { for (const [k, x] of Object.entries(o)) { if (k === 'about' || k === '_about') continue; const key = p ? p + '.' + k : k; if (x && typeof x === 'object' && !Array.isArray(x)) walk(x, key); else flat.push([key, x]); } };
    walk(spec, '');
    const canUndo = spec.take_back.enabled && S && v && v.answers > 0 && !v.result;
    el('g-panel').innerHTML = `<div class="row" style="justify-content:space-between;margin-top:0"><b>Settings</b><button class="iconbtn" id="g-close" aria-label="Close">✕</button></div>
      ${sw('win_probability', 'Win probability', "The advisor's chance that you win, with a graph of the game.")}
      ${sw('suggestion', 'Suggestion', "The advisor's favourite moves, with a ★ on the board.")}
      ${sw('opponent_thinking', "Opponent's thinking", 'What the opponent network weighed on its last move.')}
      <div class="row">${canUndo ? '<button class="big ghost" id="g-undo">↶ Take back my last move</button>' : ''}
        ${S ? '<button class="big ghost" id="g-new">New game</button>' : ''}</div>
      <details><summary><b>Game spec</b> <span class="muted small">(playvs/spec.json)</span></summary>
        <p class="muted small">${h(spec._about || '')}</p>
        <dl class="spec">${flat.map(([k, x]) => `<dt>${h(k)}</dt><dd><b>${h(JSON.stringify(x))}</b> ${h((spec.about || {})[k] || '')}</dd>`).join('')}</dl></details>`;
    el('g-panel').querySelectorAll('.tg').forEach((c) => { c.onchange = () => { prefs[c.dataset.k] = c.checked; savePrefs(); if (v) draw(v); }; });
    el('g-close').onclick = closeDrawer;
    if (el('g-undo')) el('g-undo').onclick = () => { closeDrawer(); act(() => S.undo(1)); };
    if (el('g-new')) el('g-new').onclick = () => { closeDrawer(); renderSetup(); };
    el('g-drawer').classList.add('on');
  }
  function closeDrawer() { el('g-drawer').classList.remove('on'); }

  // #auto: start and play the advisor's moves to the end (a headless test of the whole page)
  init().then(async () => {
    if (location.hash.indexOf('auto') < 0) return;
    prefs.suggestion = true;
    await start();
    for (let i = 0; i < 400 && S; i++) {
      const x = JSON.parse(S.view());
      if (!x.question) { draw(x); break; }
      const a = x.advice ? x.advice.probs[0][0] : x.question.options[0].id;
      draw(JSON.parse(S.answer(a)));
      if (i === 20) draw(JSON.parse(S.undo(1)));
    }
    document.title = 'AUTO DONE';
  });
})();
