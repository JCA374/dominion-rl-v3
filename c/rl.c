/*
 * rl.c -- one network answers EVERY decision (buys, actions, trashing, gains, discards): the agent
 * of rl-v1, trained by imitation (behaviour cloning of the GA genome) and then by policy-gradient
 * reinforcement learning (PPO) without any search at play time.
 *
 * Built into rl.so with engine/c/engine.c. Includes az-v1's az.c (which includes dominion-v1's
 * ga_agent.c) so that the GA genome, the AlphaZero-light agent and this agent can meet in a game.
 */
#include "../az-v1/c/az.c"

/* ── features: what `me` knows at any decision ──
 * az_features (175: own cards, own draw pile, opponent's cards, supply, in game, coins, buys,
 * turns, VP) + my hand (33) + my cards in play (33) + actions left + my turn + decision kind (22)
 * + the card the decision is about (33). */
#define RL_NF0 (AZ_NF + 2 * DE_NUM_CARDS + 2 + DE_NUM_KINDS + DE_NUM_CARDS)   /* 298: rl-v1's inputs */
#define RL_NH_NEXT 11                                                          /* helper inputs, see rl_helpers */
#define RL_NH_DECK 5
#define RL_NF (RL_NF0 + RL_NH_NEXT + RL_NH_DECK + 3 * DE_NUM_CARDS)              /* 413 */
#define RL_NA (DE_NUM_CARDS + 1)          /* option value (a card id, or a Sentry code), then PASS */
#define RL_PASS DE_NUM_CARDS

int rl_num_features(void) { return RL_NF; }
int rl_num_features_base(void) { return RL_NF0; }
int rl_num_actions(void) { return RL_NA; }
int rl_num_params(int nh, int nf) { return nh * nf + nh + nh * nh + nh + RL_NA * nh + RL_NA + nh + 1; }

/* ── helper inputs: what the next hand will bring, and what a card would change ──
 * Exact, from what `me` knows: the next hand is the first 5 cards of the draw pile (its contents are known, not its
 * order); if fewer than 5 are left, the rest come from everything else reshuffled (discard + hand + in play). During
 * the opponent's turn the next hand is the current hand. Coins of a card = treasure value or the action's +$. */
static double choose(int n, int k) {
    if (k < 0 || k > n) return 0;
    double r = 1;
    for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1);
    return r;
}

/* distribution of the coins of k cards drawn from `cnt` (counts per card id) into P[0..HS-1]. k can exceed 5: during
 * the opponent's turn the next hand is the current hand, which may hold more (Council Room, Library). */
#define HK 16                                 /* most cards drawn */
#define HS (3 * HK + 1)                       /* coin sums 0..48 */
static void draw_dist(const int *cnt, int k, double *P) {
    if (k > HK - 1) k = HK - 1;
    int nv[4] = {0}, n = 0;
    for (int c = 0; c < DE_NUM_CARDS; c++) { nv[DE_CARDS[c].coins] += cnt[c]; n += cnt[c]; }
    double f[HK][HS] = {{0}};
    f[0][0] = 1;
    for (int v = 0; v < 4; v++) {
        double g[HK][HS] = {{0}};
        for (int j = 0; j <= k; j++)
            for (int s = 0; s < HS; s++) {
                if (f[j][s] == 0) continue;
                for (int t = 0; t <= nv[v] && j + t <= k; t++) g[j + t][s + v * t] += f[j][s] * choose(nv[v], t);
            }
        memcpy(f, g, sizeof f);
    }
    double tot = choose(n, k);
    for (int s = 0; s < HS; s++) P[s] = tot > 0 ? f[k][s] / tot : (s == 0);
}

static void rl_helpers(const de_game *g, int me, float *h) {
    const de_player *p = &g->p[me];
    const int N = DE_NUM_CARDS;
    int deck[DE_NUM_CARDS] = {0}, pool[DE_NUM_CARDS] = {0}, own[DE_NUM_CARDS] = {0};
    int a, b, nd = 0, np = 0;
    if (g->current != me) {                       /* next hand = this hand */
        for (int i = 0; i < p->hand_n; i++) deck[p->hand[i]]++;
        nd = p->hand_n; a = nd; b = 0;
    } else {
        for (int i = 0; i < p->deck_n; i++) deck[p->deck[i]]++;
        for (int i = 0; i < p->discard_n; i++) pool[p->discard[i]]++;
        for (int i = 0; i < p->hand_n; i++) pool[p->hand[i]]++;
        for (int i = 0; i < p->play_n; i++) pool[p->play[i]]++;
        nd = p->deck_n; np = p->discard_n + p->hand_n + p->play_n;
        a = nd < 5 ? nd : 5; b = 5 - a < np ? 5 - a : np;
    }
    double Pa[HS], Pb[HS], P[HS] = {0};
    draw_dist(deck, a, Pa);
    draw_dist(pool, b, Pb);
    for (int i = 0; i < HS; i++) for (int j = 0; i + j < HS; j++) P[i + j] += Pa[i] * Pb[j];
    double ec = 0, p3 = 0, p5 = 0, p6 = 0, p8 = 0, evp = 0;
    int prov = g->supply[DE_PROVINCE] > 0, duchy = g->supply[DE_DUCHY] > 0, est = g->supply[DE_ESTATE] > 0;
    for (int s = 0; s < HS; s++) {
        ec += s * P[s];
        if (s >= 3) p3 += P[s];
        if (s >= 5) p5 += P[s];
        if (s >= 6) p6 += P[s];
        if (s >= 8) p8 += P[s];
        double v = s >= 8 && prov ? 6 : s >= 5 && duchy ? 3 : s >= 2 && est ? 1 : 0;
        evp += v * P[s];
    }
    /* linear expectations: a cards from the draw pile, b from the reshuffle */
    double ea = 0, eact = 0, ebuy = 0, ecard = 0;
    for (int c = 0; c < N; c++) {
        double w = (nd ? (double)a * deck[c] / nd : 0) + (np ? (double)b * pool[c] / np : 0);
        if (DE_CARDS[c].types & DE_ACTION) ea += w;
        eact += w * DE_CARDS[c].actions;
        ebuy += w * DE_CARDS[c].buys;
        ecard += w * DE_CARDS[c].cards;
    }
    int vp = de_score(g, me);
    h[0] = (float)(ec / 8); h[1] = (float)p3; h[2] = (float)p5; h[3] = (float)p6; h[4] = (float)p8;
    h[5] = (float)(ea / 3); h[6] = (float)(eact / 3); h[7] = (float)(ebuy / 2); h[8] = (float)(ecard / 5);
    h[9] = (float)(evp / 6); h[10] = (float)((vp + evp) / 20);
    /* the whole deck, per 5 cards; and what one more copy of each card would change */
    int n = 0;
    double S[5] = {0};
    for (int i = 0; i < p->deck_n; i++) own[p->deck[i]]++;
    for (int i = 0; i < p->hand_n; i++) own[p->hand[i]]++;
    for (int i = 0; i < p->discard_n; i++) own[p->discard[i]]++;
    for (int i = 0; i < p->play_n; i++) own[p->play[i]]++;
    for (int c = 0; c < N; c++) {
        n += own[c];
        S[0] += own[c] * DE_CARDS[c].coins;
        S[1] += own[c] * ((DE_CARDS[c].types & DE_ACTION) != 0);
        S[2] += own[c] * DE_CARDS[c].actions;
        S[3] += own[c] * DE_CARDS[c].buys;
        S[4] += own[c] * DE_CARDS[c].cards;
    }
    float *d = h + RL_NH_NEXT;
    const double scale[5] = {8, 3, 3, 2, 5};
    for (int k = 0; k < 5; k++) d[k] = n ? (float)(5 * S[k] / n / scale[k]) : 0;
    float *dl = d + RL_NH_DECK;
    for (int c = 0; c < N; c++) {
        if (g->supply[c] < 0 || !n) { dl[c] = dl[N + c] = dl[2 * N + c] = 0; continue; }
        double x[3] = {DE_CARDS[c].coins, DE_CARDS[c].actions, DE_CARDS[c].buys}, base[3] = {S[0], S[2], S[3]};
        for (int k = 0; k < 3; k++) dl[k * N + c] = (float)(2 * 5 * ((base[k] + x[k]) / (n + 1) - base[k] / n));
    }
}

void rl_features(const de_game *g, int me, int kind, int subject, float *x) {
    const de_player *p = &g->p[me];
    const int N = DE_NUM_CARDS;
    az_features(g, me, x);
    float *e = x + AZ_NF;
    memset(e, 0, (RL_NF - AZ_NF) * sizeof(float));
    for (int i = 0; i < p->hand_n; i++) e[p->hand[i]] += 0.2f;
    for (int i = 0; i < p->play_n; i++) e[N + p->play[i]] += 0.2f;
    e[2 * N] = g->current == me ? g->actions / 3.0f : 0;
    e[2 * N + 1] = g->current == me;
    e[2 * N + 2 + kind] = 1;
    if (subject >= 0 && subject < N) e[2 * N + 2 + DE_NUM_KINDS + subject] = 1;
    rl_helpers(g, me, x + RL_NF0);
}

/* x -> relu(W1 x + b1) -> relu(W2 h + b2) -> logits (Wp), value tanh(Wv) */
/* nf: how many of the inputs the net reads (RL_NF0 for nets without the helper inputs) */
static float rl_eval(const float *prm, int nh, int nf, const float *x, float *logits) {
    float h1[1024], h2[1024];
    const float *w1 = prm, *b1 = w1 + nh * nf, *w2 = b1 + nh, *b2 = w2 + nh * nh;
    const float *wp = b2 + nh, *bp = wp + RL_NA * nh, *wv = bp + RL_NA, *bv = wv + nh;
    dense(w1, b1, x, nf, nh, h1, 1);
    dense(w2, b2, h1, nh, nh, h2, 1);
    dense(wp, bp, h2, nh, RL_NA, logits, 0);
    float v = bv[0];
    for (int i = 0; i < nh; i++) v += wv[i] * h2[i];
    return tanhf(v);
}

float rl_forward(const float *prm, int nh, int nf, const float *x, float *logits) { return rl_eval(prm, nh, nf, x, logits); }

/* ── records (the training data) ── */
typedef struct {
    float *X, *LOGP, *V;
    unsigned char *LEGAL;
    int *A, *SEAT, *TURN, *KIND;
    int max, n;
} rl_rec;
static rl_rec *r_rec;
static int *r_plan;               /* per record: the planner's choice (or -1 = not planned); rl_set_plan_buf */
static float *r_pval;             /* per record: RL_NA planner values (NAN = not an option) */

void rl_set_plan_buf(int *plan, float *pval) { r_plan = plan; r_pval = pval; }

static void rl_record(const de_game *g, int me, int kind, int subject, const int *ok, int a, float logp, float v) {
    rl_rec *r = r_rec;
    if (!r || r->n >= r->max) return;
    int k = r->n++;
    rl_features(g, me, kind, subject, r->X + (size_t)k * RL_NF);
    for (int i = 0; i < RL_NA; i++) r->LEGAL[(size_t)k * RL_NA + i] = (unsigned char)ok[i];
    r->A[k] = a; r->LOGP[k] = logp; r->V[k] = v; r->SEAT[k] = me;
    r->TURN[k] = g->p[me].turns + 1; r->KIND[k] = kind;
    if (r_plan) r_plan[k] = -1;
}

/* ── seats ── */
enum { RL_GA, RL_AZ, RL_NET };

typedef struct {
    int mode;                   /* RL_GA: the genome plays all; RL_AZ: az-v1's agent; RL_NET: this net */
    const int *genome;
    const float *prm;           /* RL_NET weights */
    int nh, sample, record;     /* sample: draw from the policy (training) instead of the argmax */
    float temp;
    az_agent az;                /* RL_AZ */
    uint64_t rng;
    int decisions;
    int nf;                     /* inputs the net reads (RL_NF0 or RL_NF) */
    int plan_k;                 /* >0: the turn planner (K determinisations) looks at this seat's action decisions */
    int plan_play;              /* 1: play the planner's choice; 0: only label the decision (records) */
} rl_seat;

/* ── the turn planner (rl-v3) ──
 * At an action decision (DE_PLAY_ACTION) it tries every option: for each of K determinisations of what `me`
 * cannot see (my draw order, the opponent's hand: de_hide), the copied game plays the option, then the rest of
 * the turn (further actions, treasures, buys, cleanup) with the planner net answering every decision greedily
 * (de_resume_action). The leaf is the planner net's value of the position after the turn, from my side (the
 * opponent to move). Every option sees the same K determinisations (common random numbers). The planner net is
 * rl_set_planner's, or else the seat's own net. */
static int g_in_plan;                 /* >0 inside a planner rollout: no records, no trace, no planning */
static const float *g_plan_prm;
static int g_plan_nh, g_plan_nf;
static long g_plan_rollouts;
static int g_plan_leaf;               /* 0: the planner net's value after the turn; 1: coins at my first buy decision
                                       * (+$1 per extra buy), the value only breaking ties ("coins judge") */
static int g_plan_me, g_plan_coins;   /* inside a rollout: the planning player, coins seen at their first buy (-1) */

void rl_set_planner(const float *prm, int nh, int nf) { g_plan_prm = prm; g_plan_nh = nh; g_plan_nf = nf; }
void rl_set_plan_leaf(int leaf) { g_plan_leaf = leaf; }

/* The chain rule (an experiment against the planner): at an action decision with one action left, a terminal
 * choice (no +Actions, not Throne Room) is replaced by the net's favourite +Actions option, if there is one. */
static int g_chain_rule[2];
static long g_chain_fired;
void rl_set_chain_rule(int s0, int s1) { g_chain_rule[0] = s0; g_chain_rule[1] = s1; }
static int rl_chain_rule(const de_game *g, int me, int kind, int a, const int *ok, const float *lg) {
    if (g_in_plan || !g_chain_rule[me & 1] || kind != DE_PLAY_ACTION || g->actions != 1 || a == RL_PASS) return a;
    if (DE_CARDS[a].actions > 0 || a == DE_THRONE_ROOM) return a;
    int b = -1;
    for (int i = 0; i < DE_NUM_CARDS; i++)
        if (ok[i] && DE_CARDS[i].actions > 0 && (b < 0 || lg[i] > lg[b])) b = i;
    if (b >= 0) g_chain_fired++;
    return b < 0 ? a : b;
}
long rl_chain_fired(void) { return g_chain_fired; }
long rl_plan_rollouts(void) { return g_plan_rollouts; }

static int rl_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass);

static float rl_leaf(de_game *c, int me, const float *prm, int nh, int nf) {
    if (c->over) {
        int w = de_winners(c);
        return w == 3 ? 0.f : (w >> me & 1) ? 1.f : -1.f;
    }
    c->current = (me + 1) % c->nplayers;          /* the opponent to move: my next hand is my hand */
    c->actions = 1; c->buys = 1; c->coins = 0; c->merchants = 0;
    float x[RL_NF], lg[RL_NA];
    rl_features(c, me, DE_PLAY_ACTION, -1, x);
    return rl_eval(prm, nh, nf, x, lg);
}

/* val[RL_NA]: mean leaf value per legal option (NAN otherwise). Returns the best option (RL_PASS = end the
 * action phase). */
static int rl_plan(const rl_seat *s, const de_game *g, int me, const int *ok, int K, uint64_t *rng, float *val) {
    const float *prm = g_plan_prm ? g_plan_prm : s->prm;
    int nh = g_plan_prm ? g_plan_nh : s->nh, nf = g_plan_prm ? g_plan_nf : s->nf;
    rl_seat rs[2];
    for (int i = 0; i < 2; i++) {
        memset(&rs[i], 0, sizeof rs[i]);
        rs[i].mode = RL_NET; rs[i].genome = s->genome; rs[i].prm = prm; rs[i].nh = nh; rs[i].nf = nf;
        rs[i].temp = 1; rs[i].rng = 1;
    }
    de_agent ag[2] = {{rl_decide, &rs[0]}, {rl_decide, &rs[1]}};
    double sum[RL_NA] = {0};
    static de_game c;                             /* (one game per process at a time) */
    g_in_plan++;
    for (int k = 0; k < K; k++) {
        uint64_t seed = de_rng_next(rng) | 1;
        for (int a = 0; a < RL_NA; a++) {
            if (!ok[a]) continue;
            c = *g;
            de_hide(&c, me, seed);
            g_plan_me = me; g_plan_coins = -1;
            de_resume_action(&c, ag, a == RL_PASS ? -1 : a, 0);
            float v = rl_leaf(&c, me, prm, nh, nf);
            sum[a] += g_plan_leaf == 1 ? (g_plan_coins < 0 ? 0 : g_plan_coins) + 0.001 * v : v;
            g_plan_rollouts++;
        }
    }
    g_in_plan--;
    int best = -1;
    for (int a = 0; a < RL_NA; a++) {
        val[a] = ok[a] ? (float)(sum[a] / K) : NAN;
        if (ok[a] && (best < 0 || val[a] > val[best])) best = a;
    }
    return best;
}

/* the planner's say at a decision of seat s (or -1 when it does not plan this one); fills val */
static int rl_plan_hook(rl_seat *s, const de_game *g, int kind, int me, const int *ok, int nopt, float *val) {
    if (g_in_plan || s->plan_k <= 0 || kind != DE_PLAY_ACTION || nopt < 2 || (!g_plan_prm && !s->prm)) return -1;
    return rl_plan(s, g, me, ok, s->plan_k, &s->rng, val);
}

static void rl_record_plan(int pa, const float *val) {
    if (!r_rec || !r_rec->n) return;
    int k = r_rec->n - 1;
    if (r_plan) r_plan[k] = pa;
    if (r_pval) memcpy(r_pval + (size_t)k * RL_NA, val, RL_NA * sizeof(float));
}

/* Card school (curriculum): a seat's first buys are forced to a card it rarely buys (g_force_card, up to
 * g_force_n copies, in its first g_force_turns turns) and those buys are not recorded: the net does not learn to buy
 * the card from them, it only gets to play a deck that holds it. */
static int g_force_card[2] = {-1, -1}, g_force_n = 2, g_force_turns = 6;
void rl_set_force(int c0, int c1, int n, int turns) {
    g_force_card[0] = c0; g_force_card[1] = c1; g_force_n = n; g_force_turns = turns;
}
static int rl_forced_buy(const de_game *g, int me, const int *opts, int n) {
    int c = g_force_card[me & 1];
    if (g_in_plan || c < 0 || g->p[me].turns >= g_force_turns || de_count(&g->p[me], c) >= g_force_n) return -1;
    for (int i = 0; i < n; i++) if (opts[i] == c) return c;
    return -1;
}

static int rl_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass) {
    rl_seat *s = ctx;
    if (g_in_plan && kind == DE_BUY && me == g_plan_me && g_plan_coins < 0)
        g_plan_coins = g->coins + (g->buys - 1);
    if (s->mode == RL_AZ) return az_decide(&s->az, g, kind, me, subject, opts, n, may_pass);
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    int nopt = n + (may_pass != 0);
    if (s->mode == RL_GA) {
        ga_agent ga = {s->genome, 0};
        int c = ga_decide(&ga, g, kind, me, subject, opts, n, may_pass);
        if (s->record && nopt > 1) {
            rl_record(g, me, kind, subject, ok, c < 0 ? RL_PASS : c, 0, 0);
            float val[RL_NA];
            int pa = rl_plan_hook(s, g, kind, me, ok, nopt, val);
            if (pa >= 0) rl_record_plan(pa, val);
        }
        if (kind == DE_BUY && !g_in_plan) {           /* az-v1's clip trace (one row per buy) */
            float pi[AZ_NA] = {0};
            pi[c < 0 ? AZ_PASS : c] = 1;
            trace(g, me, c < 0 ? AZ_PASS : c, pi, 0);
        }
        return c;
    }
    if (kind == DE_BUY && nopt > 1) {
        int f = rl_forced_buy(g, me, opts, n);
        if (f >= 0) return f;                         /* card school: not recorded, not traced */
    }
    if (nopt == 1) {                                  /* forced: nothing to learn */
        if (kind == DE_BUY && !g_in_plan) {
            float pi[AZ_NA] = {0};
            pi[n ? opts[0] : AZ_PASS] = 1;
            trace(g, me, n ? opts[0] : AZ_PASS, pi, 0);
        }
        return n ? opts[0] : -1;
    }
    float x[RL_NF], lg[RL_NA], mx = -1e30f, sum = 0, pr[RL_NA];
    rl_features(g, me, kind, subject, x);
    float v = rl_eval(s->prm, s->nh, s->nf, x, lg);
    int a = RL_PASS;
    for (int i = 0; i < RL_NA; i++) if (ok[i] && lg[i] > mx) { mx = lg[i]; a = i; }
    for (int i = 0; i < RL_NA; i++) { pr[i] = ok[i] ? expf((lg[i] - mx) / s->temp) : 0; sum += pr[i]; }
    if (s->sample) {
        double u = unif(&s->rng) * sum;
        for (int i = 0; i < RL_NA; i++) {
            if (!ok[i]) continue;
            a = i;
            if ((u -= pr[i]) < 0) break;
        }
    }
    s->decisions++;
    float val[RL_NA];
    int pa = rl_plan_hook(s, g, kind, me, ok, nopt, val);
    if (pa >= 0 && s->plan_play) a = pa;
    a = rl_chain_rule(g, me, kind, a, ok, lg);
    if (kind == DE_BUY && !g_in_plan) {               /* clip trace: the policy's probabilities */
        float pi[AZ_NA];
        for (int i = 0; i < AZ_NA; i++) pi[i] = pr[i] / sum;
        trace(g, me, a, pi, v);
    }
    if (s->record && !g_in_plan) {
        rl_record(g, me, kind, subject, ok, a, logf(pr[a] / sum), v);
        if (pa >= 0) rl_record_plan(pa, val);
    }
    return a == RL_PASS ? -1 : a;
}

static int g_plan_k[2], g_plan_play[2];
/* the planner for the next games: per seat K (0 = off) and whether it plays its choice (1) or only labels (0) */
void rl_set_plan(int k0, int k1, int play0, int play1) {
    g_plan_k[0] = k0; g_plan_k[1] = k1; g_plan_play[0] = play0; g_plan_play[1] = play1;
}

/* One game. Per seat i: mode[i], genome[i] (RL_GA; also the AZ agent's non-buy play), prm[i]
 * (RL_NET weights with nh[i] hidden units; or az-v1 net weights, nh 128, for RL_AZ), sample/record/temp (RL_NET),
 * az_sims (RL_AZ: 0 = its net's argmax, else IS-MCTS with this many simulations, leaf lam 0.5).
 * out: [winners mask, score0, score1, turns0, turns1, capped|error<<1]. Returns records written. */
int rl_play(const int *mode, const float *prm0, const float *prm1, const int *nh, const int *nf,
            const int *genome0, const int *genome1, const int *sample, const int *record,
            float temp, const int *az_sims, float az_lam, const int *kingdom, int kn,
            uint64_t seed, uint64_t rng_seed,
            float *X, unsigned char *LEGAL, int *A, float *LOGP, float *V, int *SEAT, int *TURN,
            int *KIND, int max_rec, int *out) {
    rl_rec rec = {X, LOGP, V, LEGAL, A, SEAT, TURN, KIND, max_rec, 0};
    r_rec = &rec;
    const float *prm[2] = {prm0, prm1};
    const int *genome[2] = {genome0, genome1};
    uint64_t r = rng_seed | 1;
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i] = (rl_seat){mode[i], genome[i], prm[i], nh[i], sample[i], record[i], temp > 0 ? temp : 1,
                          {0}, de_rng_next(&r) | 1, 0, nf[i], 0, 0};
        st[i].plan_k = g_plan_k[i];
        st[i].plan_play = g_plan_play[i];
        if (mode[i] == RL_AZ)
            st[i].az = (az_agent){az_sims[i] ? AZ_SEARCH : AZ_NET, {prm[i], 128}, genome[i], az_sims[i], 0, 0,
                                  0, 1.5f, az_lam, de_rng_next(&r) | 1, 0, 0};
    }
    de_agent agents[2] = {{rl_decide, &st[0]}, {rl_decide, &st[1]}};
    de_game g;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, agents);
    out[0] = de_winners(&g);
    out[1] = de_score(&g, 0);
    out[2] = de_score(&g, 1);
    out[3] = g.p[0].turns;
    out[4] = g.p[1].turns;
    out[5] = g.capped | (g.error << 1);
    r_rec = NULL;
    return rec.n;
}
