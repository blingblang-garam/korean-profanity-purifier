# coding: utf-8
"""[v2 최종] 실제 데이터로 학습 → 평가 → 분석 (여러 시드)

1차(real_data.py)와 달라진 점
  학습  · 학습 쌍 2배(실제 댓글 주입 24,000 + 템플릿 4,000 + 깨끗한 댓글 6,000), 언어 모델 말뭉치 30,000문장
        · ch06 LSTM 추가: 양방향 LSTM 탐지기(TimeBiLSTMMasked), LSTM 언어 모델(ch06 Rnnlm)
  순화  · 후보 채점: 문장 전체 퍼플렉서티 → 바꾼 자리 주변 국소 퍼플렉서티
        · 후보 제한: 탐지 구간과 같은 비속어 계열의 순화어만 + 조사 보존 (새끼들이 → 사람들이)
  평가  · 시드 3개 평균 ± 표준편차, 심판 분류기는 한 번만 학습해 모든 시스템·시드에 같은 것을 씀
        · Curse-detection 명시적 욕설 문장의 은(silver) 구간 라벨(정규식)로 글자 단위 재현율
        · E5(음절 vs 자모)를 실제 데이터로 재실험

실행:  cd project && python final_pipeline.py [--seeds 0 1 2] [--quick]
결과:  saved/final/ (로그, results.json, 시스템별 출력 CSV, 시드 0 모델 = 데모용)
"""
import sys
import os
import re
import csv
import json
import time
import random
import pickle
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from common.optimizer import SGD
from common.trainer import RnnlmTrainer

from jamo import normalize
from make_pairs import make_pairs
from loader import JamoVocab, SyllableVocab, detector_dataset, lm_corpus, load_curse_detection, load_kold
from models import RnnDetector, SentenceLM, LstmSentenceLM
from purifier import Purifier, dictionary_baseline
from lexicon import Lexicon
from evaluate import char_bleu, span_prf, evaluate, HANDWRITTEN
from train_demo import train_detector
from real_data import load_clean_comments, inject_real, PROFANE_RE, DATA, MAX_LEN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'saved', 'final')
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f, self.stdout = open(path, 'w', encoding='utf-8'), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()


def silver_labels(text):
    """정규식(PROFANE_RE)에 걸리는 글자 = 은 라벨. 명시적 욕설만 표시하므로 재현율 확인용."""
    lab = [0] * len(text)
    for m in PROFANE_RE.finditer(text):
        for i in range(m.start(), m.end()):
            lab[i] = 1
    return lab


class NgramJudge:
    """독립 심판 v2: 글자 1~3-gram 나이브 베이즈 (교과 외, RNN과 구조가 전혀 다른 모델).
    1차 심판(UnSmile로 학습한 양방향 RNN 문장 분류기)은 정확도 0.60~0.68이고, 명시적 욕설 200문장 중
    욕설로 판정한 비율이 실행에 따라 0.24~0.90으로 흔들려 STA·J를 믿기 어려웠다."""

    def __init__(self, texts, labels):
        from collections import Counter
        self.cnt, self.docs = [Counter(), Counter()], [0, 0]
        for t, l in zip(texts, labels):
            self.cnt[l].update(set(self._grams(t)))
            self.docs[l] += 1
        V = set(self.cnt[0]) | set(self.cnt[1])
        tot = [sum(c.values()) for c in self.cnt]
        self.w = {g: np.log((self.cnt[1][g] + 1) / (tot[1] + len(V))) - np.log((self.cnt[0][g] + 1) / (tot[0] + len(V)))
                  for g in V}
        self.prior = np.log(self.docs[1] / self.docs[0])

    @staticmethod
    def _grams(t):
        t = t.replace(' ', '_')
        return [t[i:i + n] for n in (1, 2, 3) for i in range(len(t) - n + 1)]

    def prob(self, texts):
        """욕설 확률 (로그 오즈 → 시그모이드)"""
        z = np.array([self.prior + sum(self.w.get(g, 0.0) for g in set(self._grams(t))) for t in texts])
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_judge_v2(test200):
    """Curse-detection('욕설 여부' 0/1)에서 평가 200문장을 빼고 90% 학습 · 10% 검증"""
    exclude = set(test200)
    cd = [(t, l) for t, l in load_curse_detection(os.path.join(DATA, 'Curse-detection-data', 'dataset.txt'))
          if t and t not in exclude]
    random.Random(5).shuffle(cd)
    n_val = len(cd) // 10
    val, tr = cd[:n_val], cd[n_val:]
    judge = NgramJudge([t for t, _ in tr], [l for _, l in tr])
    pred = judge.prob([t for t, _ in val]) >= 0.5
    gold = np.array([l for _, l in val])
    acc = float(np.mean(pred == gold))
    _, _, f1 = span_prf([pred.astype(int)], [gold.astype(int)])
    print(f'  심판 v2 글자 n-gram 나이브 베이즈 (Curse-detection 검증 {len(val)}문장): 정확도 {acc:.3f}, 욕설 F1 {f1:.3f} | 학습 {len(tr)}문장')
    return judge, acc


def train_detector_es(det, X, Y, epochs, seed, vocab, dev):
    """[v3] 조기 종료: 매 에포크 실제 dev(Curse-detection 절반)의 문장 F1을 재고, 가장 좋았던 에포크의 가중치로 되돌린다.
    학습 곡선(learning_curve.py)에서 합성 데이터 성능은 계속 오르는데 실제 성능은 일찍 멈추거나 떨어지는 것을 확인했다."""
    from learning_curve import sentence_scores, pr_at
    from common.util import clip_grads
    from loader import batches
    opt, rng = SGD(1.0), np.random.RandomState(seed)
    texts, gold = dev
    best, best_ep, best_params = -1.0, 0, None
    for ep in range(1, epochs + 1):
        for xs, ts, mask in batches(X, Y, 32, rng):
            det.forward(xs, ts, mask)
            det.backward()
            clip_grads(det.grads, 5.0)
            opt.update(det.params, det.grads)
        p, r = pr_at(sentence_scores(det, vocab, texts), gold, 0.5)
        f1 = 2 * p * r / (p + r + 1e-9)
        print(f'  에폭 {ep:2d} | dev 정밀도 {p:.3f} 재현율 {r:.3f} F1 {f1:.3f}', flush=True)
        if f1 > best:
            best, best_ep, best_params = f1, ep, [q.copy() for q in det.params]
    for q, b in zip(det.params, best_params):
        q[...] = b
    print(f'  → {best_ep}에폭 가중치 사용 (dev F1 {best:.3f})')
    return best_ep


def train_lm(cls, sents, vocab, epochs, seed):
    np.random.seed(seed)                 # 교재 모델은 전역 np.random으로 초기화
    lm = cls(len(vocab), 48, 128, bos_id=vocab.tok_to_id['<s>'])
    corpus = lm_corpus(sents, vocab)
    trainer = RnnlmTrainer(lm, SGD(0.5 if cls is SentenceLM else 20.0))
    trainer.fit(corpus[:-1], corpus[1:], max_epoch=epochs, batch_size=20, time_size=35,
                max_grad=0.25, eval_interval=2000)
    return lm


def detector_char_preds(purifier, texts):
    out = []
    for t in texts:
        probs = purifier.char_probs(t)
        out.append([int(p >= purifier.thr and ch != ' ') for p, ch in zip(probs, t)])
    return out


def run_seed(seed, data, vocab, judge, ref, args, t0):
    """시드 하나: 학습 데이터 생성 → 탐지기 3종 + 언어 모델 2종 학습 → 평가. 지표 dict와 순화 출력 반환."""
    clean, test200 = data['clean'], data['test200']
    rng = random.Random(seed)
    order = clean[:]
    rng.shuffle(order)
    held = data['held']                                       # 퍼플렉서티 측정용 (모든 시드 공통, 학습에서 제외)
    pool = [s for s in order if s not in data['held_set']]
    n_inj, n_neg, n_lm = (6000, 2000, 8000) if args.quick else (24000, 6000, 30000)
    lm_sents = pool[:n_lm]
    if args.data == 'v3':
        from data_v3 import inject_v3, trap_negative
        real_pairs = [p for p in (inject_v3(s, rng) for s in pool[:n_inj]) if p]
        traps = [trap_negative(s, rng) for s in pool[30000:32000]]
        lm_sents = lm_sents + data['extra_lm']
    else:
        real_pairs = [p for p in (inject_real(s, rng) for s in pool[:n_inj]) if p]
        traps = []
    tpl_pairs = make_pairs(4000, seed=seed + 1)
    neg = [(s, s, [0] * len(s)) for s in pool[n_inj:n_inj + n_neg]] + traps
    train_pairs = real_pairs + tpl_pairs + neg
    print(f'\n========== 시드 {seed} ==========')
    print(f'[학습 쌍] 실제 댓글 주입 {len(real_pairs)} + 템플릿 {len(tpl_pairs)} + 깨끗한 댓글 {len(neg)} | LM 말뭉치 {len(lm_sents)}문장')
    X, Y = detector_dataset(train_pairs, vocab)

    m = {}
    dets = {}
    sdir = os.path.join(OUT, 'quick' if args.quick else '', f'seed{seed}')
    os.makedirs(sdir, exist_ok=True)
    fname = lambda kind, name: os.path.join(sdir, f'{kind}_{name.replace(" ", "_")}.pkl')
    times = {}
    if args.eval_only:
        with open(os.path.join(sdir, 'times.json'), encoding='utf-8') as f:
            times = json.load(f)
    specs = [('단방향 RNN', False, 'rnn', 12), ('양방향 RNN', True, 'rnn', 12), ('양방향 LSTM', True, 'lstm', args.lstm_epochs)]
    for name, bi, cell, ep in specs:
        det = RnnDetector(len(vocab), 32, 64, bidirectional=bi, cell=cell, seed=seed)
        if args.eval_only:
            det.load(fname('det', name))
        else:
            ts = time.time()
            print(f'[탐지기 {name}] 학습 {ep}에폭 ({ts - t0:.0f}s)')
            if args.data == 'v3':
                times[f'선택 에포크/{name}'] = train_detector_es(det, X, Y, 12, seed, vocab, data['dev'])
            else:
                train_detector(det, X, Y, epochs=ep, lr=1.0, seed=seed)
            det.save(fname('det', name))
            times[f'학습시간/{name}'] = time.time() - ts
        dets[name] = det

    lms = {}
    for name, cls in [('RNN LM', SentenceLM), ('LSTM LM', LstmSentenceLM)]:
        if args.eval_only:
            lm = cls(len(vocab), 48, 128, bos_id=vocab.tok_to_id['<s>'])
            lm.load(fname('lm', name))
        else:
            ts = time.time()
            print(f'[언어 모델 {name}] 학습 ({ts - t0:.0f}s)')
            lm = train_lm(cls, lm_sents, vocab, args.lm_epochs, seed)
            lm.save(fname('lm', name))
            times[f'학습시간/{name}'] = time.time() - ts
        ppl = float(np.median(lm.sentence_perplexity([vocab.encode(s)[0] for s in held])))
        print(f'  [{name}] 처음 보는 깨끗한 댓글 {len(held)}개 퍼플렉서티(중앙값) {ppl:.2f}')
        m[f'LM 퍼플렉서티/{name}'] = ppl
        lms[name] = lm
    if not args.eval_only:
        with open(os.path.join(sdir, 'times.json'), 'w', encoding='utf-8') as f:
            json.dump(times, f)
    m.update(times)

    # ── ① Curse-detection 문장 단위 탐지 ──
    cur = data['curse_test']                 # dev 절반은 에포크 고르기(v3)에 쓰므로 탐지 평가는 test 절반으로만
    gold = np.array([l for _, l in cur])
    explicit = np.array([bool(PROFANE_RE.search(t)) for t, _ in cur])
    systems = {'E1 금칙어 사전': lambda t: dictionary_baseline(t) != t}
    for name, det in dets.items():
        p = Purifier(det, None, vocab)
        systems[name] = lambda t, p=p: bool(p.detect(t)[1])
    for name, fn in systems.items():
        pred = np.array([fn(t) for t, _ in cur])
        prec = (pred & (gold == 1)).sum() / max(pred.sum(), 1)
        rec = (pred & (gold == 1)).sum() / (gold == 1).sum()
        rec_exp = (pred & (gold == 1) & explicit).sum() / ((gold == 1) & explicit).sum()
        fpr = (pred & (gold == 0)).sum() / (gold == 0).sum()
        m[f'탐지/{name}/정밀도'], m[f'탐지/{name}/재현율'] = prec, rec
        m[f'탐지/{name}/F1'] = 2 * prec * rec / (prec + rec + 1e-9)
        m[f'탐지/{name}/명시적 재현율'], m[f'탐지/{name}/정상문장 오탐률'] = rec_exp, fpr

    # ── ①-b 은 라벨 글자 재현율 (명시적 욕설 문장) ──
    exp_texts = [t for t, l in cur if l == 1 and PROFANE_RE.search(t)]
    silver = [silver_labels(t) for t in exp_texts]
    for name, det in dets.items():
        _, r, _ = span_prf(detector_char_preds(Purifier(det, None, vocab), exp_texts), silver)
        m[f'은라벨 글자 재현율/{name}'] = r

    # ── ② KOLD ──
    for name, det in dets.items():
        p = Purifier(det, None, vocab)
        inside = total = 0
        for t, lab in data['kold_off']:
            for i, pr in enumerate(p.char_probs(t)):
                if pr >= 0.5 and t[i] != ' ':
                    total += 1
                    inside += lab[i] if i < len(lab) else 0
        m[f'KOLD/{name}/구간 안 비율'] = inside / max(total, 1)
        m[f'KOLD/{name}/비공격 오탐률'] = float(np.mean([bool(p.detect(t)[1]) for t, _ in data['kold_non']]))

    # ── ③ 실제 욕설 200문장 순화 ──
    bi, bl = dets['양방향 RNN'], dets['양방향 LSTM']
    rnn_lm, lstm_lm = lms['RNN LM'], lms['LSTM LM']
    pur = {
        'E1 금칙어 사전': dictionary_baseline,
        'E2 단방향+고정치환': Purifier(dets['단방향 RNN'], rnn_lm, vocab, mode='fixed').purify,
        'E3 양방향+고정치환': Purifier(bi, rnn_lm, vocab, mode='fixed').purify,
        'E4 양방향+RNNLM(문장)': Purifier(bi, rnn_lm, vocab).purify,
        'E4a +국소 채점': Purifier(bi, rnn_lm, vocab, score='local').purify,
        'E4b +계열 후보': Purifier(bi, rnn_lm, vocab, family=True).purify,
        'E4c +국소+계열': Purifier(bi, rnn_lm, vocab, score='local', family=True).purify,
        'E9 LSTM탐지+LSTM LM(국소+계열)': Purifier(bl, lstm_lm, vocab, score='local', family=True).purify,
        'E10 E9+밈·비속어 사전 검색': Purifier(bl, lstm_lm, vocab, score='local', family=True, lexicon=Lexicon()).purify,
    }
    if args.data == 'v3' and os.path.exists(os.path.join(OUT, 'usage_lstm.pkl')):
        from usage_model import UsageTagger
        pur['E11 E10+용법·강도'] = Purifier(bl, lstm_lm, vocab, score='local', family=True, lexicon=Lexicon(),
                                         usage=UsageTagger(vocab, os.path.join(OUT, 'usage_lstm.pkl'))).purify
    jp = judge.prob
    outs_all = {}
    # 순화: 개발 중에 출력을 보며 고친 200문장 / 순화B: test 절반에서 새로 뽑아 개발 중 한 번도 안 본 200문장
    for P, texts in [('순화', test200), ('순화B', data['test200b'])]:
      m[f'{P}/입력 심판 욕설 판정률'] = float(np.mean(jp(texts) >= 0.5))
      for name, fn in pur.items():
        ts = time.time()
        outs = [fn(t) if name.startswith('E1 ') else fn(t)['output'] for t in texts]
        sec = (time.time() - ts) / len(texts)
        tox = jp(outs)
        sta = (tox < 0.5).astype(float)
        sim = np.array([char_bleu(o, s) for o, s in zip(outs, texts)])
        ppl = ref['lm'].sentence_perplexity([vocab.encode(o)[0] if o else np.array([2], 'i') for o in outs])
        fl = np.minimum(1.0, ref['ppl'] / ppl)
        j = sta * sim * fl
        m[f'{P}/{name}/STA'], m[f'{P}/{name}/SIM'], m[f'{P}/{name}/FL'] = sta.mean(), sim.mean(), fl.mean()
        m[f'{P}/{name}/J'] = j.mean()
        lex_ok = np.array([0.0 if PROFANE_RE.search(o) else 1.0 for o in outs])
        m[f'{P}/{name}/J_lex'] = float((lex_ok * sim * fl).mean())   # 심판 대신 비속어 패턴 잔존으로 STA를 본 J
        m[f'{P}/{name}/패턴 잔존'] = float(np.mean([bool(PROFANE_RE.search(o)) for o in outs]))
        m[f'{P}/{name}/변경 없음'] = float(np.mean([o == normalize(s) for o, s in zip(outs, texts)]))
        m[f'{P}/{name}/ms'] = sec * 1000
        outs_all[name if P == '순화' else name + ' [B]'] = outs

    # ── ④ 손으로 쓴 25문장 (정답 순화문 있음, 과교정 확인용: 개나리·새끼 고양이·시발역 등 함정 5문장) ──
    for name, fn in pur.items():
        f = fn if name.startswith('E1 ') else (lambda t, fn=fn: fn(t)['output'])
        r = evaluate(f, HANDWRITTEN, lm=ref['lm'], vocab=vocab)
        for k in ('STA', 'SIM', 'FL', 'J', 'EM', '과교정'):
            m[f'손글씨/{name}/{k}'] = r[k]

    # ── E5 실제 데이터: 음절 vs 자모 (양방향 RNN, 같은 학습 쌍) ──
    if not args.skip_e5:
        sv = SyllableVocab([p[0] for p in train_pairs] + lm_sents)
        Xs, Ys = detector_dataset(train_pairs, sv)
        det_s = RnnDetector(len(sv), 32, 64, bidirectional=True, seed=seed)
        if args.eval_only:
            det_s.load(fname('det', '음절 양방향 RNN'))
        else:
            print(f'[E5 음절 탐지기] 학습 (어휘 {len(sv)}) ({time.time() - t0:.0f}s)')
            if args.data == 'v3':
                m['선택 에포크/음절'] = train_detector_es(det_s, Xs, Ys, 12, seed, sv, data['dev'])
            else:
                train_detector(det_s, Xs, Ys, epochs=12, lr=1.0, seed=seed)
            det_s.save(fname('det', '음절 양방향 RNN'))
        seen = set(''.join(p[0] for p in train_pairs))
        for name, det, voc in [('음절', det_s, sv), ('자모', bi, vocab)]:
            p = Purifier(det, None, voc)
            pred = np.array([bool(p.detect(t)[1]) for t, _ in cur])
            rec_exp = (pred & (gold == 1) & explicit).sum() / ((gold == 1) & explicit).sum()
            prec = (pred & (gold == 1)).sum() / max(pred.sum(), 1)
            # 학습 데이터에 한 번도 안 나온 글자를 포함한 명시적 욕설 문장 = 처음 보는 변형
            novel = [t for t in exp_texts if any(ch not in seen for ch in t)]
            _, r_nov, _ = span_prf(detector_char_preds(p, novel), [silver_labels(t) for t in novel])
            m[f'E5/{name}/명시적 재현율'], m[f'E5/{name}/정밀도'] = rec_exp, prec
            m[f'E5/{name}/처음 보는 글자 문장 글자 재현율'] = r_nov
        m['E5/처음 보는 글자 문장 수'] = len(novel)
        m['E5/음절 어휘'] = len(sv)

    return m, outs_all, dets, lms


def summarize(results):
    keys = list(results[0].keys())
    summary = {}
    for k in keys:
        v = np.array([float(r[k]) for r in results], float)
        summary[k] = (float(v.mean()), float(v.std()))
    return summary


def print_summary(s, seeds):
    def row(k, fmt='{:.3f}'):
        mu, sd = s[k]
        return (fmt + ' ± ' + fmt).format(mu, sd)
    print(f'\n\n################ 최종 요약 (시드 {seeds} 평균 ± 표준편차) ################')
    print('\n[탐지] Curse-detection 문장 단위')
    for name in ['E1 금칙어 사전', '단방향 RNN', '양방향 RNN', '양방향 LSTM']:
        print(f'  {name:12} 정밀도 {row(f"탐지/{name}/정밀도")}  재현율 {row(f"탐지/{name}/재현율")}  '
              f'명시적 재현율 {row(f"탐지/{name}/명시적 재현율")}  정상 오탐 {row(f"탐지/{name}/정상문장 오탐률")}')
    print('\n[탐지] 은 라벨 글자 재현율 · KOLD')
    for name in ['단방향 RNN', '양방향 RNN', '양방향 LSTM']:
        print(f'  {name:12} 은라벨 재현율 {row(f"은라벨 글자 재현율/{name}")}  KOLD 구간 안 {row(f"KOLD/{name}/구간 안 비율")}  '
              f'KOLD 비공격 오탐 {row(f"KOLD/{name}/비공격 오탐률")}')
    print('\n[언어 모델] 처음 보는 깨끗한 댓글 퍼플렉서티(중앙값) · 학습 시간(초)')
    for name in ['RNN LM', 'LSTM LM']:
        print(f'  {name:8} {row(f"LM 퍼플렉서티/{name}", "{:.2f}")}   {row(f"학습시간/{name}", "{:.0f}")}s')
    print('\n[순화] 실제 욕설 200문장')
    names = sorted({k.split('/')[1] for k in s if k.startswith('순화/') and k.count('/') == 2}, key=lambda x: (int(re.match(r'E(\d+)', x).group(1)), x))
    print(f'  (입력 200문장 심판 욕설 판정률 {row("순화/입력 심판 욕설 판정률")})')
    for name in names:
        print(f'  {name:30} STA {row(f"순화/{name}/STA")}  SIM {row(f"순화/{name}/SIM")}  FL {row(f"순화/{name}/FL")}  '
              f'J {row(f"순화/{name}/J")}  J_lex {row(f"순화/{name}/J_lex")}  잔존 {row(f"순화/{name}/패턴 잔존")}  '
              f'변경없음 {row(f"순화/{name}/변경 없음")}  {s[f"순화/{name}/ms"][0]:.1f}ms')
    print(f'\n[순화B] 개발 중 한 번도 안 본 실제 욕설 200문장 (test 절반에서 추출) — 입력 심판 욕설 판정률 {row("순화B/입력 심판 욕설 판정률")}')
    for name in names:
        print(f'  {name:30} STA {row(f"순화B/{name}/STA")}  SIM {row(f"순화B/{name}/SIM")}  FL {row(f"순화B/{name}/FL")}  '
              f'J {row(f"순화B/{name}/J")}  J_lex {row(f"순화B/{name}/J_lex")}  잔존 {row(f"순화B/{name}/패턴 잔존")}  '
              f'변경없음 {row(f"순화B/{name}/변경 없음")}')
    print('\n[순화] 손으로 쓴 25문장 (정답 있음)')
    for name in names:
        print(f'  {name:30} J {row(f"손글씨/{name}/J")}  정답 일치 {row(f"손글씨/{name}/EM")}  과교정 {row(f"손글씨/{name}/과교정")}')
    if 'E5/음절 어휘' in s:
        print(f'\n[E5 실제 데이터] 음절 vs 자모 (음절 어휘 {s["E5/음절 어휘"][0]:.0f}, 처음 보는 글자 포함 문장 {s["E5/처음 보는 글자 문장 수"][0]:.0f}개)')
        for name in ['음절', '자모']:
            print(f'  {name}  명시적 재현율 {row(f"E5/{name}/명시적 재현율")}  정밀도 {row(f"E5/{name}/정밀도")}  '
                  f'처음 보는 글자 문장 글자 재현율 {row(f"E5/{name}/처음 보는 글자 문장 글자 재현율")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--quick', action='store_true', help='작은 데이터로 동작 확인')
    ap.add_argument('--lstm-epochs', type=int, default=8)
    ap.add_argument('--lm-epochs', type=int, default=4)
    ap.add_argument('--skip-e5', action='store_true')
    ap.add_argument('--data', choices=['v2', 'v3'], default='v2', help='v3 = data_v3.py 학습 데이터 + 조기 종료 + LM 말뭉치 확대')
    ap.add_argument('--eval-only', action='store_true', help='저장된 시드별 모델로 평가만 다시')
    args = ap.parse_args()
    global OUT
    if args.data == 'v3':
        OUT = os.path.join(HERE, 'saved', 'final_v3')
        os.makedirs(OUT, exist_ok=True)
    tag = 's' + '_'.join(map(str, args.seeds))           # 시드를 나눠 병렬 실행해도 로그·결과가 안 겹치게
    prefix = ('quick_' if args.quick else 'final_') + ('eval_' if args.eval_only else '')
    sys.stdout = Tee(os.path.join(OUT, f'{prefix}log_{tag}.txt'))
    t0 = time.time()

    # ── 데이터 (모든 시드 공통) ──
    clean = load_clean_comments()
    rs = random.Random(1234)
    shuffled = clean[:]
    rs.shuffle(shuffled)
    held = shuffled[:500]
    curse = [(t, l) for t, l in load_curse_detection(os.path.join(DATA, 'Curse-detection-data', 'dataset.txt')) if t]
    kold = [(normalize(t), lab) for t, lab in load_kold(os.path.join(DATA, 'KOLD', 'data', 'kold_v1.json'))]
    pool = [t for t, l in curse if l == 1 and PROFANE_RE.search(t) and 8 <= len(t) <= MAX_LEN]
    test200 = random.Random(42).sample(pool, 200)              # 1차와 같은 200문장
    data = dict(clean=clean, held=held, held_set=set(held), curse=curse, test200=test200,
                curse80=[(t, l) for t, l in curse if len(t) <= 80],
                kold_off=[(t, l) for t, l in kold if any(l) and len(t) <= 80][:1000],
                kold_non=[(t, l) for t, l in kold if not any(l) and len(t) <= 80][:1000])
    # Curse-detection(평가 200문장 제외, 80자 이하)을 learning_curve.py와 같은 방식으로 dev / test 절반으로 나눈다
    cur = [(t, l) for t, l in curse if len(t) <= 80 and t not in set(test200)]
    random.Random(7).shuffle(cur)
    dev = cur[:len(cur) // 2]
    data['dev'] = ([t for t, _ in dev], np.array([l for _, l in dev]))
    data['curse_test'] = cur[len(cur) // 2:]
    pool_b = [t for t, l in data['curse_test'] if l == 1 and PROFANE_RE.search(t) and 8 <= len(t) <= MAX_LEN]
    data['test200b'] = random.Random(43).sample(pool_b, min(200, len(pool_b)))
    print(f'[평가 B] test 절반의 명시적 욕설 문장 {len(pool_b)}개 중 {len(data["test200b"])}개')
    print(f'[데이터] 깨끗한 댓글 {len(clean)}개 | Curse-detection {len(curse)}문장 | KOLD {len(kold)}문장 | 평가 200문장')

    vocab = JamoVocab(clean + [p[0] for p in make_pairs(4000, seed=1)])
    print(f'  자모 어휘 {len(vocab)}개')

    # ── 심판 + 유창성 기준 LM: 한 번만 학습해 모든 시드·시스템에 공통으로 사용 ──
    print(f'\n[심판] 학습 ({time.time() - t0:.0f}s)')
    judge, judge_acc = train_judge_v2(test200 + data['test200b'])
    ref_dir = os.path.join(OUT, 'quick' if args.quick else '')
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, f'ref_lm_{tag}.pkl')
    if args.eval_only:                                     # 기준 LM은 시드 999 고정이라 어느 실행에서 저장했든 같다
        import glob
        ref_lm = SentenceLM(len(vocab), 48, 128, bos_id=vocab.tok_to_id['<s>'])
        ref_lm.load(ref_path if os.path.exists(ref_path) else sorted(glob.glob(os.path.join(ref_dir, 'ref_lm_*.pkl')))[0])
    else:
        ref_lm = train_lm(SentenceLM, shuffled[500:][:8000 if args.quick else 30000], vocab, args.lm_epochs, 999)
        ref_lm.save(ref_path)
    ref_ppl = float(np.median(ref_lm.sentence_perplexity([vocab.encode(s)[0] for s in held])))
    print(f'  유창성(FL) 기준 LM: 처음 보는 깨끗한 댓글 퍼플렉서티 중앙값 {ref_ppl:.2f} (시스템 안의 LM과 별도 시드)')
    ref = {'lm': ref_lm, 'ppl': ref_ppl}
    data['extra_lm'] = []
    if args.data == 'v3':
        from data_v3 import load_unlabeled_clean
        # LM 말뭉치 거르기는 v3 학습 때와 같은 심판(평가 200문장만 뺀 것)으로 해야 같은 말뭉치·음절 어휘가 나온다
        judge_lm = judge if 'test200b' not in data else train_judge_v2(test200)[0]
        data['extra_lm'], n_all = load_unlabeled_clean(judge_lm, 60000, exclude=held)
        print(f'[v3 LM 말뭉치] HateSpeech 비라벨 댓글 문장 {n_all}개 중 정규식·심판 통과 {len(data["extra_lm"])}개 추가')

    results, outs_by_seed = [], {}
    for seed in args.seeds:
        m, outs, dets, lms = run_seed(seed, data, vocab, judge, ref, args, t0)
        results.append(m)
        outs_by_seed[seed] = outs
        with open(os.path.join(OUT, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab.id_to_tok, f)
        with open(os.path.join(OUT, f'outputs_seed{seed}.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            ka = [k for k in outs if not k.endswith(' [B]')]
            w.writerow(['원문'] + ka)
            for i, t in enumerate(test200):
                w.writerow([t] + [outs[k][i] for k in ka])
        with open(os.path.join(OUT, f'outputs_seed{seed}_B.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            kb = [k for k in outs if k.endswith(' [B]')]
            w.writerow(['원문'] + kb)
            for i, t in enumerate(data['test200b']):
                w.writerow([t] + [outs[k][i] for k in kb])
        print(f'  시드 {seed} 끝 ({time.time() - t0:.0f}s)')

    s = summarize(results)
    print_summary(s, args.seeds)
    with open(os.path.join(OUT, f'{prefix}results_{tag}.json'), 'w', encoding='utf-8') as f:
        per_seed = [{k: float(v) for k, v in r.items()} for r in results]
        json.dump({'seeds': args.seeds, 'judge_acc': judge_acc, 'ref_ppl': ref_ppl, 'per_seed': per_seed, 'summary': s}, f, ensure_ascii=False, indent=1)
    print(f'\n총 소요 {time.time() - t0:.0f}초 → {OUT}')


if __name__ == '__main__':
    main()
