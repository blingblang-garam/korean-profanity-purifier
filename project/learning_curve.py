# coding: utf-8
"""학습 곡선: 에포크마다 탐지기의 정밀도·재현율을 실제 데이터로 잰다 (오버피팅 확인)

- 학습 데이터는 final_pipeline.py와 같다 (같은 시드 → 같은 학습 쌍)
- 매 에포크 끝에 세 가지를 잰다
    합성 검증  : 학습에 안 쓴 깨끗한 댓글 2,000개에 같은 방식으로 욕설을 끼워 넣은 문장의 글자 단위 F1
    실제 개발셋 : Curse-detection을 반으로 나눈 dev 절반의 문장 단위 정밀도·재현율 (평가 200문장 제외)
                 → 에포크 고르기는 dev로만 하고, 나머지 test 절반은 최종 보고용으로 남겨 둔다
    학습 손실
- 문장 점수 = 공백이 아닌 글자 중 최대 비속어 확률 → 임계값을 바꿔 가며 정밀도-재현율 곡선도 그린다

실행:  cd project && python learning_curve.py --seed 0 [--epochs 16]
결과:  saved/final/curve/curve_seed{n}.json → plot_learning_curve.py
"""
import sys
import os
import json
import time
import random
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from common.optimizer import SGD
from common.util import clip_grads

from jamo import normalize
from make_pairs import make_pairs
from loader import JamoVocab, detector_dataset, batches, pad_batch, load_curse_detection
from models import RnnDetector
from evaluate import span_prf
from real_data import load_clean_comments, inject_real, PROFANE_RE, DATA, MAX_LEN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'saved', 'final', 'curve')
os.makedirs(OUT, exist_ok=True)
THRS = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]


def char_probs_batch(det, vocab, texts, bs=256):
    """문장 리스트 → 글자별 비속어 확률 리스트 (Purifier.char_probs를 배치로)"""
    enc = [vocab.encode(t) for t in texts]
    out = []
    for s in range(0, len(texts), bs):
        chunk = enc[s:s + bs]
        ids = [e[0] if len(e[0]) else np.array([1], 'i') for e in chunk]
        preds = det.predict(ids)
        for (_, cidx), p, t in zip(chunk, preds, texts[s:s + bs]):
            cp = [0.0] * len(t)
            for prob, c in zip(p, cidx):
                cp[c] = max(cp[c], float(prob))
            out.append(cp)
    return out


def sentence_scores(det, vocab, texts):
    return np.array([max([p for p, ch in zip(cp, t) if ch != ' '] or [0.0])
                     for cp, t in zip(char_probs_batch(det, vocab, texts), texts)])


def pr_at(scores, gold, thr):
    pred = scores >= thr
    tp = int((pred & (gold == 1)).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int((gold == 1).sum()), 1)
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=16)
    args = ap.parse_args()
    seed, t0 = args.seed, time.time()

    # ── final_pipeline.py와 같은 데이터 ──
    clean = load_clean_comments()
    shuffled = clean[:]
    random.Random(1234).shuffle(shuffled)
    held_set = set(shuffled[:500])
    curse = [(t, l) for t, l in load_curse_detection(os.path.join(DATA, 'Curse-detection-data', 'dataset.txt')) if t]
    pool200 = [t for t, l in curse if l == 1 and PROFANE_RE.search(t) and 8 <= len(t) <= MAX_LEN]
    test200 = set(random.Random(42).sample(pool200, 200))
    vocab = JamoVocab(clean + [p[0] for p in make_pairs(4000, seed=1)])

    rng = random.Random(seed)
    order = clean[:]
    rng.shuffle(order)
    pool = [s for s in order if s not in held_set]
    real_pairs = [p for p in (inject_real(s, rng) for s in pool[:24000]) if p]
    train_pairs = real_pairs + make_pairs(4000, seed=seed + 1) + [(s, s, [0] * len(s)) for s in pool[24000:30000]]
    X, Y = detector_dataset(train_pairs, vocab)

    # 합성 검증: 학습에 안 쓴 댓글(30000번 이후)에 욕설 주입 + 같은 수의 깨끗한 문장
    vr = random.Random(100 + seed)
    syn = [p for p in (inject_real(s, vr) for s in pool[32000:33500]) if p] + [(s, s, [0] * len(s)) for s in pool[33500:34000]]
    syn_texts, syn_gold = [p[0] for p in syn], [p[2] for p in syn]

    # 실제 개발셋: Curse-detection (평가 200문장 제외, 80자 이하)을 반으로 나눔 — 모든 시드 공통
    cur = [(t, l) for t, l in curse if len(t) <= 80 and t not in test200]
    random.Random(7).shuffle(cur)
    dev = cur[:len(cur) // 2]
    dev_texts = [t for t, _ in dev]
    dev_gold = np.array([l for _, l in dev])
    dev_explicit = np.array([bool(PROFANE_RE.search(t)) for t in dev_texts])
    print(f'[시드 {seed}] 학습 쌍 {len(train_pairs)} | 합성 검증 {len(syn)} | 실제 dev {len(dev)}문장 '
          f'(욕설 {int(dev_gold.sum())}, 명시적 욕설 {int((dev_explicit & (dev_gold == 1)).sum())})')

    # 사전 기준선 (에포크와 무관)
    from purifier import dictionary_baseline
    dpred = np.array([dictionary_baseline(t) != t for t in dev_texts])
    tp = int((dpred & (dev_gold == 1)).sum())
    result = {'seed': seed, 'dict': {'precision': tp / max(int(dpred.sum()), 1), 'recall': tp / int(dev_gold.sum())},
              'detectors': {}}

    specs = [('단방향 RNN', False, 'rnn'), ('양방향 RNN', True, 'rnn'), ('양방향 LSTM', True, 'lstm')]
    for name, bi, cell in specs:
        det = RnnDetector(len(vocab), 32, 64, bidirectional=bi, cell=cell, seed=seed)
        opt, brng = SGD(1.0), np.random.RandomState(seed)       # final_pipeline(train_detector)과 같은 설정
        hist = []
        for ep in range(1, args.epochs + 1):
            tot = cnt = 0
            for xs, ts, mask in batches(X, Y, 32, brng):
                loss = det.forward(xs, ts, mask)
                det.backward()
                clip_grads(det.grads, 5.0)
                opt.update(det.params, det.grads)
                tot, cnt = tot + loss, cnt + 1
            # 합성 검증 글자 F1
            sp = char_probs_batch(det, vocab, syn_texts)
            syn_pred = [[int(p >= 0.5 and ch != ' ') for p, ch in zip(cp, t)] for cp, t in zip(sp, syn_texts)]
            _, _, syn_f1 = span_prf(syn_pred, syn_gold)
            # 실제 dev: 임계값별 정밀도·재현율
            sc = sentence_scores(det, vocab, dev_texts)
            pr = {str(th): pr_at(sc, dev_gold, th) for th in THRS}
            prec, rec = pr_at(sc, dev_gold, 0.5)
            pred = sc >= 0.5
            rec_exp = int((pred & (dev_gold == 1) & dev_explicit).sum()) / int(((dev_gold == 1) & dev_explicit).sum())
            fpr = int((pred & (dev_gold == 0)).sum()) / int((dev_gold == 0).sum())
            row = {'epoch': ep, 'loss': tot / cnt, 'syn_f1': syn_f1, 'precision': prec, 'recall': rec,
                   'f1': 2 * prec * rec / (prec + rec + 1e-9), 'explicit_recall': rec_exp, 'fpr': fpr, 'pr': pr}
            hist.append(row)
            print(f'  {name} 에폭 {ep:2d} | 손실 {row["loss"]:.4f} | 합성 F1 {syn_f1:.3f} | '
                  f'실제 정밀도 {prec:.3f} 재현율 {rec:.3f} (명시적 {rec_exp:.3f}) | {time.time() - t0:.0f}s', flush=True)
        result['detectors'][name] = hist
        with open(os.path.join(OUT, f'curve_seed{seed}.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
    print(f'끝 ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
