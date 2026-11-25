# coding: utf-8
"""[v4] 욕설의 '용법' 분류기 — 특성을 살리는 순화의 첫 단계 (Ch06 LSTM)

같은 '씨발'이라도 문장에서 하는 일이 다르다.
  강조  : 오늘 씨발 덥네 / 존나 맛있다 / 개빡친다      → 정말 · 너무 · 완전 · 엄청
  감탄  : 씨발 또 버그네 / 아 씨발 / ㅅㅂ             → 아 · 아오 · 하 · 아 진짜
  지칭  : 저 새끼 왜 저래 / 남친새끼가 / 이 새끼야     → 사람 · 너 · 얘 (대상은 남긴다)
  모욕  : 병신같이 운전하네 / 미친 기자 / 좆같다        → 바보같이 · 이상한 · 별로다 (평가·서술)

학습 데이터는 탐지기와 같은 '역방향 생성' 방식: 깨끗한 실제 댓글에 용법을 정해 놓고 욕설을 끼워 넣으면
(문장, 욕설 위치, 용법) 라벨이 공짜로 생긴다.

모델: 자모 Embedding → ch06 TimeLSTM 양방향(TimeBiLSTMMasked) → 욕설 구간의 은닉 상태 평균 → Affine → Softmax(4)

실행:  cd project && python usage_model.py      → saved/final_v3/usage_lstm.pkl, 학습·평가 로그
"""
import sys
import os
import re
import json
import time
import random
import pickle
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from common.time_layers import TimeEmbedding, TimeAffine
from common.functions import softmax
from common.optimizer import Adam
from common.util import clip_grads

from jamo import normalize
from make_pairs import mutate
from loader import JamoVocab, pad_batch
from layers_ext import TimeBiLSTMMasked

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'saved', 'final_v3')
LABELS = ['강조', '감탄', '지칭', '모욕']

# ── 용법별 욕설 표현 (학습 데이터 생성용) ──
EMPH = ['존나', '졸라', '좆나', '씨발', '개', '존1나', 'ㅈㄴ', '겁나 씨발', '씹']           # 형용사·동사 앞
EXCL = ['씨발', '시발', 'ㅅㅂ', 'ㅆㅂ', '아 씨발', '젠장', '썅', '아 시발', '씨바', '염병']   # 문장 앞·끝 감탄
REF_ATTACH = ['새끼', '새끼가', '새끼들', '새끼들이', '새끼는', '놈', '놈이', '놈들', '년', '년이', '충', '새기']
REF_STAND = ['저 새끼', '그 새끼', '이 새끼', '저 놈', '그 년', '저 새끼들', '이 새끼야', '야 이 새끼야', 'ㅅㄲ',
             '저 ㅅㄲ', '새끼들아', '이 병신아', '병신들아', '이 미친놈아']
INSULT_ADJ = ['미친', '병신같은', '좆같은', '개같은', '또라이같은', '븅신같은']             # 명사 앞 수식
INSULT_ADV = ['병신같이', '등신같이', '좆같이', '븅신같이']                               # 동사 앞
INSULT_PRED = ['좆같다', '좆같네', '빡친다', '빡치네', '지랄하네', '미쳤네', '병신이네', '개좆같다', '지랄이다',
               '좆같아', '빡쳐', '염병하네']                                                # 문장 끝 서술


def _place(words, k, piece):
    """words의 k번째 위치에 piece를 끼워 넣은 문장과 piece의 글자 구간"""
    left = ' '.join(words[:k])
    s = len(left) + (1 if left else 0)
    sent = ' '.join(words[:k] + [piece] + words[k:])
    return sent, (s, s + len(piece))


def make_example(sent, label, rng):
    """깨끗한 댓글 + 용법 → (오염 문장, (시작, 끝), 라벨) 또는 None"""
    words = sent.split(' ')
    hang = [k for k, w in enumerate(words) if re.fullmatch('[가-힣]{2,}', w)]
    if label == '강조':
        if not hang:
            return None
        k = rng.choice(hang)
        bad = mutate(rng.choice(EMPH), rng, 0.4)
        if bad == '개' and rng.random() < 0.7:                    # 접두 '개'는 붙여 씀 (개빡친다)
            s = len(' '.join(words[:k])) + (1 if k else 0)
            return ' '.join(words[:k] + ['개' + words[k]] + words[k + 1:]), (s, s + 1), label
        return _place(words, k, bad) + (label,)
    if label == '감탄':
        bad = mutate(rng.choice(EXCL), rng, 0.4)
        k = 0 if rng.random() < 0.7 else len(words)
        return _place(words, k, bad) + (label,)
    if label == '지칭':
        if hang and rng.random() < 0.5:                             # 앞말에 붙이기 (남친새끼가)
            k = rng.choice(hang)
            w = words[k]
            stem = w[:-1] if len(w) >= 3 and w[-1] in '이가은는을를도' else w
            bad = mutate(rng.choice(REF_ATTACH), rng, 0.3)
            s = len(' '.join(words[:k])) + (1 if k else 0) + len(stem)
            return ' '.join(words[:k] + [stem + bad] + words[k + 1:]), (s, s + len(bad)), label
        bad = rng.choice(REF_STAND)
        k = 0 if rng.random() < 0.5 else len(words)
        sent2, (s, e) = _place(words, k, bad)
        core = re.sub(r'^(저|그|이|야 이) ', '', bad)              # '저 새끼'에서 욕설은 '새끼'만
        s = s + (len(bad) - len(core))
        return sent2, (s, e), label
    if label == '모욕':
        r = rng.random()
        if r < 0.35 and hang:
            k = rng.choice(hang)
            return _place(words, k, mutate(rng.choice(INSULT_ADJ), rng, 0.3)) + (label,)
        if r < 0.6 and len(words) >= 2:
            k = rng.randint(max(1, len(words) - 2), len(words) - 1)
            return _place(words, k, rng.choice(INSULT_ADV)) + (label,)
        return _place(words, len(words), mutate(rng.choice(INSULT_PRED), rng, 0.3)) + (label,)


# 손으로 쓴 평가 문장 (실제 댓글 말투, 생성 규칙과 별개로 작성) — (문장, 욕설 표기, 용법)
HANDWRITTEN_USAGE = [
    ('오늘 존나 덥네', '존나', '강조'), ('이거 개맛있다 ㅋㅋ', '개', '강조'), ('시험 씨발 너무 어렵다', '씨발', '강조'),
    ('과제 ㅈㄴ 많아', 'ㅈㄴ', '강조'), ('진짜 졸라 웃기네', '졸라', '강조'), ('배송 존1나 느리네', '존1나', '강조'),
    ('팀플 개빡친다', '개', '강조'), ('줄이 씹 길다', '씹', '강조'),
    ('씨발 또 버그네', '씨발', '감탄'), ('아 씨발 지각이다', '씨발', '감탄'), ('ㅅㅂ 폰 떨어뜨림', 'ㅅㅂ', '감탄'),
    ('젠장 비 온다', '젠장', '감탄'), ('시발 버스 놓쳤네', '시발', '감탄'), ('왜 또 안 되냐 씨발', '씨발', '감탄'),
    ('ㅆㅂ 컴파일 에러', 'ㅆㅂ', '감탄'), ('아 썅 졌다', '썅', '감탄'),
    ('저 새끼 왜 저래', '새끼', '지칭'), ('남친새끼가 연락을 안 해', '새끼', '지칭'), ('야 이 새끼야 빨리 와', '새끼', '지칭'),
    ('그 놈 말 많다', '놈', '지칭'), ('옆집 새끼들 시끄럽다', '새끼들', '지칭'), ('기자놈들 또 이러네', '놈들', '지칭'),
    ('저 ㅅㄲ 뭐냐', 'ㅅㄲ', '지칭'), ('상사새끼가 또 야근 시킴', '새끼', '지칭'),
    ('병신같이 운전하네', '병신같이', '모욕'), ('미친 기사 또 나왔네', '미친', '모욕'), ('서비스 좆같네', '좆같네', '모욕'),
    ('이번 패치 개좆같다', '개좆같다', '모욕'), ('또 지랄하네', '지랄하네', '모욕'), ('좆같은 회사 그만두고 싶다', '좆같은', '모욕'),
    ('등신같이 굴지 마', '등신같이', '모욕'), ('일 진짜 빡친다', '빡친다', '모욕'),
]


# 같은 단어가 위치(문맥)로만 용법이 갈리는 문장 — 단어만 보는 방식으로는 못 맞힌다
CONTEXT_TEST = [('오늘 씨발 덥네', '씨발', '강조'), ('씨발 오늘 덥네', '씨발', '감탄'), ('시험 시발 어렵다', '시발', '강조'),
                ('시발 시험 망했다', '시발', '감탄'), ('밥이 ㅅㅂ 맛없네', 'ㅅㅂ', '강조'), ('ㅅㅂ 밥 맛없네', 'ㅅㅂ', '감탄'),
                ('이 노래 씨발 좋다', '씨발', '강조'), ('늦었다 씨발', '씨발', '감탄'), ('사람 씨발 많네', '씨발', '강조'),
                ('아 씨발 또 졌어', '씨발', '감탄'), ('줄이 ㅆㅂ 길다', 'ㅆㅂ', '강조'), ('ㅆㅂ 줄 길다', 'ㅆㅂ', '감탄')]


class UsageClassifier:
    """자모 Embedding → 양방향 LSTM(ch06 TimeLSTM ×2) → 욕설 구간 평균 → Affine → Softmax"""

    def __init__(self, vocab_size, wordvec_size=32, hidden_size=48, n_class=4, seed=0):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.RandomState(seed).randn
        lstm = lambda: [(rn(D, 4 * H) / np.sqrt(D)).astype('f'), (rn(H, 4 * H) / np.sqrt(H)).astype('f'), np.zeros(4 * H, 'f')]
        self.embed = TimeEmbedding((rn(V, D) / 100).astype('f'))
        self.rnn = TimeBiLSTMMasked(*lstm(), *lstm())
        self.W = (rn(2 * H, n_class) / np.sqrt(2 * H)).astype('f')
        self.b = np.zeros(n_class, 'f')
        self.params = self.embed.params + self.rnn.params + [self.W, self.b]
        self.grads = self.embed.grads + self.rnn.grads + [np.zeros_like(self.W), np.zeros_like(self.b)]

    def _score(self, xs, mask, span_mask):
        self.rnn.set_mask(mask)
        hs = self.rnn.forward(self.embed.forward(xs))                  # (N, T, 2H)
        cnt = np.maximum(span_mask.sum(axis=1, keepdims=True), 1)
        pooled = (hs * span_mask[:, :, None]).sum(axis=1) / cnt         # 욕설 구간 평균
        self.cache = (hs.shape, span_mask, cnt, pooled)
        return pooled @ self.W + self.b

    def forward(self, xs, mask, span_mask, ts):
        y = softmax(self._score(xs, mask, span_mask))
        self.y_t = (y, ts)
        return float(-np.mean(np.log(y[np.arange(len(ts)), ts] + 1e-7)))

    def backward(self):
        (N, T, H2), span_mask, cnt, pooled = self.cache
        y, ts = self.y_t
        ds = y.copy()
        ds[np.arange(N), ts] -= 1
        ds = (ds / N).astype('f')
        self.grads[-2][...] = pooled.T @ ds
        self.grads[-1][...] = ds.sum(axis=0)
        dpooled = ds @ self.W.T
        dhs = (span_mask[:, :, None] * dpooled[:, None, :] / cnt[:, :, None]).astype('f')
        self.embed.backward(self.rnn.backward(dhs))

    def predict_proba(self, xs, mask, span_mask):
        return softmax(self._score(xs, mask, span_mask))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([p.copy() for p in self.params], f)

    def load(self, path):
        with open(path, 'rb') as f:
            for p, q in zip(self.params, pickle.load(f)):
                p[...] = q


def encode_examples(examples, vocab):
    """(문장, (시작, 끝), 라벨) → 자모 ID, 구간 마스크(자모 단위), 라벨 ID"""
    X, M, T = [], [], []
    for sent, (s, e), lab in examples:
        ids, cidx = vocab.encode(sent)
        X.append(ids)
        M.append(np.array([1.0 if s <= c < e else 0.0 for c in cidx], 'f'))
        T.append(LABELS.index(lab) if isinstance(lab, str) else lab)
    return X, M, np.array(T)


def batch(X, M, idx):
    xs, mask = pad_batch([X[i] for i in idx])
    sm = np.zeros(xs.shape, 'f')
    for r, i in enumerate(idx):
        sm[r, :len(M[i])] = M[i]
    return xs, mask, sm


class UsageTagger:
    """순화기에서 쓰는 래퍼: (정규화된 문장, 구간) → (용법, 확률)"""

    def __init__(self, vocab, path=os.path.join(OUT, 'usage_lstm.pkl')):
        self.vocab = vocab
        self.model = UsageClassifier(len(vocab))
        self.model.load(path)

    def predict(self, text, s, e):
        X, M, _ = encode_examples([(text, (s, e), 0)], self.vocab)
        if not M[0].any():
            return None, None
        p = self.model.predict_proba(*batch(X, M, [0]))[0]
        return LABELS[int(p.argmax())], p


def load_vocab():
    with open(os.path.join(OUT, 'vocab.pkl'), 'rb') as f:
        ids = pickle.load(f)
    v = JamoVocab()
    v.id_to_tok, v.tok_to_id = list(ids), {t: i for i, t in enumerate(ids)}
    return v


def main():
    from real_data import load_clean_comments
    t0 = time.time()
    vocab = load_vocab()
    clean = load_clean_comments()
    rng = random.Random(0)
    rng.shuffle(clean)
    ex = []
    for i, s in enumerate(clean[:24000]):
        e = make_example(s, LABELS[i % 4], rng)
        if e:
            ex.append(e)
    rng.shuffle(ex)
    n_val = 2000
    val, tr = ex[:n_val], ex[n_val:]
    print(f'[용법 데이터] 학습 {len(tr)} · 합성 검증 {len(val)} · 라벨 분포 ' +
          ', '.join(f'{l} {sum(1 for x in tr if x[2] == l)}' for l in LABELS))
    for s, (a, b), l in tr[:6]:
        print(f'  [{l}] {s}   ← {s[a:b]!r}')
    X, M, T = encode_examples(tr, vocab)
    Xv, Mv, Tv = encode_examples(val, vocab)
    hw = []
    for sent, word, lab in HANDWRITTEN_USAGE:
        n = normalize(sent)
        i = n.find(word)
        hw.append((n, (i, i + len(word)), lab))
    Xh, Mh, Th = encode_examples(hw, vocab)

    model = UsageClassifier(len(vocab))
    opt, r = Adam(0.003), np.random.RandomState(0)
    acc = lambda Xs, Ms, Ts: float(np.mean(np.concatenate(
        [model.predict_proba(*batch(Xs, Ms, list(range(i, min(i + 256, len(Xs)))))).argmax(1)
         for i in range(0, len(Xs), 256)]) == Ts))
    hist = []
    for ep in range(1, 7):
        idx = r.permutation(len(X))
        tot = 0
        for st in range(0, len(idx), 32):
            b = idx[st:st + 32]
            tot += model.forward(*batch(X, M, b), T[b]) * len(b)
            model.backward()
            clip_grads(model.grads, 5.0)
            opt.update(model.params, model.grads)
        hist.append({'epoch': ep, 'loss': tot / len(X), 'syn_acc': acc(Xv, Mv, Tv), 'hand_acc': acc(Xh, Mh, Th)})
        print(f'  에폭 {ep} | 손실 {hist[-1]["loss"]:.4f} | 합성 검증 정확도 {hist[-1]["syn_acc"]:.3f} | '
              f'손글씨 32문장 정확도 {hist[-1]["hand_acc"]:.3f} | {time.time() - t0:.0f}s', flush=True)
    model.save(os.path.join(OUT, 'usage_lstm.pkl'))

    # 손글씨 문장별 결과 + 혼동 행렬
    P = model.predict_proba(*batch(Xh, Mh, list(range(len(Xh))))).argmax(1)
    conf = np.zeros((4, 4), int)
    for t, p in zip(Th, P):
        conf[t, p] += 1
    print('\n[손글씨 32문장] 혼동 행렬 (행 = 정답, 열 = 예측)  ' + ' '.join(LABELS))
    for i, l in enumerate(LABELS):
        print(f'  {l}  ' + '  '.join(f'{c:3d}' for c in conf[i]))
    print('\n[틀린 문장]')
    for (s, _, l), p in zip(hw, P):
        if LABELS[p] != l:
            print(f'  {s}  정답 {l} → 예측 {LABELS[p]}')
    # 비교 기준: 단어만 보고 학습 데이터에서 가장 흔했던 용법을 고르는 방식 (문맥을 안 봄)
    import collections
    maj = collections.defaultdict(collections.Counter)
    for sent, (a, b), lab in tr:
        maj[sent[a:b]][lab] += 1
    base = lambda w: maj[w].most_common(1)[0][0] if w in maj else '강조'
    tagger = UsageTagger.__new__(UsageTagger)
    tagger.vocab, tagger.model = vocab, model
    ctx = []
    for sent, word, lab in CONTEXT_TEST:
        n = normalize(sent)
        a = n.find(word)
        ctx.append((n, lab, base(word), tagger.predict(n, a, a + len(word))[0]))
    b_acc = float(np.mean([b == l for _, l, b, _ in ctx]))
    m_acc = float(np.mean([m == l for _, l, _, m in ctx]))
    print(f'\n[위치만 다른 {len(ctx)}문장] 단어 다수결 {b_acc:.3f} vs 양방향 LSTM {m_acc:.3f}')
    for n, l, b, m in ctx:
        print(f'  {n:14} 정답 {l} | 단어 다수결 {b} | LSTM {m}')
    syn_base = float(np.mean([base(s[a:e]) == l for s, (a, e), l in val]))
    print(f'  (합성 검증 2000개: 단어 다수결 {syn_base:.3f} vs LSTM {hist[-1]["syn_acc"]:.3f})')
    with open(os.path.join(OUT, 'usage_log.json'), 'w', encoding='utf-8') as f:
        json.dump({'history': hist, 'confusion': conf.tolist(), 'labels': LABELS, 'context_test': ctx,
                   'context_acc': {'단어 다수결': b_acc, 'LSTM': m_acc}, 'syn_base': syn_base}, f, ensure_ascii=False)
    print(f'\n저장: {OUT}/usage_lstm.pkl ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
