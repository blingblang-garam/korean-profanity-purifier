# coding: utf-8
"""탐지 → 분기 → 후보 치환 → 언어 모델 선택 → 재검증 (교과 외 + Ch05)

욕설 재생성·과교정을 막는 3중 장치 [교과 외]
 ① 깨끗한 후보만: 순화 후보는 정해진 순화어 목록에서만 고르고, 언어 모델도 깨끗한 문장으로만 학습
 ② 탐지기 재검증: 바꿔 넣은 표현이 다시 비속어로 판정되면 다음 후보로
 ③ 원래 표현도 후보: 탐지기가 잘못 잡은 정상 표현은 언어 모델이 더 자연스럽다고 보면 그대로 남김
"""
import re
import numpy as np
from jamo import normalize, decompose, to_choseong
from make_pairs import REVERSE, PROFANITY

CANDIDATES = list(REVERSE.keys())          # 순화어 목록 (정말, 너무, 사람, 화나게 ...)
INVERSE = {}                                # 고정 치환용: 비속어 → 첫 번째 순화어
for clean, bads in REVERSE.items():
    for b in bads:
        INVERSE.setdefault(b, clean)


# ── [v2] 비속어 계열 → 순화어 후보 제한 + 조사 보존 ──
# 1차 실데이터에서 '존나' 자리에 '사람들아'가 뽑히는 등 품사가 안 맞는 후보가 골라졌다.
# 탐지 구간을 역방향 사전의 비속어 형태와 자모 편집 거리로 맞춰 보고, 같은 계열의 순화어만 후보로 남긴다.
PARTICLES = sorted(['들아', '들이', '들은', '들을', '들도', '들', '아', '야', '이', '가', '은', '는', '을', '를',
                    '도', '만', '이랑', '같은', '같이', '처럼', '한테', '이나', '나'], key=len, reverse=True)
NOUN_CLEANS = {'사람', '애들', '바보', '사람들', '난리'}    # 조사를 다시 붙이는 순화어 (부사·서술어에는 안 붙임)
_NOISE = re.compile(r'[0-9.,!?~*]')
# 학습 데이터 생성(make_pairs.REVERSE)에는 없지만 실제 댓글에서 흔한 꼴 — 순화 후보 찾기에만 쓴다
EXTRA_FAMILY = {
    '이상한': ['좆같은', '개좆같은', '병신같은', '븅신같은', '씨발같은'],
    '너무': ['시발', '씹', '개씹'],
    '진짜': ['시발', '씨바'],
    '정말': ['존나게', '졸라게', '존내', '개씹'],
    '사람': ['새기', '쉐끼', '색기', '놈아', '쉑'],
    '사람들': ['새기들', '쉐끼들', '쉑들', '쉑덜'],
    '난리': ['지랄', '염병', 'ㅈㄹ', 'ㅇㅈㄹ', '지랄발광'],
    '바보': ['ㅂㅅ', '빙신', '병1신'],
}
FAMILY = []                                  # (비속어 형태의 자모, 순화어)
for _clean, _bads in list(REVERSE.items()) + list(EXTRA_FAMILY.items()):
    for _b in _bads:
        for _form in {_b, to_choseong(_b)}:
            FAMILY.append((decompose(_form)[0], _clean))


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ── [v4] 용법·강도에 맞춘 후보 (usage_model.py의 분류 결과 사용) ──
INTENS = {'중': ['정말', '너무', '진짜'], '강': ['진짜 너무', '완전', '엄청']}
EXCL_C = {'중': ['아', '아 진짜'], '강': ['아오', '아 진짜']}
_STRONG_PREFIX = re.compile(r'^(개|씹|좆)[가-힣]')


def intensity(text, word, n_spans):
    """강도 규칙: 접두 욕설이 붙은 어절(개빡친다), 같은 욕 반복(존나존나), 느낌표 2개 이상이면 '강', 아니면 '중'"""
    if _STRONG_PREFIX.match(word) or text.count('!') >= 2 or re.search(r'(존나|ㅈㄴ|씨발){2}', text):
        return '강'
    return '중'


JOSA_PAIRS = {'이': '가', '가': '이', '은': '는', '는': '은', '을': '를', '를': '을', '아': '야', '야': '아',
              '이랑': '랑', '랑': '이랑', '이나': '나', '나': '이나'}


def _josa(word, particle):
    """순화어 받침에 맞게 조사를 고른다 (지랄이 → 난리가, 븅신아 → 바보야). 받침 있으면 이·은·을·아, 없으면 가·는·를·야."""
    if particle not in JOSA_PAIRS or not word:
        return particle
    code = ord(word[-1]) - 0xAC00
    has_jong = 0 <= code <= 11171 and code % 28 != 0
    first = {'이': True, '가': False, '은': True, '는': False, '을': True, '를': False, '아': True, '야': False,
             '이랑': True, '랑': False, '이나': True, '나': False}
    if first.get(particle) is None:
        return particle
    return particle if first[particle] == has_jong else JOSA_PAIRS[particle]


def family_match(word, max_dist=0.34, family=None):
    """구간 글자열 → (가장 가까운 비속어 계열의 순화어 목록, 거리). 숫자·기호 삽입(시1발)은 지우고 비교."""
    toks = [t for t in decompose(_NOISE.sub('', word))[0] if t != '▁']
    if not toks:
        return [], 1.0
    scored = {}
    for form, clean in (FAMILY if family is None else family):
        d = _edit(toks, form) / max(len(toks), len(form))
        scored[clean] = min(scored.get(clean, 1.0), d)
    best = min(scored.values())
    if best > max_dist:
        return [], best
    return sorted(c for c, d in scored.items() if d <= best + 1e-9), best


def family_candidates(span, overlap=None, family=None, nouns=None):
    """구간(어절) → (순화 후보, 계열 거리).
    - 끝에 조사가 붙어 있으면 떼고 맞춰 본 뒤 순화어에 다시 붙인다 (새끼들이 → 사람들이). 부사·서술어에는 안 붙인다.
    - 어절에 정상 글자가 섞여 있으면(노가다새끼들아) 가장 잘 맞는 부분만 바꾸고 나머지는 남긴다.
      overlap=(s, e)이면 탐지기가 잡은 글자와 겹치는 부분만 고려한다.
    - 어느 계열과도 안 맞으면 ([], 1.0)"""
    best, out, best_d = None, [], 1.0
    n = len(span)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if overlap is not None and (j <= overlap[0] or i >= overlap[1]):
                continue
            piece = span[i:j]
            if not re.search('[가-힣ㄱ-ㅎ]', piece):
                continue
            options = [(piece, '')] + [(piece[:-len(p)], p) for p in PARTICLES if piece.endswith(p) and len(piece) > len(p)]
            for stem, particle in options:
                cleans, d = family_match(stem, family=family)
                if not cleans:
                    continue
                cost = d + 0.15 * (n - len(piece))            # 어절을 많이 덮을수록 유리
                if best is None or cost < best - 1e-9:
                    pre, post = span[:i], span[j:]
                    if pre and family_match(pre, family=family)[0]:              # 앞 조각도 비속어면 지움 (개빡치게, 병신새끼들)
                        pre = ''
                    pre = pre + ' ' if len(pre) >= 2 else pre
                    best, best_d = cost, d
                    out = [pre + c + (_josa(c, particle) if c in (nouns or NOUN_CLEANS) else '') + post for c in cleans]
    return out, best_d


def find_spans(char_probs, text, thr, low_ratio=0.3):
    """글자 확률 → 연속 구간 [(start, end)]. 공백은 구간을 끊는다.
    히스테리시스: thr 이상인 글자에서 시작해, 같은 어절 안의 이웃 글자가 thr*low_ratio 이상이면 구간을 넓힌다
    (실제 댓글에서 '병신새끼들'의 일부만 잡혀 '진짜신사람' 같은 조각 출력이 나오던 문제 대응)."""
    low = thr * low_ratio
    on = [ch != ' ' and p >= thr for ch, p in zip(text, char_probs)]
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(text):
            if not on[i] and ch != ' ' and char_probs[i] >= low and \
                    ((i > 0 and on[i - 1]) or (i + 1 < len(text) and on[i + 1])):
                on[i] = changed = True
    spans, start = [], None
    for i in range(len(text)):
        if on[i] and start is None:
            start = i
        if not on[i] and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _join(left, mid, right):
    return re.sub(r'\s+', ' ', left + mid + right).strip()


class Purifier:
    def __init__(self, detector, lm, vocab, thr=0.5, mode='lm', recheck=True, keep_original=True,
                 score='sentence', family=False, window=8, lexicon=None, usage=None):
        """mode='lm' → E4(언어 모델 후보 선택), mode='fixed' → E2/E3(고정 치환)
        [v2] score='local' → 바꿔 넣은 자리부터 뒤 window 자모까지만 평균 교차 엔트로피로 채점
             family=True  → 구간과 같은 비속어 계열의 순화어(+조사)만 후보로 (못 맞추면 전체 목록)"""
        self.det, self.lm, self.vocab = detector, lm, vocab
        self.thr, self.mode = thr, mode
        self.recheck, self.keep_original = recheck, keep_original
        self.score, self.family, self.window = score, family, window
        self.unknown_thr = 0.9
        # [v3+] 밈·신조어 사전: 보호 목록 + 욕설·비하 계열 추가 (lexicon.py)
        self.lexicon = lexicon
        self.fam = FAMILY + (lexicon.family() if lexicon else [])
        self.nouns = NOUN_CLEANS | (lexicon.nouns() if lexicon else set())
        self.usage = usage                       # [v4] usage_model.UsageTagger

    @staticmethod
    def _usage_cands(lab, lvl, fam):
        """용법·강도 → 후보. 지칭은 계열 후보(사람 …) 그대로, 나머지는 강도를 살리는 표현을 더한다."""
        pure = all(c in INTENS['중'] or c in ('아',) for c in fam)      # 어절 전체가 강조어·감탄사뿐 (존나, 씨발)
        if lab == '감탄':
            return EXCL_C[lvl] + ([] if pure else fam)
        if lab == '강조':
            if pure:
                return INTENS[lvl]
            # 개빡친다: 접두 '개'(강조)는 강도 부사로 살리고 나머지는 계열 순화어 → '진짜 너무 화난다'
            return [a + ' ' + c for a in INTENS[lvl][:2] for c in fam] + fam
        if lab == '모욕' and lvl == '강':
            return ['진짜 ' + c for c in fam] + fam
        return fam

    @staticmethod
    def _eojeol_units(text, spans):
        """[v2] 탐지 구간을 어절 경계까지 넓힌다 → (어절 시작, 어절 끝, 탐지 시작, 탐지 끝).
        한 어절에 구간이 여러 개면 하나로 합친다 ('병끼도있냐' 같은 조각 출력 방지)."""
        units = []
        for s, e in spans:
            ws = text.rfind(' ', 0, s) + 1
            we = text.find(' ', e)
            we = len(text) if we < 0 else we
            if units and units[-1][0] == ws:
                units[-1] = (ws, we, units[-1][2], e)
            else:
                units.append((ws, we, s, e))
        return units

    def _scores(self, sents, cands):
        """후보 문장 → 점수 (낮을수록 자연스럽다)"""
        ids = [self.vocab.encode(x)[0] for x in sents]
        if self.score == 'sentence':
            return self.lm.sentence_perplexity(ids)
        # 국소: 모든 후보 문장이 공유하는 앞부분(왼쪽 문맥)은 점수가 같으므로 빼고,
        # 바꿔 넣은 자리 ~ 그 뒤 window 자모(문장 끝 포함)만 평균한다.
        start = min(len(x) for x in ids)
        for k in range(start):
            if any(x[k] != ids[0][k] for x in ids):
                start = k
                break
        shortest = min(len(x) for x in ids)
        ce = self.lm.token_ce(ids)
        out = []
        for x, c in zip(ids, ce):
            end = start + (len(x) - shortest) + self.window
            out.append(float(np.exp(c[start:end + 1].mean())))
        return np.array(out)

    def char_probs(self, text):
        ids, char_idx = self.vocab.encode(text)
        if len(ids) == 0:
            return [0.0] * len(text)
        p = self.det.predict([ids])[0]
        out = [0.0] * len(text)
        for prob, c in zip(p, char_idx):
            out[c] = max(out[c], float(prob))
        return out

    def detect(self, text):
        text = normalize(text)
        probs = self.char_probs(text)
        return text, find_spans(probs, text, self.thr), probs

    def purify(self, text, verbose=False, trace=None):
        """trace에 dict를 넘기면 단계별 중간값을 기록한다 (아키텍처 시각화용, trace_pipeline.py)"""
        text, spans, probs = self.detect(text)
        log = []
        T = trace if trace is not None else {}
        T.update({'det_spans': list(spans), 'lex_spans': [], 'units': []})
        if self.lexicon is not None:                     # 탐지기 ∪ 사전 검색
            extra = [sp for sp in self.lexicon.find_spans(text)
                     if not any(sp[0] < e and s < sp[1] for s, e in spans)]
            if extra:
                T['lex_spans'] = list(extra)
                for s, e in extra:
                    probs[s:e] = [1.0] * (e - s)
                spans = sorted(spans + extra)
                log.append('사전 검색으로 추가: ' + ', '.join(repr(text[s:e]) for s, e in extra))
        if not spans:                                    # 분기: 비속어 없으면 원문 그대로
            return {'input': text, 'output': text, 'spans': [], 'log': log}

        found = [text[s:e] for s, e in spans]
        out = text
        units = [(s, e, s, e) for s, e in spans]
        if self.family:
            units = self._eojeol_units(text, spans)
        # 뒤 구간부터 바꿔야 앞 구간 인덱스가 밀리지 않는다
        for s, e, ds, de in reversed(units):
            span = out[s:e]
            left, right = out[:s], out[e:]
            U = {'word': span, 'start': s, 'end': e, 'detected': text[ds:de]}
            T['units'].insert(0, U)
            if self.mode == 'fixed':
                out = _join(left, INVERSE.get(span, '**'), right)
                continue

            keep = self.keep_original
            if self.family:
                if self.lexicon is not None:
                    nxt = right.strip().split(' ')[0] if right.strip() else ''
                    hit = self.lexicon.protect(span, nxt)
                    if hit:
                        log.append(f'{span!r} 유지 (사전: {hit["kind"]} {hit["head"]!r})')
                        U.update(protected={'kind': hit['kind'], 'head': hit['head'], 'meaning': hit['meaning']})
                        continue
                fam, d = family_candidates(span, (ds - s, de - s), self.fam, self.nouns)
                U.update(family=list(fam), family_dist=round(float(d), 3))
                if fam:
                    # 사전의 비하 표현(짱깨, 틀딱)은 지우지 않고 중립 표현으로 바꾼다 (지우면 누구 얘기인지 사라짐)
                    slur = self.lexicon is not None and any(
                        f in span for e in self.lexicon.entries if e['kind'] == '비하' for f in e['forms'] if len(f) >= 2)
                    cands = fam if slur and all(fam) else fam + ['']
                    if self.usage is not None and not slur:
                        lab, up = self.usage.predict(text, ds, de)
                        lvl = intensity(text, span, len(units))
                        U.update(usage=lab, usage_probs=[round(float(x), 3) for x in up], intensity=lvl)
                        cands = self._usage_cands(lab, lvl, fam) + ['']
                        log.append(f'{span!r} 용법 {lab} · 강도 {lvl}')
                        prev = left.split()[-1] if left.split() else ''
                        kept = [c for c in cands if not (c and c.split()[0] == prev)]   # '진짜 진짜 화난다' 방지
                        cands = kept if any(kept) else cands
                    keep = keep and d > 0.2              # 사전 형태와 거의 같은 확실한 비속어는 원래 표현을 후보에서 뺀다
                elif max(probs[ds:de]) >= self.unknown_thr:
                    cands = ['']                          # 계열 모름 + 탐지기 확신 → 어절 삭제 / 원래 표현 비교
                else:
                    cands = []                            # 계열 모름 + 애매 → 오탐으로 보고 그대로 둔다
                    log.append(f'{span!r} 유지 (계열 없음, p={max(probs[ds:de]):.2f})')
                    continue
            else:
                cands = CANDIDATES + ['']
            if keep:
                cands = cands + [span]
            sents = [_join(left, c, right) for c in cands]
            ppl = self._scores(sents, cands)
            order = sorted(range(len(cands)), key=lambda k: ppl[k])
            U.update(cands=[{'cand': cands[k], 'sent': sents[k], 'score': round(float(ppl[k]), 2)} for k in order], recheck=[])
            chosen = order[0]
            for k in order:
                c = cands[k]
                if c == span or c == '' or not self.recheck:
                    chosen = k
                    break
                cs = len(left)                                  # 바꿔 넣은 구간만 재검증
                p = self.char_probs(sents[k])[cs:cs + len(c)]
                U['recheck'].append({'cand': c, 'max_p': round(float(max(p)), 3), 'pass': bool(max(p) < self.thr)})
                if max(p) < self.thr:
                    chosen = k
                    break
                log.append(f'재검증 탈락: {c!r} (p={max(p):.2f})')
            log.append(f'{span!r} → {cands[chosen]!r}  (ppl {ppl[chosen]:.2f}, 1순위 {cands[order[0]]!r} {ppl[order[0]]:.2f})')
            U['chosen'] = cands[chosen]
            out = sents[chosen]
        if verbose:
            print('\n'.join(log))
        return {'input': text, 'output': out, 'spans': found, 'log': log}


def dictionary_baseline(text):
    """E1: 금칙어 사전 정확히 일치하는 것만 ** 로 가림 (변형 욕설은 통과)"""
    text = normalize(text)
    for w in PROFANITY:
        text = text.replace(w, '**')
    return text
