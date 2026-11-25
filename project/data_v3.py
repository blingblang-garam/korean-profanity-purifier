# coding: utf-8
"""[v3] 학습 데이터 개선 — v2 평가·학습 곡선에서 찾은 문제를 데이터로 고친다 [교과 외]

v2에서 확인한 것
  · 합성 검증 F1은 0.99까지 오르는데 실제 댓글 재현율은 떨어짐 → 학습 데이터가 실제 댓글과 다르게 생김
  · 실제로 놓친 욕설 1위는 앞말에 붙여 쓴 '새끼'(남친새끼가, 노가다새끼들아) — v2 주입은 띄어 쓴 어절만 만듦
  · 남은 패턴 상위: 새끼 · 미친 · 씹 · 지랄 · 좆 · 애미 · 병신
  · 함정 단어 과교정(새끼 고양이 → 사람 고양이, 개나리 → 정말리)

v3에서 바꾼 것
  ① attach  : 앞말에 붙여 쓰기 (남친 → 남친새끼가, 기자 → 기자놈들) — 조사가 붙은 꼴까지
  ② prefix  : 접두 욕설 종류 확대 (개·씹·좆 + 어절: 개씹창, 좆같은)
  ③ adj     : 명사 앞 '미친'·'씨발' 끼워 넣기 (미친 기자, 씨발 회사)
  ④ address : 문장 끝 호칭 욕설 확대 (애미, 느금마, 미친놈아 ...)
  ⑤ 함정 음성 예시 : 깨끗한 실제 댓글에 '새끼 고양이', '개나리', '시발역' 같은 정상 표현을 넣고 라벨 0
  ⑥ 언어 모델 말뭉치 : HateSpeech 비라벨 댓글(약 200만 개)에서 정규식 + n-gram 심판 둘 다 통과한 문장 추가
"""
import os
import re
import random
from make_pairs import mutate
from real_data import inject_real, PROFANE_RE, DATA, MAX_LEN
from jamo import normalize

# 앞말에 붙는 욕설 명사 (조사 붙은 꼴 포함)
ATTACH = ['새끼', '새끼가', '새끼는', '새끼들', '새끼들이', '새끼들은', '새끼들아', '새끼야', '새기', '쉑', '쉑들',
          '놈', '놈이', '놈들', '놈들아', '년', '년이', '년들', '충', '새끼도', '새끼한테']
PREFIX = ['개', '개', '씹', '좆', '개씹']
ADJ = ['미친', '씨발', '좆같은', '병신같은', '개같은']
ADDRESS = ['새끼들아', '병신아', '병신들', '새끼야', '등신아', '애미', '니애미', '느금마', '미친놈아', '씹새끼야', '지랄하네', 'ㅅㅂ', 'ㅈㄹ']
TRAPS = ['새끼 고양이', '새끼 강아지', '개나리', '개구리', '개미', '개강', '개인', '시발역', '시발점', '신발', '존경',
         '병원', '시베리아', '년도', '놈놈놈', '조나단', '수박씨', '씨앗', '발바닥', '미친 듯이 노력', '개발자']


def _label(text, bad):
    return [bad if ch != ' ' else 0 for ch in text]


def inject_v3(sent, rng, p_mutate=0.5):
    """v2 inject_real 연산 + v3 연산 ①~④ 중 하나 이상. 반환 (오염 문장, 순화 정답(근사), 글자 라벨)"""
    words = sent.split(' ')
    op = rng.choice(['v2', 'v2', 'attach', 'attach', 'attach', 'prefix', 'adj', 'address'])
    if op == 'v2':
        return inject_real(sent, rng, p_mutate)
    slots = [k for k, w in enumerate(words) if re.fullmatch('[가-힣]{2,}', w)]
    if op in ('attach', 'prefix', 'adj') and not slots:
        return inject_real(sent, rng, p_mutate)
    pieces = [[(w, 0)] for w in words]
    clean_words = list(words)
    if op == 'attach':
        k = rng.choice(slots)
        w = words[k]
        stem = w[:-1] if len(w) >= 3 and w[-1] in '이가은는을를도의' else w     # 조사를 떼고 붙임
        bad = mutate(rng.choice(ATTACH), rng, p_mutate * 0.5)
        pieces[k] = [(stem, 0), (bad, 1)]
        clean_words[k] = stem + ' 사람'
    elif op == 'prefix':
        k = rng.choice(slots)
        pieces[k] = [(rng.choice(PREFIX), 1), (words[k], 0)]
        clean_words[k] = '정말 ' + words[k]
    elif op == 'adj':
        k = rng.choice(slots)
        pieces[k] = [(mutate(rng.choice(ADJ), rng, p_mutate), 1), (' ', 0), (words[k], 0)]
        clean_words[k] = words[k]
    else:
        pieces.append([(mutate(rng.choice(ADDRESS), rng, p_mutate), 1)])
    dirty, labels = '', []
    for ps in pieces:
        if dirty:
            dirty += ' '
            labels.append(0)
        for text, b in ps:
            dirty += text
            labels += _label(text, b)
    return dirty, ' '.join(w for w in clean_words if w), labels


def trap_negative(sent, rng):
    """깨끗한 댓글 + 함정 표현 (라벨 전부 0)"""
    words = sent.split(' ')
    words.insert(rng.randint(0, len(words)), rng.choice(TRAPS))
    s = ' '.join(words)
    return s, s, [0] * len(s)


def load_unlabeled_clean(judge, n, seed=0, exclude=()):
    """HateSpeech 비라벨 댓글 → 문장 단위로 쪼개고 정규화 → 길이 5~40, 비속어 패턴 없음, 심판 욕설 확률 < 0.2"""
    rng = random.Random(seed)
    exclude = set(exclude)
    paths = [os.path.join(DATA, 'korean-hate-speech', 'unlabeled', f'unlabeled_comments_{i}.txt') for i in range(1, 6)]
    cands = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                for seg in re.split(r'(?<=[.?!])\s+|\n', line.strip()):
                    s = normalize(seg)
                    if 5 <= len(s) <= MAX_LEN and re.search('[가-힣]', s) and not PROFANE_RE.search(s) and s not in exclude:
                        cands.append(s)
    cands = sorted(set(cands))
    rng.shuffle(cands)
    pool = cands[:n * 2]
    probs = judge.prob(pool)
    out = [s for s, p in zip(pool, probs) if p < 0.2][:n]
    return out, len(cands)
