# coding: utf-8
"""[v3+] 밈·신조어·비속어 사전 검색 — RAG의 '검색' 부분만 가져온 방식 [교과 외]

RAG:   질문 → 문서 검색 → LLM이 문서를 참고해 생성
여기:  탐지된 어절 → 사전(lexicon/meme_lexicon.tsv)에서 비슷한 항목 검색(자모 편집 거리) → 항목 정보로 후보 결정
       → 교재 RNN 언어 모델이 후보 중 선택

- 새 유행어·비하어는 사전에 한 줄 추가하면 바로 반영된다 (다시 학습할 필요 없음)
- '밈'·'정상' 항목은 보호 목록: 탐지기가 잡아도 그대로 둔다 (킹받네, 개나리, 시발역, 새끼 고양이)
- '욕설'·'비하' 항목은 순화 후보를 준다 (짱깨 → 중국인, 틀딱 → 노인)
- 인터넷 없이 동작. 사전 갱신은 발표 전에 파일을 고치는 방식
"""
import os
import re
from jamo import normalize, decompose

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, 'lexicon', 'meme_lexicon.tsv')
_NOISE = re.compile(r'[0-9.,!?~*]')


def _jamo(text):
    return [t for t in decompose(_NOISE.sub('', normalize(text)))[0] if t != '▁']


class Lexicon:
    def __init__(self, path=DEFAULT):
        self.entries = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                cols = (line.split('\t') + [''] * 7)[:7]
                head, variants, kind, clean, meaning, cond, date = cols
                forms = [head] + [v for v in variants.split('|') if v]
                nxt = cond[5:].split('|') if cond.startswith('next:') else []
                self.entries.append({'head': head, 'forms': [normalize(x) for x in forms], 'kind': kind,
                                     'clean': [c for c in clean.split('|')] if clean else [''],
                                     'meaning': meaning, 'next': nxt, 'date': date,
                                     'jamo': [_jamo(x) for x in forms]})

    def __len__(self):
        return len(self.entries)

    def protect(self, word, next_word=''):
        """어절이 '밈'·'정상' 항목으로 시작하면(조건이 있으면 다음 어절까지 맞으면) 그 항목을 돌려준다."""
        for e in self.entries:
            if e['kind'] not in ('밈', '정상'):
                continue
            if any(word.startswith(f) for f in e['forms']):
                if e['next'] and not any(next_word.startswith(n) for n in e['next']):
                    continue
                return e
        return None

    def find_spans(self, text):
        """검색으로 탐지 보완: 욕설·비하 항목이 글자 그대로 들어 있는 구간 [(시작, 끝)].
        탐지기가 학습 때 못 본 비하어(짱깨, 틀딱)도 사전에 있으면 잡는다. 한 글자 항목(ㅗ, 븅)은 어절 전체가 그것일 때만."""
        spans = []
        words, pos = text.split(' '), 0
        for wi, w in enumerate(words):
            nxt = words[wi + 1] if wi + 1 < len(words) else ''
            if not self.protect(w, nxt):
                for e in self.entries:
                    if e['kind'] not in ('욕설', '비하'):
                        continue
                    for f in e['forms']:
                        if not f or (len(f) < 2 and w != f):
                            continue
                        i = w.find(f)
                        while i >= 0:
                            spans.append((pos + i, pos + i + len(f)))
                            i = w.find(f, i + 1)
            pos += len(w) + 1
        return spans

    def family(self):
        """'욕설'·'비하' 항목 → purifier.family_candidates에 넣을 (자모, 순화어) 목록"""
        out = []
        for e in self.entries:
            if e['kind'] in ('욕설', '비하'):
                for j in e['jamo']:
                    for c in e['clean']:
                        out.append((j, c))
        return out

    def nouns(self):
        """조사를 다시 붙일 순화어 (비하 항목의 중립 표현 + 사람 계열)"""
        return {c for e in self.entries if e['kind'] == '비하' for c in e['clean'] if c} | {'사람', '이상한 사람'}

    def lookup(self, word, k=3):
        """검색: 어절과 자모 편집 거리가 가까운 항목 k개 → [(거리, 항목)] (데모 표시용)"""
        from purifier import _edit
        toks = _jamo(word)
        if not toks:
            return []
        scored = []
        for e in self.entries:
            if e['next']:                       # 문맥 조건이 있는 보호 항목('새끼'+고양이)은 검색 결과에 안 띄운다
                continue
            d = min(_edit(toks, j) / max(len(toks), len(j)) for j in e['jamo'] if j)
            # 어절 안에 항목이 그대로 들어 있으면 가깝게 본다 (노가다새끼들아 ⊃ 새끼)
            if any(f and f in normalize(word) for f in e['forms']):
                d = min(d, 0.1)
            scored.append((round(d, 2), e))
        scored.sort(key=lambda x: x[0])
        return scored[:k]
