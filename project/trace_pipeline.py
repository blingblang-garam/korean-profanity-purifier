# coding: utf-8
"""아키텍처 시각화용: 문장 하나가 최종 순화기를 지나가며 남기는 중간값 전부 (architecture.html이 그린다)

실행:  python trace_pipeline.py   → architecture_examples.json (서버 없이 html만 열어도 보이는 예시)
"""
import os
import json
from jamo import normalize, decompose
from purifier import dictionary_baseline

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = ['씨발 또 버그네', '그만 부들거려 노가다새끼들아', '짱깨들이 또 시끄럽네', '개나리가 피었다', '팀플 진짜 개빡친다', '시1발 오늘 존나 덥네']


def trace(purifier, text):
    raw = text.strip()[:120]
    norm = normalize(raw)
    toks, cidx = decompose(norm)
    ids, _ = purifier.vocab.encode(norm)
    jp = purifier.det.predict([ids])[0] if len(ids) else []
    T = {}
    r = purifier.purify(raw, trace=T)
    char_p = purifier.char_probs(norm) if norm else []
    L = purifier.lexicon
    for u in T.get('units', []):
        u['lex_hits'] = [{'head': e['head'], 'kind': e['kind'], 'meaning': e['meaning'],
                          'clean': [c for c in e['clean'] if c], 'dist': d} for d, e in (L.lookup(u['word'], 2) if L else [])]
        u['from_lexicon'] = any(u['start'] <= a < u['end'] for a, _ in T.get('lex_spans', []))
    return {
        'raw': raw, 'normalized': norm, 'dict': dictionary_baseline(raw),
        'jamo': [{'tok': t, 'char': c, 'p': round(float(p), 3)} for t, c, p in zip(toks, cidx, jp)],
        'char_probs': [round(float(p), 3) for p in char_p],
        'det_spans': T.get('det_spans', []), 'lex_spans': T.get('lex_spans', []),
        'units': T.get('units', []), 'output': r['output'], 'log': r['log'],
        'shapes': {'T': len(ids), 'V': len(purifier.vocab)}, 'lexicon_size': len(L) if L else 0,
    }


def main():
    import sys
    sys.argv = sys.argv[:1]
    import demo
    vocab, systems = demo.load_models(os.path.join(demo.FINAL, 'seed0'))
    p = systems[list(systems)[-1]]
    out = [trace(p, t) for t in EXAMPLES]
    with open(os.path.join(HERE, 'architecture_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 서버 없이 더블클릭으로 열 수 있는 단독 파일 (예시 문장의 실제 중간값을 html 안에 넣음)
    with open(os.path.join(HERE, 'architecture.html'), encoding='utf-8') as f:
        page = f.read()
    with open(os.path.join(HERE, 'architecture_standalone.html'), 'w', encoding='utf-8') as f:
        f.write(page.replace('__EXAMPLES_JSON__', json.dumps(out, ensure_ascii=False).replace('</', '<\\/')))
    for t in out:
        print(t['raw'], '→', t['output'], '|', [(u['word'], u.get('usage'), u.get('chosen')) for u in t['units']])


if __name__ == '__main__':
    main()
