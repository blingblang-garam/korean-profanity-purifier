# coding: utf-8
"""발표 데모: 문장을 넣으면 탐지 확률 → 구간 → 후보별 점수 → 순화 결과를 한 화면에 보여 준다.

실행:  cd project && python demo.py            → 브라우저에서 http://localhost:8765
       python demo.py --cli                   → 터미널에서 한 줄씩 입력
모델:  final_pipeline.py가 저장한 saved/final/seed0/ (없으면 먼저 final_pipeline.py 실행)
외부 라이브러리 없이 표준 http.server만 사용 → 발표장 노트북에서 인터넷 없이 동작
"""
import sys
import os
import json
import pickle
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

from jamo import normalize, decompose
from loader import JamoVocab
from models import RnnDetector, SentenceLM, LstmSentenceLM
from purifier import Purifier, dictionary_baseline, family_candidates, intensity
from lexicon import Lexicon

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, 'saved', 'final_v3')       # v3 모델 (없으면 --models saved/final 로 v2)

EXAMPLES = ['씨발 또 버그네', '오늘 씨발 덥네', '팀플 진짜 개빡친다', '짱깨들이 또 시끄럽네', '오늘 진짜 킹받네', '시1발 오늘 존나 덥네', '그만 부들거려 노가다새끼들아', '실업급여 허위로 타먹은적 있는 새끼들은 돌 던질 자격없음',
            '꼴리지는 않고 그냥 존나 귀엽다', 'ㅅㅂ 과제 ㅈㄴ 많아', '새끼 고양이가 너무 귀엽다', '시발역에서 내렸다',
            '개나리가 피었다', '팀플 진짜 개빡친다', '저 ㅅㄲ 왜 저래']


def load_models(seed_dir):
    with open(os.path.join(os.path.dirname(seed_dir), 'vocab.pkl'), 'rb') as f:
        id_to_tok = pickle.load(f)
    vocab = JamoVocab()
    vocab.id_to_tok = list(id_to_tok)
    vocab.tok_to_id = {t: i for i, t in enumerate(id_to_tok)}
    V, bos = len(vocab), vocab.tok_to_id['<s>']
    m = {}
    for key, bi, cell in [('단방향_RNN', False, 'rnn'), ('양방향_RNN', True, 'rnn'), ('양방향_LSTM', True, 'lstm')]:
        det = RnnDetector(V, 32, 64, bidirectional=bi, cell=cell)
        det.load(os.path.join(seed_dir, f'det_{key}.pkl'))
        m[key] = det
    rnn_lm = SentenceLM(V, 48, 128, bos_id=bos)
    rnn_lm.load(os.path.join(seed_dir, 'lm_RNN_LM.pkl'))
    lstm_lm = LstmSentenceLM(V, 48, 128, bos_id=bos)
    lstm_lm.load(os.path.join(seed_dir, 'lm_LSTM_LM.pkl'))
    systems = {
        '1차 방식 (문장 퍼플렉서티)': Purifier(m['양방향_RNN'], rnn_lm, vocab),
        '양방향 RNN + RNN LM (국소+계열)': Purifier(m['양방향_RNN'], rnn_lm, vocab, score='local', family=True),
        '양방향 LSTM + LSTM LM (국소+계열)': Purifier(m['양방향_LSTM'], lstm_lm, vocab, score='local', family=True),
        'LSTM + 밈·비속어 사전 검색': Purifier(m['양방향_LSTM'], lstm_lm, vocab, score='local', family=True,
                                         lexicon=Lexicon()),
    }
    upath = os.path.join(os.path.dirname(seed_dir), 'usage_lstm.pkl')
    if os.path.exists(upath):                           # [v4] 용법 분류기가 있으면 최종 = 사전 검색 + 용법·강도
        from usage_model import UsageTagger
        systems['최종: + 용법·강도 (강조·감탄·지칭·모욕)'] = Purifier(
            m['양방향_LSTM'], lstm_lm, vocab, score='local', family=True, lexicon=Lexicon(), usage=UsageTagger(vocab, upath))
    return vocab, systems


def analyze(text, systems, main_key):
    """한 문장 → 화면에 그릴 JSON"""
    text = text.strip()[:120]
    res = {'input': normalize(text), 'dict': dictionary_baseline(text), 'systems': []}
    for name, p in systems.items():
        r = p.purify(text)
        _, spans, probs = p.detect(text)
        res['systems'].append({'name': name, 'output': r['output'], 'spans': spans, 'log': r['log'],
                               'probs': [round(float(x), 3) for x in probs]})
    # 메인 시스템의 후보별 점수 표 (발표용: 왜 이 후보가 뽑혔는지) — Purifier.purify와 같은 순서로 다시 계산
    p = systems[main_key]
    L = p.lexicon
    norm, spans, probs = p.detect(text)
    lexspans = []
    if L is not None:
        lexspans = [sp for sp in L.find_spans(norm) if not any(sp[0] < e and s < sp[1] for s, e in spans)]
        spans = sorted(spans + lexspans)
    table = []
    for s, e, ds, de in Purifier._eojeol_units(norm, spans):
        word = norm[s:e]
        left, right = norm[:s], norm[e:]
        nxt = right.strip().split(' ')[0] if right.strip() else ''
        item = {'word': word, 'detected': norm[ds:de], 'rows': [], 'protected': '', 'lex': []}
        if L is not None:
            item['lex'] = [{'head': x['head'], 'kind': x['kind'], 'meaning': x['meaning'],
                            'clean': ', '.join(c or '(삭제)' for c in x['clean']), 'dist': d} for d, x in L.lookup(word, 2)]
            hit = L.protect(word, nxt)
            if hit:
                item['protected'] = f"{hit['kind']} '{hit['head']}' — {hit['meaning']}"
                table.append(item)
                continue
        fam, d = family_candidates(word, (ds - s, de - s), p.fam, p.nouns)
        item['family_dist'] = round(d, 2)
        if fam:
            slur = L is not None and any(f in word for x in L.entries if x['kind'] == '비하' for f in x['forms'] if len(f) >= 2)
            cands = fam if slur and all(fam) else fam + ['']
            if p.usage is not None and not slur:
                lab, prob = p.usage.predict(norm, ds, de)
                lvl = intensity(norm, word, 0)
                item['usage'] = f'{lab} ({prob.max():.2f}) · 강도 {lvl}'
                cands = Purifier._usage_cands(lab, lvl, fam) + ['']
                prev = left.split()[-1] if left.split() else ''
                kept = [c for c in cands if not (c and c.split()[0] == prev)]
                cands = kept if any(kept) else cands
            if d > 0.2:
                cands.append(word)
        else:
            cands = ['', word]
        sents = [' '.join((left + c + right).split()) for c in cands]
        sc = p._scores(sents, cands)
        rows = sorted(zip(cands, sents, sc.tolist()), key=lambda x: x[2])
        item['rows'] = [{'cand': c or '(삭제)', 'sent': st, 'score': round(v, 2)} for c, st, v in rows]
        table.append(item)
    res['lexspans'] = lexspans
    res['table'] = table
    res['jamo'] = ' '.join(decompose(norm)[0])
    return res


PAGE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>비속어 순화기 데모</title><style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--sub:#6b6b70;--line:#e3e3e0;--acc:#1f7a4d;--bad:#c0392b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}.lead{color:var(--sub);margin:0 0 20px}
.row{display:flex;gap:8px}input{flex:1;font-size:20px;padding:14px;border:1px solid var(--line);border-radius:10px}
button{font-size:18px;padding:0 22px;border:0;border-radius:10px;background:var(--acc);color:#fff;cursor:pointer}
.ex{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 22px}.ex span{font-size:14px;padding:5px 10px;border:1px solid var(--line);border-radius:20px;background:#fff;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:14px}
.card h2{font-size:15px;color:var(--sub);margin:0 0 10px;font-weight:600}.out{font-size:22px;line-height:1.6}
.heat span{display:inline-block;font-size:24px;padding:4px 2px;border-radius:4px;min-width:14px;text-align:center}
.sys{display:grid;grid-template-columns:260px 1fr;gap:6px 14px;font-size:18px;align-items:baseline}.sys .n{color:var(--sub);font-size:14px}
table{border-collapse:collapse;width:100%;font-size:16px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
th{color:var(--sub);font-weight:600;font-size:13px}tr.best td{font-weight:700;color:var(--acc)}.use{display:inline-block;font-size:13px;padding:2px 8px;border-radius:10px;background:#eef5f0;color:var(--acc);margin-left:6px}
.lex{font-size:14px;color:#2f6f9f;margin:4px 0 8px}.lex .dist{color:var(--sub);font-size:12px}.keep{font-size:16px;color:var(--acc);font-weight:600}
.jamo{font-family:ui-monospace,monospace;font-size:13px;color:var(--sub);word-break:break-all}
@media (max-width:640px){.sys{grid-template-columns:1fr}}
</style></head><body><main>
<h1>문맥 보존형 비속어 순화기</h1><p class="lead">탐지(양방향 LSTM + 밈·비속어 사전 검색) → 어절 단위 구간 → 용법·강도 분류 → 같은 계열 순화어 후보 → 언어 모델 국소 퍼플렉서티로 선택 → 재검증</p>
<div class="row"><input id="q" placeholder="문장을 입력하세요" autofocus><button onclick="go()">순화</button></div>
<div class="ex" id="ex"></div><div id="res"></div></main>
<script>
const EX=__EXAMPLES__;const ex=document.getElementById('ex');
EX.forEach(t=>{const s=document.createElement('span');s.textContent=t;s.onclick=()=>{q.value=t;go()};ex.appendChild(s)});
const q=document.getElementById('q');q.addEventListener('keydown',e=>{if(e.key==='Enter')go()});
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function go(){if(!q.value.trim())return;const r=await fetch('/api?q='+encodeURIComponent(q.value));const d=await r.json();
const main=d.systems[d.systems.length-1];let h='';
h+='<div class="card"><h2>① 탐지 확률 (글자별, 진할수록 비속어) — '+esc(main.name)+'</h2><div class="heat">';
const inLex=i=>d.lexspans.some(([a,b])=>i>=a&&i<b);
[...d.input].forEach((ch,i)=>{const p=main.probs[i]||0;h+='<span style="background:rgba(192,57,43,'+p.toFixed(2)+');color:'+(p>.55?'#fff':'inherit')+(inLex(i)?';outline:2px solid #2f6f9f':'')+'" title="'+p+'">'+(ch===' '?'&nbsp;':esc(ch))+'</span>'});
h+='</div>'+(d.lexspans.length?'<p class="jamo" style="color:#2f6f9f">파란 테두리 = 탐지기는 놓쳤지만 사전 검색으로 찾은 글자</p>':'')+'<p class="jamo">자모 입력: '+esc(d.jamo)+'</p></div>';
if(d.table.length){h+='<div class="card"><h2>② 후보 채점 (국소 퍼플렉서티, 낮을수록 자연스러움)</h2>';
d.table.forEach(t=>{h+='<p>어절 <b>'+esc(t.word)+'</b> (탐지: '+esc(t.detected)+(t.family_dist!==undefined?', 계열 거리 '+t.family_dist:'')+')'+(t.usage?' <span class="use">용법 '+esc(t.usage)+'</span>':'')+'</p>';
if(t.lex.length){h+='<p class="lex">사전 검색: '+t.lex.map(x=>'<b>'+esc(x.head)+'</b> ['+esc(x.kind)+'] '+esc(x.meaning)+(x.kind==='욕설'||x.kind==='비하'?' → '+esc(x.clean):'')+' <span class="dist">거리 '+x.dist+'</span>').join(' · ')+'</p>'}
if(t.protected){h+='<p class="keep">그대로 둠 — 사전 보호 항목: '+esc(t.protected)+'</p>';return}
h+='<table><tr><th>후보</th><th>바꾼 문장</th><th>점수</th></tr>';
t.rows.forEach((r,i)=>{h+='<tr class="'+(i===0?'best':'')+'"><td>'+esc(r.cand)+'</td><td>'+esc(r.sent)+'</td><td>'+r.score+'</td></tr>'});h+='</table>'});h+='</div>'}
h+='<div class="card"><h2>③ 결과 비교</h2><div class="sys"><div class="n">입력</div><div>'+esc(d.input)+'</div><div class="n">E1 금칙어 사전</div><div>'+esc(d.dict)+'</div>';
d.systems.forEach(s=>{h+='<div class="n">'+esc(s.name)+'</div><div class="'+(s===main?'out':'')+'">'+esc(s.output)+'</div>'});h+='</div></div>';
document.getElementById('res').innerHTML=h}
</script></body></html>'''


def serve(port, systems, main_key):
    page = PAGE.replace('__EXAMPLES__', json.dumps(EXAMPLES, ensure_ascii=False)).encode('utf-8')

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            if u.path == '/api':
                q = parse_qs(u.query).get('q', [''])[0]
                body = json.dumps(analyze(q, systems, main_key), ensure_ascii=False).encode('utf-8')
                ctype = 'application/json; charset=utf-8'
            else:
                body, ctype = page, 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f'데모: http://localhost:{port}  (Ctrl+C로 종료)')
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8765)))
    ap.add_argument('--cli', action='store_true')
    ap.add_argument('--models', default=FINAL, help='모델 폴더 (기본 saved/final_v3)')
    args = ap.parse_args()
    vocab, systems = load_models(os.path.join(args.models, f'seed{args.seed}'))
    main_key = list(systems)[-1]                    # 최종: LSTM + 사전 검색 (평가_분석_요약.md)
    if args.cli:
        print('문장을 입력하세요 (빈 줄이면 종료)')
        for line in sys.stdin:
            if not line.strip():
                break
            r = analyze(line, systems, main_key)
            print(f'  사전   : {r["dict"]}')
            for s in r['systems']:
                print(f'  {s["name"][:22]:22}: {s["output"]}')
        return
    serve(args.port, systems, main_key)


if __name__ == '__main__':
    main()
