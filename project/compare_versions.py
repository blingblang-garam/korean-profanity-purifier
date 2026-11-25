# coding: utf-8
"""v2(학습 데이터 1차 개선) vs v3(붙여 쓰기·함정 음성 예시·LM 말뭉치 확대·조기 종료) 비교 그림

실행:  cd project && python compare_versions.py   → saved/final_v3/fig_v2_vs_v3.png
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_figures import BASE, MAIN  # 글꼴 설정도 함께 불러온다

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {'v2': os.path.join(HERE, 'saved', 'final', 'final_eval_results_s0_1_2.json'),
         'v3': os.path.join(HERE, 'saved', 'final_v3', 'final_eval_results_s0_1_2.json')}


def main():
    S = {v: json.load(open(p, encoding='utf-8'))['summary'] for v, p in FILES.items()}
    panels = [
        ('실제 욕설 200문장 J_lex', [('E4c 양방향 RNN', '순화/E4c +국소+계열/J_lex'), ('E9 LSTM', '순화/E9 LSTM탐지+LSTM LM(국소+계열)/J_lex')], (0, 0.7)),
        ('비속어 패턴 잔존율 (낮을수록 좋음)', [('E4c 양방향 RNN', '순화/E4c +국소+계열/패턴 잔존'), ('E9 LSTM', '순화/E9 LSTM탐지+LSTM LM(국소+계열)/패턴 잔존')], (0, 0.7)),
        ('명시적 욕설 재현율 (test 절반)', [('양방향 RNN', '탐지/양방향 RNN/명시적 재현율'), ('양방향 LSTM', '탐지/양방향 LSTM/명시적 재현율')], (0, 1.05)),
        ('손글씨 25문장 과교정 (낮을수록 좋음)', [('E4c 양방향 RNN', '손글씨/E4c +국소+계열/과교정'), ('E9 LSTM', '손글씨/E9 LSTM탐지+LSTM LM(국소+계열)/과교정')], (0, 1.05)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    w = 0.36
    for ax, (title, items, ylim) in zip(axes, panels):
        for j, (ver, col) in enumerate([('v2', BASE), ('v3', MAIN)]):
            ms = [S[ver][k][0] for _, k in items]
            sd = [S[ver][k][1] for _, k in items]
            xs = [i + (j - 0.5) * w for i in range(len(items))]
            ax.bar(xs, ms, w, yerr=sd, color=col, capsize=3, label=ver)
            for x, m in zip(xs, ms):
                ax.text(x, m + ylim[1] * 0.03, f'{m:.2f}', ha='center', fontsize=10)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([l for l, _ in items], fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
    axes[0].legend(frameon=False)
    fig.suptitle('학습 데이터 개선 전후 (v2 → v3, 시드 3개 평균 ± 표준편차)', fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, 'saved', 'final_v3', 'fig_v2_vs_v3.png')
    fig.savefig(out, dpi=200)
    print('저장:', out)


if __name__ == '__main__':
    main()
