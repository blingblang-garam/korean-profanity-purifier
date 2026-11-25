# coding: utf-8
"""saved/final/results.json → 발표용 그림 (saved/final/fig_*.png)

실행:  cd project && python make_figures.py
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
FINAL = os.path.join(HERE, 'saved', 'final_v3' if 'v3' in sys.argv[1:] else 'final')   # python make_figures.py v3

for name in ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic']:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

BASE, MAIN, GRAY = '#9aa5ad', '#1f7a4d', '#d5d9dc'


def bars(ax, labels, means, sds, colors, fmt='{:.2f}', ylim=None):
    x = range(len(labels))
    ax.bar(x, means, yerr=sds, color=colors, capsize=4, edgecolor='none')
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + (ylim[1] if ylim else max(means)) * 0.02, fmt.format(m), ha='center', va='bottom', fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    with open(os.path.join(FINAL, 'final_eval_results_s0_1_2.json'), encoding='utf-8') as f:
        R = json.load(f)
    S = R['summary']
    n = len(R['seeds'])
    g = lambda k: S[k]

    # ① 실제 욕설 200문장: J (심판) / J_lex (패턴 잔존) 시스템별
    systems = [('E1 금칙어 사전', 'E1\n사전'), ('E3 양방향+고정치환', 'E3\n고정치환'), ('E4 양방향+RNNLM(문장)', 'E4 (1차)\n문장 채점'),
               ('E4a +국소 채점', 'E4a\n+국소'), ('E4b +계열 후보', 'E4b\n+계열'), ('E4c +국소+계열', 'E4c\n+국소+계열'),
               ('E9 LSTM탐지+LSTM LM(국소+계열)', 'E9\nLSTM'), ('E10 E9+밈·비속어 사전 검색', 'E10\n+사전 검색')]
    systems = [x for x in systems if f'순화/{x[0]}/J' in S]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, metric, title in [(axes[0], 'J', 'J = STA × SIM × FL (심판 = 글자 n-gram 분류기)'),
                              (axes[1], 'J_lex', 'J_lex (STA = 비속어 패턴이 안 남았는가)')]:
        ms = [g(f'순화/{k}/{metric}')[0] for k, _ in systems]
        sd = [g(f'순화/{k}/{metric}')[1] for k, _ in systems]
        best = max(range(len(ms)), key=lambda i: ms[i])
        colors = [MAIN if i == best else (BASE if i < 3 else GRAY) for i in range(len(ms))]
        bars(ax, [l for _, l in systems], ms, sd, colors, ylim=(0, 1))
        ax.set_title(title, fontsize=12)
    fig.suptitle(f'실제 욕설 200문장 순화 (시드 {n}개 평균 ± 표준편차)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL, 'fig_purify_200.png'), dpi=200)

    # ② 탐지: 정밀도 / 명시적 재현율 / 정상 문장 오탐률
    dets = ['E1 금칙어 사전', '단방향 RNN', '양방향 RNN', '양방향 LSTM']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k, title in [(axes[0], '정밀도', '정밀도 (욕설 판정 중 실제 욕설)'), (axes[1], '명시적 재현율', '명시적 욕설 문장 재현율'),
                         (axes[2], '정상문장 오탐률', '정상 문장 오탐률 (낮을수록 좋음)')]:
        ms = [g(f'탐지/{d}/{k}')[0] for d in dets]
        sd = [g(f'탐지/{d}/{k}')[1] for d in dets]
        bars(ax, ['사전', '단방향\nRNN', '양방향\nRNN', '양방향\nLSTM'], ms, sd, [BASE, GRAY, MAIN, '#2f6f9f'], ylim=(0, 1.05))
        ax.set_title(title, fontsize=12)
    fig.suptitle('Curse-detection 문장 단위 탐지 (학습에 안 쓴 실제 댓글 5,824문장)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL, 'fig_detect.png'), dpi=200)

    # ③ E5 음절 vs 자모 (실제 데이터)
    if 'E5/음절 어휘' in S:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ks = [('명시적 재현율', '명시적 욕설\n문장 재현율'), ('정밀도', '정밀도'), ('처음 보는 글자 문장 글자 재현율', '처음 보는 글자가 든\n문장의 글자 재현율')]
        w = 0.36
        for j, (unit, col) in enumerate([('음절', BASE), ('자모', MAIN)]):
            ms = [g(f'E5/{unit}/{k}')[0] for k, _ in ks]
            sd = [g(f'E5/{unit}/{k}')[1] for k, _ in ks]
            xs = [i + (j - 0.5) * w for i in range(len(ks))]
            ax.bar(xs, ms, w, yerr=sd, color=col, capsize=3, label=unit)
            for x, m in zip(xs, ms):
                ax.text(x, m + 0.03, f'{m:.2f}', ha='center', fontsize=10)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([l for _, l in ks], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False)
        ax.set_title('E5 입력 단위: 음절 vs 자모 (실제 데이터 재실험)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FINAL, 'fig_e5.png'), dpi=200)

    # ④ RNN vs LSTM 언어 모델
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = ['RNN LM', 'LSTM LM']
    bars(axes[0], ['RNN (ch05)', 'LSTM (ch06)'], [g(f'LM 퍼플렉서티/{k}')[0] for k in names],
         [g(f'LM 퍼플렉서티/{k}')[1] for k in names], [BASE, MAIN])
    axes[0].set_title('처음 보는 깨끗한 댓글 퍼플렉서티 (낮을수록 좋음)', fontsize=11)
    bars(axes[1], ['RNN (ch05)', 'LSTM (ch06)'], [g(f'학습시간/{k}')[0] / 60 for k in names],
         [g(f'학습시간/{k}')[1] / 60 for k in names], [BASE, MAIN], fmt='{:.1f}분')
    axes[1].set_title('학습 시간 (M1 CPU, NumPy)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL, 'fig_lm.png'), dpi=200)
    print('그림 저장:', ', '.join(f for f in sorted(os.listdir(FINAL)) if f.startswith('fig_')))


if __name__ == '__main__':
    main()
