# coding: utf-8
"""learning_curve.py 결과 → 에포크별 정밀도·재현율 그림 (saved/final/curve/fig_*.png)

fig_curve_precision_recall.png : 에포크별 정밀도 / 재현율 (탐지기 3종, 시드 평균 ± 표준편차, 사전 기준선)
fig_curve_syn_vs_real.png      : 합성 검증 F1 vs 실제 dev F1 vs 학습 손실 (오버피팅 = 둘이 벌어지는 지점)
fig_curve_pr.png               : 임계값을 바꾼 정밀도-재현율 곡선 (에포크 2 / 8 / 16)

실행:  cd project && python plot_learning_curve.py
"""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
CURVE = os.path.join(HERE, 'saved', 'final', 'curve')

for name in ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic']:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})

COLORS = {'단방향 RNN': '#9aa5ad', '양방향 RNN': '#1f7a4d', '양방향 LSTM': '#2f6f9f'}
DICT_COLOR = '#c0392b'


def stack(runs, name, key):
    """시드별 기록 → (에포크, 평균, 표준편차)"""
    arr = np.array([[row[key] for row in r['detectors'][name]] for r in runs])
    return np.arange(1, arr.shape[1] + 1), arr.mean(0), arr.std(0)


def band(ax, x, mu, sd, color, label, ls='-'):
    ax.plot(x, mu, ls, color=color, lw=2, label=label, marker='o', ms=3.5)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)


def main():
    runs = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(CURVE, 'curve_seed*.json')))]
    runs = [r for r in runs if len(r['detectors']) == 3]
    names = list(COLORS)
    n = len(runs)
    dict_p = runs[0]['dict']['precision']
    dict_r = runs[0]['dict']['recall']

    # ① 에포크별 정밀도 / 재현율
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, key, title, dval in [(axes[0], 'precision', '정밀도 (욕설이라고 한 것 중 진짜 욕설)', dict_p),
                                 (axes[1], 'recall', '재현율 (진짜 욕설 중 찾아낸 것)', dict_r)]:
        for name in names:
            x, mu, sd = stack(runs, name, key)
            band(ax, x, mu, sd, COLORS[name], name)
            best = int(np.argmax(mu))
        ax.axhline(dval, color=DICT_COLOR, ls='--', lw=1.5, label='금칙어 사전')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('에포크')
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, len(x) + 1, 2))
        ax.grid(axis='y', alpha=0.3)
    axes[1].legend(frameon=False, loc='lower right', fontsize=10)
    fig.suptitle(f'에포크별 탐지 성능 — 실제 댓글 dev {"" if n == 1 else f"(시드 {n}개 평균 ± 표준편차)"}  ·  임계값 0.5', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(CURVE, 'fig_curve_precision_recall.png'), dpi=200)

    # ② 합성 검증 vs 실제 dev vs 학습 손실
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for i, name in enumerate(names):
        ax = axes[i]
        x, syn, syn_sd = stack(runs, name, 'syn_f1')
        _, real, real_sd = stack(runs, name, 'f1')
        _, loss, _ = stack(runs, name, 'loss')
        band(ax, x, syn, syn_sd, '#7f8c8d', '합성 검증 F1 (글자)', ls='--')
        band(ax, x, real, real_sd, COLORS[name], '실제 dev F1 (문장)')
        best = int(np.argmax(real))
        ax.axvline(best + 1, color=COLORS[name], ls=':', lw=1)
        ax.text(best + 1.2, 0.22, f'실제 최고\n{best + 1}에폭', color=COLORS[name], fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(x, loss, color='#c0392b', lw=1, alpha=0.6)
        ax2.set_ylabel('학습 손실', color='#c0392b', fontsize=10)
        ax2.tick_params(axis='y', colors='#c0392b', labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(2, len(x) + 1, 2))
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('에포크')
        if i == 0:
            ax.legend(frameon=False, loc='center right', fontsize=9)
    fig.suptitle('합성 데이터 성능은 오르는데 실제 데이터 성능은? (빨간 선 = 학습 손실)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(CURVE, 'fig_curve_syn_vs_real.png'), dpi=200)

    # ③ 정밀도-재현율 곡선 (임계값 0.05~0.95), 시드 0
    r0 = runs[0]
    epochs = [e for e in (2, 8, 16) if e <= len(r0['detectors'][names[0]])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    shades = ['#c7d3dc', '#7f9bb0', None]
    for ax, name in zip(axes, names):
        for e, sh in zip(epochs, shades):
            pr = r0['detectors'][name][e - 1]['pr']
            th = sorted(pr, key=float)
            P = [pr[t][0] for t in th]
            R = [pr[t][1] for t in th]
            ax.plot(R, P, color=sh or COLORS[name], lw=2, marker='o', ms=3, label=f'{e}에폭')
            i5 = th.index('0.5')
            ax.plot(R[i5], P[i5], 'o', ms=8, mfc='none', mec='black')
        ax.plot(dict_r, dict_p, '*', ms=14, color=DICT_COLOR, label='금칙어 사전')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('재현율')
        ax.set_title(name, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc='lower left')
    axes[0].set_ylabel('정밀도')
    fig.suptitle('임계값을 0.05~0.95로 바꾼 정밀도-재현율 곡선 (실제 dev, 시드 0) · 검은 원 = 임계값 0.5', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(CURVE, 'fig_curve_pr.png'), dpi=200)

    # 요약 표
    print(f'시드 {n}개 | 사전: 정밀도 {dict_p:.3f} 재현율 {dict_r:.3f}')
    for name in names:
        _, f1, _ = stack(runs, name, 'f1')
        _, p, _ = stack(runs, name, 'precision')
        _, r, _ = stack(runs, name, 'recall')
        _, s, _ = stack(runs, name, 'syn_f1')
        b = int(np.argmax(f1))
        print(f'{name:10} 실제 F1 최고 {b + 1}에폭 ({f1[b]:.3f}, P {p[b]:.3f} R {r[b]:.3f}) | 마지막 {len(f1)}에폭 F1 {f1[-1]:.3f} '
              f'(P {p[-1]:.3f} R {r[-1]:.3f}) | 합성 F1 {s[b]:.3f} → {s[-1]:.3f}')


if __name__ == '__main__':
    main()
