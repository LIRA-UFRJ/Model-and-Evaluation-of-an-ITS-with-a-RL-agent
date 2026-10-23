"""Gráficos e tabelas. Cada função devolve a figura; `salvar_figura` grava fundo branco e transparente."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .aluno import prob_acerto
from .avaliacao import ResultadoAvaliacao
from .config import Config

CORES = {
    "Simple": "#6baed6",
    "Random": "#969696",
    "EpsilonGulosa": "#08306b",
    "EpsilonDecaindo": "#6a3d9a",
    "Easy": "#31a354",
    "Medium": "#fd8d3c",
    "Hard": "#de2d26",
}
CORES_DIFICULDADE = [CORES["Easy"], CORES["Medium"], CORES["Hard"]]


def salvar_figura(fig, pasta, nome):
    os.makedirs(pasta, exist_ok=True)
    fig.savefig(os.path.join(pasta, f"{nome}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(pasta, f"{nome}_transparente.png"), dpi=300, bbox_inches="tight", transparent=True)


def tabela_latex(res: ResultadoAvaliacao):
    df = pd.DataFrame({
        "Policy": res.nomes,
        "Unconditional Mean Return": res.G_media.mean(axis=1),
        "Unconditional Mean Duration": res.T_media.mean(axis=1),
    })
    return df.to_latex(index=False, float_format="%.2f", caption="Policy Performance Comparison",
                       label="tab:policy_performance", column_format="lrr")


def _cor(nome):
    return CORES.get(nome)


def plot_por_h(res: ResultadoAvaliacao, medida, nomes, titulo, destaque=None, scatter=False):
    """Uma curva por política: x = habilidade h, y = média de `medida` ("G" ou "T").

    Políticas fora de `destaque` (se dado) aparecem apagadas.
    """
    valores = res.G_media if medida == "G" else res.T_media
    fig, ax = plt.subplots(figsize=(6, 6))
    for nome in nomes:
        alpha = 1 if destaque is None or nome in destaque else 0.3
        y = valores[res.indice(nome)]
        ax.plot(res.h_range, y, label=nome, color=_cor(nome), alpha=alpha)
        if scatter:
            ax.scatter(res.h_range, y, color=_cor(nome), alpha=alpha)
    ax.set_title(titulo)
    ax.set_xlabel("Ability Level")
    ax.set_ylabel("Average G values" if medida == "G" else "Average Episode Duration")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_distribuicao_questoes(res: ResultadoAvaliacao, nomes):
    """Frequência relativa (%) de cada dificuldade oferecida por cada política."""
    contagem = np.array([res.contagem_questoes[res.indice(n)] for n in nomes], dtype=float)
    percentual = 100 * contagem / contagem.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    largura = 0.2
    x = np.arange(len(nomes))
    for d in range(percentual.shape[1]):
        ax.bar(x + d * largura, percentual[:, d], largura, label=f"Question Difficulty {d + 1}",
               color=CORES_DIFICULDADE[d % len(CORES_DIFICULDADE)])
    ax.set_title("Amount of Questions by Type and Policy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Relative Frequency of Questions")
    ax.set_xticks(x + largura)
    ax.set_xticklabels(nomes)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_resultados(res: ResultadoAvaliacao, principais, fixas, pasta=None):
    """Os seis gráficos do artigo. `principais` = ex. Simple/Random/EpsilonGulosa, `fixas` = Easy/Medium/Hard."""
    figuras = {
        "retorno_principais": plot_por_h(res, "G", principais, "Average G per h"),
        "retorno_fixas": plot_por_h(res, "G", fixas, "Average G per h"),
        "retorno_todas": plot_por_h(res, "G", principais + fixas, "Average G per h", destaque=principais),
        "duracao_principais": plot_por_h(res, "T", principais, "Average Episode Duration per h", scatter=True),
        "duracao_fixas": plot_por_h(res, "T", fixas, "Average Episode Duration per h", scatter=True),
        "distribuicao_questoes": plot_distribuicao_questoes(res, principais),
    }
    if pasta is not None:
        for nome, fig in figuras.items():
            salvar_figura(fig, pasta, nome)
    return figuras


def plot_curva_aprendizado(historico, meia_vida=100):
    """Q(estado_referencia, ·) e retorno por episódio ao longo do treino, com média móvel exponencial."""
    q = np.array(historico.q_referencia)
    x = np.arange(len(q))

    fig_q, ax = plt.subplots(figsize=(6, 6))
    for d in range(q.shape[1]):
        cor = CORES_DIFICULDADE[d % len(CORES_DIFICULDADE)]
        ax.plot(x, q[:, d], label=f"Question Difficulty {d + 1}", linewidth=1.5, color=cor)
        ema = pd.Series(q[:, d]).ewm(halflife=meia_vida, adjust=False).mean()
        ax.plot(x, ema, linestyle=":", linewidth=1, color=cor, alpha=0.55,
                label=f"EMA(Half-life={meia_vida}) Question Difficulty {d + 1}")
    ax.legend(ncol=3, fontsize="x-small", loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_yscale("log")
    ax.set_xlabel("Amount of student sessions")
    ax.set_ylabel("Q-value component (log scale)")
    fig_q.tight_layout()

    fig_g, ax = plt.subplots(figsize=(6, 6))
    ema = pd.Series(historico.retornos).ewm(halflife=meia_vida, adjust=False).mean()
    ax.plot(np.arange(len(ema)), ema, linestyle=":", linewidth=1, label=f"EMA(Half-life={meia_vida})")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Amount of student sessions")
    ax.set_ylabel("Return of the session (log scale)")
    fig_g.tight_layout()

    return fig_q, fig_g


def plot_sigmoides(cfg: Config):
    """Probabilidade de acerto P_d(h) de cada dificuldade."""
    h = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots()
    for d in range(cfg.n_dificuldades):
        ax.plot(h, prob_acerto(h, d, cfg), label=f"Dificuldade {d}")
        ax.axvline(x=(d + 1) / (cfg.n_dificuldades + 1), alpha=0.2, color="black")
    ax.set_xlabel("h")
    ax.set_ylabel("$P_d$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_exponenciais(cfg: Config):
    """Os dois termos da probabilidade de continuar na sessão: exp(-lf F) e exp(-lvf t)."""
    x = np.linspace(0, 20, 200)
    exp1 = np.exp(-cfg.lambdas[0] * x)
    exp2 = np.exp(-cfg.lambdas[1] * x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x, exp1, "o-", markersize=8, markevery=10)
    ax1.set_xlabel("F")
    ax1.set_ylabel(r"exp($-\lambda_1 F$)")
    ax2.plot(x, exp2, "o-", markersize=8, markevery=10)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"exp($-\lambda_2 t$)")

    margem = 0.03 * (exp2.max() - exp2.min())
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.set_ylim(exp2.min() - margem, exp2.max() + margem)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    return fig
