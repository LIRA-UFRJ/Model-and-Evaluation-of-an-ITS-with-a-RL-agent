"""Experimento do artigo: treina com Monte Carlo (ε fixo e GLIE) e compara com as baselines."""

import os

import matplotlib.pyplot as plt
import numpy as np

from its_rl import graficos
from its_rl.aluno import Aluno
from its_rl.ambiente import estado_inicial
from its_rl.avaliacao import avaliar_politicas
from its_rl.config import Config
from its_rl.politicas import baselines, gulosa
from its_rl.treino import glie_mc_on_policy, mc_on_policy

SEED = 0
PASTA_RESULTADOS = "results"

N_ALUNOS = 2000           # alunos amostrados no treino (e na avaliação)
SESSOES_POR_ALUNO = 1000  # sessões por aluno no treino (e amostras por aluno na avaliação)

EPSILON = 0.7             # EpsilonGulosa: ε fixo usado em todo o treino
EPSILON_INICIAL = 1.0     # EpsilonDecaindo (GLIE): ε do primeiro episódio
TAU = 20_000              # EpsilonDecaindo (GLIE): ε_k = EPSILON_INICIAL / (1 + k/TAU); cai pela metade após TAU episódios

K_VALORES = [0, 2, 4]     # o experimento roda uma vez para cada k (faixas de habilidade no estado)


def experimento_principal(cfg: Config, rng):
    print(f"\n========== k = {cfg.k} ==========")
    pasta =os.path.join(PASTA_RESULTADOS, f"k{cfg.k}_eps{EPSILON}_glie_eps{EPSILON_INICIAL}_tau{TAU}")
    os.makedirs(pasta, exist_ok=True)

    # Treino: h ~ Uniforme(0, 1), os mesmos alunos para os dois algoritmos
    habilidades = rng.uniform(0, 1, size=N_ALUNOS)

    tabela_fixo, _ = mc_on_policy(cfg, habilidades, SESSOES_POR_ALUNO, EPSILON, rng)
    tabela_fixo.salvar(os.path.join(pasta, "tabela_q_eps_fixo.npz"))

    tabela_glie, historico_glie = glie_mc_on_policy(cfg, habilidades, SESSOES_POR_ALUNO, EPSILON_INICIAL, TAU, rng)
    tabela_glie.salvar(os.path.join(pasta, "tabela_q_glie.npz"))
    print(f"GLIE: ε no último episódio de treino = {historico_glie.epsilons[-1]:.4g}")

    # Avaliação: cada política aprendida é avaliada como gulosa (ε = 0) em relação ao seu Q.
    # O nome diz com qual política ela foi treinada.
    politicas = baselines(cfg)
    politicas["EpsilonGulosa"] = gulosa(tabela_fixo)
    politicas["EpsilonDecaindo"] = gulosa(tabela_glie)
    h_range = np.linspace(0, 1, N_ALUNOS)
    res = avaliar_politicas(politicas, cfg, h_range, SESSOES_POR_ALUNO, rng)
    res.salvar_json(os.path.join(pasta, f"results_min{cfg.min_episode_length}.json"))

    print(graficos.tabela_latex(res))
    figuras = graficos.plot_resultados(res, principais=["Simple", "Random", "EpsilonGulosa", "EpsilonDecaindo"],
                                       fixas=["Easy", "Medium", "Hard"], pasta=pasta)
    for fig in figuras.values():  # já foram salvas; fechar evita acumular janelas entre os valores de k
        plt.close(fig)


def curva_aprendizado(cfg: Config, rng, n_alunos=10000, sessoes_por_aluno=1):
    """Quantos episódios são necessários para Q do estado inicial estabilizar (alunos h ~ Beta). ε fixo."""
    habilidades = [Aluno(cfg, rng).h for _ in range(n_alunos)]
    s0 = estado_inicial(0.5, cfg)
    _, historico = mc_on_policy(cfg, habilidades, sessoes_por_aluno, EPSILON, rng, estado_referencia=s0)

    fig_q, fig_g = graficos.plot_curva_aprendizado(historico)
    pasta = os.path.join(PASTA_RESULTADOS, "curva_aprendizado")
    sufixo = f"k={cfg.k}_eps={EPSILON}_alunos={n_alunos}_sessoes={sessoes_por_aluno}"
    graficos.salvar_figura(fig_q, pasta, f"q_valor_{sufixo}")
    graficos.salvar_figura(fig_g, pasta, f"retorno_{sufixo}")


if __name__ == "__main__":
    for k in K_VALORES:
        # Gerador novo com a mesma SEED: cada k usa os mesmos alunos e pode ser reproduzido sozinho
        rng = np.random.default_rng(SEED)
        experimento_principal(Config(k=k), rng)

    cfg = Config(k=0)
    # curva_aprendizado(cfg, np.random.default_rng(SEED))

    graficos.plot_sigmoides(cfg)
    graficos.plot_exponenciais(cfg)
    plt.show()
