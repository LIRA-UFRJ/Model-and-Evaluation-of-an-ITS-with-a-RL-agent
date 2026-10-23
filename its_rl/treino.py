"""Controle Monte Carlo on-policy, every-visit, para políticas ε-soft (Sutton & Barto, seção 5.4).

Com ε fixo, Q converge para q_{π_ε} e a política converge para a melhor política ε-soft,
não para a política ótima π*. A política ε-gulosa é calculada a partir de Q no momento de
escolher a ação, o que equivale a atualizar π(·|S_t) logo depois de cada atualização de Q(S_t, ·).
"""

from dataclasses import dataclass, field

import numpy as np
import tqdm

from .aluno import Aluno
from .ambiente import gerar_episodio
from .config import Config
from .politicas import EpsilonDecaindo, EpsilonGulosa, TabelaQ


@dataclass
class HistoricoTreino:
    retornos: list = field(default_factory=list)        # G_0 de cada episódio
    q_referencia: list = field(default_factory=list)    # Q(estado_referencia, ·) após cada episódio
    epsilons: list = field(default_factory=list)        # ε usado em cada episódio (só no GLIE)


def mc_on_policy(cfg: Config, habilidades, sessoes_por_aluno, epsilon, rng: np.random.Generator,
                 tabela: TabelaQ = None, estado_referencia=None):
    """Treina Q com episódios de alunos de habilidade `habilidades[i]`, `sessoes_por_aluno` sessões cada.

    Retorna (tabela, historico). Se `estado_referencia` for dado, guarda Q(estado_referencia, ·)
    após cada episódio (útil para curvas de convergência).
    """
    tabela = TabelaQ(cfg) if tabela is None else tabela
    politica = EpsilonGulosa(tabela, epsilon)
    historico = HistoricoTreino()

    for h in tqdm.tqdm(habilidades, desc="Treinando", unit="aluno"):
        aluno = Aluno(cfg, rng, h=h)
        for _ in range(sessoes_por_aluno):
            ep = gerar_episodio(aluno, politica, cfg, rng)

            g = 0.0
            for t in reversed(range(len(ep))):
                g = cfg.gamma * g + ep.recompensas[t]
                tabela.atualizar_media(ep.estados[t], ep.acoes[t], g)

            historico.retornos.append(g)
            if estado_referencia is not None:
                historico.q_referencia.append(tabela.q[estado_referencia].copy())

    return tabela, historico

# Aqui vamos implementar o GLIE MC CM (Greedy in the Limit with Infinite Exploration Monte Carlo Control Method)
# A diferença desse método para o que vínhamos fazendo até então é que ele atualiza o valor
# de epsilon para reduzir a exploração ao longo do tempo
def glie_mc_on_policy(cfg: Config, habilidades, sessoes_por_aluno, epsilon_inicial, tau,
                      rng: np.random.Generator, tabela: TabelaQ = None, estado_referencia=None):
    """Igual a mc_on_policy, mas o episódio k usa ε_k = epsilon_inicial / (1 + k/tau).

    ε > 0 sempre e ε → 0. Num MDP, Q → q* e a política → π*. Aqui o estado não contém o h real,
    então a propriedade de Markov vale só de forma aproximada.
    """
    tabela = TabelaQ(cfg) if tabela is None else tabela
    politica = EpsilonDecaindo(tabela, epsilon_inicial, tau)
    historico = HistoricoTreino()

    for h in tqdm.tqdm(habilidades, desc="Treinando (GLIE)", unit="aluno"):
        aluno = Aluno(cfg, rng, h=h)
        for _ in range(sessoes_por_aluno):
            politica.novo_episodio()  # atualiza ε antes do episódio
            ep = gerar_episodio(aluno, politica, cfg, rng)

            g = 0.0
            for t in reversed(range(len(ep))):
                g = cfg.gamma * g + ep.recompensas[t]
                tabela.atualizar_media(ep.estados[t], ep.acoes[t], g)

            historico.retornos.append(g)
            historico.epsilons.append(politica.epsilon)
            if estado_referencia is not None:
                historico.q_referencia.append(tabela.q[estado_referencia].copy())

    return tabela, historico
