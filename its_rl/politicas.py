"""Políticas: tabela Q, política ε-gulosa derivada de Q e políticas baseline.

Toda política é chamável como `politica(estado, episodio_ate_agora, rng) -> acao`.
"""

import numpy as np

from .config import Config


class TabelaQ:
    """Estimativas Q(s, a) e quantidade de visitas N(s, a)."""

    def __init__(self, cfg: Config):
        formato = cfg.formato_estado + (cfg.n_dificuldades,)
        self.q = np.zeros(formato)
        self.n = np.zeros(formato, dtype=np.int64)

    def atualizar_media(self, s, a, g):
        """Média amostral incremental: Q <- Q + (G - Q) / N."""
        sa = s + (a,)
        self.n[sa] += 1
        self.q[sa] += (g - self.q[sa]) / self.n[sa]

    def melhores_acoes(self, s):
        """Todas as ações que empatam no máximo de Q(s, ·)."""
        valores = self.q[s]
        return np.flatnonzero(valores == valores.max())

    def salvar(self, caminho):
        np.savez_compressed(caminho, q=self.q, n=self.n)

    @classmethod
    def carregar(cls, caminho, cfg: Config):
        tabela = cls(cfg)
        dados = np.load(caminho)
        tabela.q, tabela.n = dados["q"], dados["n"]
        return tabela


class EpsilonGulosa:
    """π(a|s) = ε/|A| + (1-ε)/|melhores| se a ∈ melhores, ε/|A| caso contrário.

    Empates no máximo de Q são divididos igualmente. Com epsilon = 0 é a política gulosa.
    Estados nunca visitados têm Q = 0 em todas as ações, então a política neles é uniforme.
    """

    def __init__(self, tabela: TabelaQ, epsilon: float):
        self.tabela = tabela
        self.epsilon = epsilon

    def __call__(self, s, ep, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.tabela.q.shape[-1]))
        return int(rng.choice(self.tabela.melhores_acoes(s)))

    def probabilidades(self, s):
        """π(·|s): vetor com a probabilidade de cada ação no estado s, com o ε atual."""
        n_acoes = self.tabela.q.shape[-1]
        p = np.full(n_acoes, self.epsilon / n_acoes)
        melhores = self.tabela.melhores_acoes(s)
        p[melhores] += (1 - self.epsilon) / len(melhores)
        return p


def gulosa(tabela: TabelaQ):
    return EpsilonGulosa(tabela, epsilon=0.0)


class EpsilonDecaindo(EpsilonGulosa):
    """ε-gulosa com ε decaindo entre episódios (GLIE): ε_k = ε₀ / (1 + k/τ), k = 0, 1, 2, ...

    ε_k > 0 para todo k (todo par (s, a) continua sendo explorado) e ε_k → 0 (a política fica
    gulosa no limite). τ controla a velocidade: depois de τ episódios, ε cai pela metade.
    A escolha da ação e as probabilidades são herdadas da EpsilonGulosa; só muda de onde vem o ε.
    """

    def __init__(self, tabela: TabelaQ, epsilon_inicial: float, tau: float):
        super().__init__(tabela, epsilon_inicial)
        self.epsilon_inicial = epsilon_inicial
        self.tau = tau
        self.k = 0  # episódios já iniciados

    def novo_episodio(self):
        self.epsilon = self.epsilon_inicial / (1 + self.k / self.tau)
        self.k += 1


# ----------------------------------------------------------------------------- baselines

class Aleatoria:
    def __init__(self, n_dificuldades):
        self.n_dificuldades = n_dificuldades

    def __call__(self, s, ep, rng):
        return int(rng.integers(self.n_dificuldades))


class Fixa:
    """Sempre oferece a mesma dificuldade."""

    def __init__(self, dificuldade):
        self.dificuldade = dificuldade

    def __call__(self, s, ep, rng):
        return self.dificuldade


class Simples:
    """Primeira questão aleatória; depois sobe a dificuldade se acertou a última e desce se errou."""

    def __init__(self, n_dificuldades):
        self.n_dificuldades = n_dificuldades

    def __call__(self, s, ep, rng):
        if len(ep) == 0:
            return int(rng.integers(self.n_dificuldades))
        ultima = ep.acoes[-1]
        if ep.notas[-1] == 1:
            return min(ultima + 1, self.n_dificuldades - 1)
        return max(ultima - 1, 0)


def baselines(cfg: Config):
    """Políticas de comparação, com os nomes usados nos gráficos."""
    return {
        "Simple": Simples(cfg.n_dificuldades),
        "Random": Aleatoria(cfg.n_dificuldades),
        "Easy": Fixa(0),
        "Medium": Fixa(1),
        "Hard": Fixa(2),
    }
