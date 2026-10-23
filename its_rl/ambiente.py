"""Estado do MDP e geração de episódios (sessões).

Estado: (acertos_0, feitas_0, acertos_1, feitas_1, ..., acertos_{n-1}, feitas_{n-1}[, tipo_do_aluno]),
onde o índice final só existe quando cfg.k > 0.
"""

from dataclasses import dataclass, field

import numpy as np

from .aluno import Aluno
from .config import Config


@dataclass
class Episodio:
    estados: list = field(default_factory=list)      # S_0, ..., S_{T-1}
    acoes: list = field(default_factory=list)        # A_0, ..., A_{T-1}
    recompensas: list = field(default_factory=list)  # R_1, ..., R_T
    notas: list = field(default_factory=list)        # acertou (1) ou errou (0) cada questão
    estado_final: tuple = None                       # S_T

    def __len__(self):
        return len(self.acoes)

    def retorno(self, gamma=1.0):
        g = 0.0
        for r in reversed(self.recompensas):
            g = gamma * g + r
        return g


def tipo_do_aluno(h, k):
    """Faixa de habilidade do aluno (0, ..., k-1)."""
    return min(int(h * k), k - 1)


def estado_inicial(h, cfg: Config):
    estado = (0,) * (2 * cfg.n_dificuldades)
    if cfg.k > 0:
        estado += (tipo_do_aluno(h, cfg.k),)
    return estado


def gerar_episodio(aluno: Aluno, politica, cfg: Config, rng: np.random.Generator) -> Episodio:
    """Simula uma sessão completa do aluno seguindo `politica`.

    `politica(estado, episodio_ate_agora, rng)` devolve a dificuldade da próxima questão.
    """
    aluno.reset()
    estado = list(estado_inicial(aluno.h, cfg))
    ep = Episodio()

    while aluno.ativo:
        s = tuple(estado)
        a = politica(s, ep, rng)
        nota, r = aluno.responder(a, rng)

        ep.estados.append(s)
        ep.acoes.append(a)
        ep.recompensas.append(r)
        ep.notas.append(nota)

        estado[2 * a + 1] += 1  # questões feitas da dificuldade a
        if nota == 1:
            estado[2 * a] += 1  # questões acertadas da dificuldade a

    ep.estado_final = tuple(estado)
    return ep
