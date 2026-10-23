"""Modelo do aluno simulado: probabilidade de acerto, recompensa e desistência."""

import numpy as np

from .config import Config


def prob_acerto(h, d, cfg: Config):
    """Probabilidade de um aluno de habilidade h acertar uma questão de dificuldade d (0, ..., n-1).

    Divide [0, 1] em n_dificuldades + 1 partes iguais; a questão de dificuldade d tem uma
    sigmoide centrada no (d+1)-ésimo ponto da partição. Aceita h vetorizado (para gráficos).
    """
    x0 = (d + 1) / (cfg.n_dificuldades + 1)
    return 1 / (1 + np.exp(-cfg.sigmoid_shape * (h - x0)))


class Aluno:
    def __init__(self, cfg: Config, rng: np.random.Generator, h=None):
        self.cfg = cfg
        self.h = rng.beta(*cfg.beta_ab) if h is None else h
        self.reset()

    def reset(self):
        """Prepara o aluno para uma nova sessão."""
        self.acertos = 0
        self.erros = 0
        self.ativo = True
        self.F0, self.V0 = 0, 0  # erros/acertos no momento em que a sessão atinge min_episode_length

    def responder(self, dificuldade, rng: np.random.Generator):
        """O aluno responde uma questão e decide se continua na sessão.

        Retorna (nota, recompensa), com nota ∈ {0, 1} e recompensa = nota * (dificuldade + 1).
        """
        assert self.ativo, "o aluno já saiu da sessão"
        cfg = self.cfg
        lf, lvf = cfg.lambdas

        nota = rng.binomial(1, prob_acerto(self.h, dificuldade, cfg))
        if nota == 1:
            self.acertos += 1
        else:
            self.erros += 1

        # Probabilidade de continuar. Só conta erros/acertos feitos depois de min_episode_length.
        n_feitas = self.acertos + self.erros
        if n_feitas < cfg.min_episode_length:
            ps = 1.0
        else:
            if n_feitas == cfg.min_episode_length:
                self.F0, self.V0 = self.erros, self.acertos
            F = self.erros - self.F0
            V = self.acertos - self.V0
            ps = min(np.exp(-lf * F), np.exp(-lvf * (V + F)))

        if n_feitas == cfg.max_episode_length:
            ps = 0.0

        if rng.binomial(1, ps) == 0:
            self.ativo = False

        recompensa = nota * (dificuldade + 1)
        return nota, recompensa
