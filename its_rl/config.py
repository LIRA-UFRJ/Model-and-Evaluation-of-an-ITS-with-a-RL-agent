"""Parâmetros do ambiente (aluno simulado + sistema de questões)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Probabilidade de continuar na sessão: Ps = min(exp(-lf * F), exp(-lvf * (V + F)))
    lambdas: tuple = (0.05, 0.025)  # (lf, lvf)

    max_episode_length: int = 12  # o aluno sai obrigatoriamente após essa quantidade de questões
    min_episode_length: int = 0   # antes disso o aluno nunca sai (Ps = 1)

    n_dificuldades: int = 3  # quantidade de níveis de dificuldade (ações)

    # Quantidade de tipos de aluno no estado. k = 0: o estado não contém a habilidade.
    # k > 0: o intervalo [0, 1] de h é dividido em k faixas e o estado contém a faixa do aluno.
    k: int = 0

    gamma: float = 1.0  # fator de desconto (episódios curtos, então damos peso igual a cada passo)

    sigmoid_shape: float = 15  # inclinação das sigmoides de probabilidade de acerto
    beta_ab: tuple = (100, 110)  # distribuição de h quando o aluno é amostrado sem h definido

    @property
    def formato_estado(self):
        """Formato do espaço de estados: (acertos_d, feitas_d) para cada dificuldade d (+ tipo do aluno se k > 0)."""
        dims = (self.max_episode_length + 1,) * (2 * self.n_dificuldades)
        if self.k > 0:
            dims += (self.k,)
        return dims
