"""Estimação do retorno médio e da duração média das sessões de cada política, por habilidade h."""

import json
from dataclasses import dataclass

import numpy as np
import tqdm

from .aluno import Aluno
from .ambiente import gerar_episodio
from .config import Config


@dataclass
class ResultadoAvaliacao:
    nomes: list           # nome de cada política
    h_range: np.ndarray   # (n_alunos,)
    G_media: np.ndarray   # (n_politicas, n_alunos) retorno médio por aluno
    G_std: np.ndarray
    T_media: np.ndarray   # (n_politicas, n_alunos) duração média da sessão por aluno
    T_std: np.ndarray
    contagem_questoes: np.ndarray  # (n_politicas, n_dificuldades) total de questões oferecidas

    def indice(self, nome):
        return self.nomes.index(nome)

    def salvar_json(self, caminho):
        dados = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        with open(caminho, "w") as f:
            json.dump(dados, f)

    @classmethod
    def carregar_json(cls, caminho):
        with open(caminho) as f:
            dados = json.load(f)
        return cls(**{k: (v if k == "nomes" else np.array(v)) for k, v in dados.items()})


def avaliar_politicas(politicas: dict, cfg: Config, h_range, n_amostras, rng: np.random.Generator):
    """Para cada política e cada h em `h_range`, simula `n_amostras` sessões."""
    n_pol, n_h = len(politicas), len(h_range)
    G_media, G_std = np.zeros((n_pol, n_h)), np.zeros((n_pol, n_h))
    T_media, T_std = np.zeros((n_pol, n_h)), np.zeros((n_pol, n_h))
    contagem = np.zeros((n_pol, cfg.n_dificuldades), dtype=np.int64)

    for j, (nome, politica) in enumerate(politicas.items()):
        for i, h in enumerate(tqdm.tqdm(h_range, desc=f"Avaliando {nome}", unit="aluno")):
            aluno = Aluno(cfg, rng, h=h)
            retornos = np.zeros(n_amostras)
            duracoes = np.zeros(n_amostras)
            for n in range(n_amostras):
                ep = gerar_episodio(aluno, politica, cfg, rng)
                retornos[n] = ep.retorno(cfg.gamma)
                duracoes[n] = len(ep)
                contagem[j] += np.bincount(ep.acoes, minlength=cfg.n_dificuldades)
            G_media[j, i], G_std[j, i] = retornos.mean(), retornos.std()
            T_media[j, i], T_std[j, i] = duracoes.mean(), duracoes.std()

    return ResultadoAvaliacao(list(politicas), np.asarray(h_range), G_media, G_std, T_media, T_std, contagem)
