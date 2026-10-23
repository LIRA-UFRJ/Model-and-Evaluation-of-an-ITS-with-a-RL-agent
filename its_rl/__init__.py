"""Sistema tutor inteligente simulado + controle Monte Carlo para escolher a dificuldade das questões.

Módulos:
    config     parâmetros do ambiente
    aluno      modelo do aluno (acerto, recompensa, desistência)
    ambiente   estado do MDP e geração de episódios
    politicas  tabela Q, política ε-gulosa e baselines
    treino     controle Monte Carlo on-policy
    avaliacao  retorno e duração médios por habilidade
    graficos   figuras e tabela LaTeX
"""
