# ITS + RL

[Português](#português) · [English](#english)

---

## Português

Simulação de um sistema tutor inteligente (ITS) que escolhe a dificuldade da próxima questão para um aluno simulado. A política é aprendida com **controle Monte Carlo on-policy (every-visit) para políticas ε-soft** (Sutton & Barto, seção 5.4) e comparada com políticas de referência (baselines).

### O modelo

- **Aluno:** tem uma habilidade `h ∈ [0, 1]`. A probabilidade de acertar uma questão de dificuldade `d ∈ {0, 1, 2}` é uma sigmoide em `h` centrada em `(d+1)/4` (0,25, 0,5 e 0,75).
- **Recompensa:** `nota × (d + 1)`. Acertar uma questão difícil vale mais.
- **Desistência:** depois de cada questão, o aluno continua com probabilidade `min(exp(-λ_f · F), exp(-λ_vf · (V + F)))`, em que `F` = erros e `V` = acertos. Ele sai obrigatoriamente após `max_episode_length` questões.
- **Estado:** `(acertos_0, feitas_0, acertos_1, feitas_1, acertos_2, feitas_2)`, com a faixa de habilidade do aluno no final quando `k > 0`.
- **Ação:** a dificuldade da próxima questão.

### Algoritmo e limitações

O `main.py` treina duas tabelas Q com os mesmos alunos:

- **`mc_on_policy`**, com ε **fixo**. Converge para a **melhor política ε-soft**, não para a política ótima π\*.
- **`glie_mc_on_policy`**, com ε **decaindo** entre episódios: ε_k = ε₀ / (1 + k/τ). É o GLIE Monte Carlo Control (*Greedy in the Limit with Infinite Exploration*). Como ε > 0 sempre e ε → 0, num MDP Q → q\* e a política → π\*. A prova formal de GLIE (Singh et al., 2000) usa ε decaindo **por estado**. Com o decaimento por episódio usado aqui, estados que exigem várias ações exploratórias seguidas podem deixar de ser explorados, então a garantia vale de forma aproximada.

Nos gráficos, **"EpsilonGulosa"** e **"EpsilonDecaindo"** são as políticas **gulosas (ε = 0) em relação ao Q** aprendido por cada algoritmo. O nome indica como cada uma foi treinada, não como ela age na avaliação.
- O estado não contém a habilidade real `h` (no máximo uma faixa dela). Por isso, o problema é **parcialmente observável**: a propriedade de Markov vale só de forma aproximada.

### Estrutura

| Arquivo | Conteúdo |
|---|---|
| `main.py` | Experimento do artigo. Os parâmetros principais ficam no topo do arquivo. |
| `its_rl/config.py` | `Config`: todos os parâmetros do ambiente (λ, tamanho dos episódios, `k`, γ, …). |
| `its_rl/aluno.py` | Modelo do aluno: probabilidade de acerto, recompensa e desistência. |
| `its_rl/ambiente.py` | Estado do MDP e geração de episódios (`gerar_episodio`). |
| `its_rl/politicas.py` | `TabelaQ`, política ε-gulosa e baselines (`Simple`, `Random`, `Easy`, `Medium`, `Hard`). |
| `its_rl/treino.py` | `mc_on_policy` (ε fixo) e `glie_mc_on_policy` (GLIE): o treinamento. |
| `its_rl/avaliacao.py` | `avaliar_politicas`: retorno e duração médios por habilidade. |
| `its_rl/graficos.py` | Figuras e tabela LaTeX. |
| `código antigo/RL_OFFICIAL.py` | Versão original, antes da refatoração. Só para referência. |

**Ordem de leitura sugerida:** `config` → `aluno` → `ambiente` → `politicas` → `treino` → `avaliacao` → `main`.

### Instalação

Requer Python 3.9 ou mais recente. Os comandos abaixo criam um ambiente virtual na pasta do projeto. Veja [Por que usar um ambiente virtual](#por-que-usar-um-ambiente-virtual).

```bash
cd "caminho/para/Artigo ITS + RL"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`source .venv/bin/activate` precisa ser executado **a cada terminal novo**. Quando o ambiente está ativo, aparece `(.venv)` no início da linha. No VS Code, selecione o interpretador em `Cmd/Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

### Como rodar

```bash
python main.py
```

O script executa, em ordem:

1. **Treino:** `N_ALUNOS` alunos com `h ~ Uniforme(0, 1)`, `SESSOES_POR_ALUNO` sessões cada, primeiro com ε fixo e depois com GLIE.
2. **Avaliação:** todas as políticas, para `N_ALUNOS` valores de `h` igualmente espaçados em [0, 1], com `SESSOES_POR_ALUNO` sessões cada.
3. **Resultados:** a tabela LaTeX é impressa no terminal, os arquivos são salvos em `results/<execução>/` e as figuras são abertas. A janela de figuras **bloqueia** o fim do script até ser fechada; para rodar sem janelas, use `MPLBACKEND=Agg python main.py`.

**Tempo de execução.** Com os valores padrão (`N_ALUNOS = 2000`, `SESSOES_POR_ALUNO = 1000`), são 4 milhões de episódios de treino (2 algoritmos) e 14 milhões de avaliação (7 políticas), o que leva **cerca de 10 a 15 minutos por valor de `k`** num MacBook (≈ 30 a 45 minutos para `K_VALORES = [0, 2, 4]`). Com `k = 4`, o pico de memória é ≈ 2 GB. O tempo cresce linearmente com `N_ALUNOS` e com `SESSOES_POR_ALUNO`. Para um teste rápido, reduza os valores no topo do `main.py`, por exemplo para `N_ALUNOS = 20` e `SESSOES_POR_ALUNO = 50`.

### Parâmetros

No topo de `main.py`:

| Nome | Padrão | Significado |
|---|---|---|
| `SEED` | 0 | Semente aleatória. A mesma semente reproduz exatamente o mesmo resultado. |
| `N_ALUNOS` | 2000 | Alunos no treino e pontos de `h` na avaliação. |
| `SESSOES_POR_ALUNO` | 1000 | Sessões por aluno no treino e na avaliação. |
| `K_VALORES` | [0, 2, 4] | O experimento inteiro roda uma vez para cada `k`, cada um com um gerador novo criado com `SEED` (mesmos alunos em todos). |
| `EPSILON` | 0,7 | ε fixo (`mc_on_policy`). |
| `EPSILON_INICIAL` | 1,0 | GLIE: ε₀, o ε do primeiro episódio. |
| `TAU` | 20 000 | GLIE: τ em ε_k = ε₀ / (1 + k/τ). Depois de τ episódios, ε cai pela metade. Com 2 milhões de episódios, o ε final é ≈ 0,01. |
| `PASTA_RESULTADOS` | `results` | Pasta onde os resultados são salvos. |

No `Config` (`its_rl/config.py`), passados em `Config(...)` no `main.py`:

| Nome | Padrão | Significado |
|---|---|---|
| `k` | 0 | Número de faixas de habilidade no estado (0 = o estado não contém a habilidade). No `main.py`, vem de `K_VALORES`. |
| `lambdas` | (0,05, 0,025) | `(λ_f, λ_vf)` da probabilidade de continuar. |
| `max_episode_length` | 12 | Número máximo de questões por sessão. |
| `min_episode_length` | 0 | Antes disso o aluno não sai. Na prática, o episódio tem no mínimo `min_episode_length + 1` questões. |
| `gamma` | 1,0 | Fator de desconto. |
| `sigmoid_shape` | 15 | Inclinação das sigmoides de acerto. |
| `beta_ab` | (100, 110) | Distribuição Beta de `h` para alunos sorteados sem `h` (usada só em `curva_aprendizado`). |

### Saídas (`results/<execução>/`)

Cada execução vai para uma subpasta com o nome da configuração, por exemplo `results/k0_eps0.7_glie_eps1.0_tau20000/`. Assim, execuções com parâmetros diferentes não se sobrescrevem.

| Arquivo | Conteúdo |
|---|---|
| `tabela_q_eps_fixo.npz`, `tabela_q_glie.npz` | Arrays `q` (Q(s, a)) e `n` (visitas a cada (s, a)) de cada algoritmo. Podem ser recarregados com `TabelaQ.carregar`. |
| `results_min{min}.json` | Retorno e duração (média e desvio-padrão) por política e por `h`, além do total de questões de cada dificuldade. |
| `retorno_*.png`, `duracao_*.png` | Retorno médio e duração média por habilidade. |
| `distribuicao_questoes.png` | Frequência de cada dificuldade por política. |

Cada figura é salva duas vezes: com fundo branco e com fundo transparente (`*_transparente.png`).

### Curva de aprendizado (opcional)

Descomente `curva_aprendizado(cfg, rng)` no `main.py`. A função mostra como Q no estado inicial e o retorno por episódio evoluem durante o treino, com alunos sorteados de `Beta(100, 110)`. Usa ε fixo. Os gráficos são salvos em `results/curva_aprendizado/`.

### Por que usar um ambiente virtual

Um **ambiente virtual** (`venv`) é uma pasta (`.venv/`) com uma cópia isolada do Python e das bibliotecas do projeto. **Não é uma máquina virtual**: não emula outro computador, só separa as bibliotecas.

- Cada projeto tem suas próprias bibliotecas e versões, sem conflito com outros projetos nem com o Python do sistema.
- No macOS, o Python do sistema pode bloquear ou desaconselhar `pip install` global. O venv evita esse problema.
- `requirements.txt` + `venv` permitem que qualquer pessoa recrie o mesmo ambiente com dois comandos.
- Para remover, basta apagar a pasta `.venv/`. Ela está no `.gitignore` e não vai para o repositório.

---

## English

Simulation of an intelligent tutoring system (ITS) that chooses the difficulty of the next question for a simulated student. The policy is learned with **on-policy every-visit Monte Carlo control for ε-soft policies** (Sutton & Barto, section 5.4) and compared against baseline policies.

### Model

- **Student:** has an ability `h ∈ [0, 1]`. The probability of answering a question of difficulty `d ∈ {0, 1, 2}` correctly is a sigmoid in `h` centred at `(d+1)/4` (0.25, 0.5, 0.75).
- **Reward:** `correct × (d + 1)`. Hard questions are worth more.
- **Dropout:** after each question, the student continues with probability `min(exp(-λ_f · F), exp(-λ_vf · (V + F)))`, where `F` = wrong answers and `V` = correct answers. The student always leaves after `max_episode_length` questions.
- **State:** `(correct_0, attempted_0, correct_1, attempted_1, correct_2, attempted_2)`, followed by the student's ability band when `k > 0`.
- **Action:** the difficulty of the next question.

### Algorithm and limitations

`main.py` trains two Q tables on the same students:

- **`mc_on_policy`**, with a **fixed** ε. It converges to the **best ε-soft policy**, not to the optimal policy π\*.
- **`glie_mc_on_policy`**, with ε **decaying** between episodes: ε_k = ε₀ / (1 + k/τ). This is GLIE Monte Carlo Control (*Greedy in the Limit with Infinite Exploration*). Because ε > 0 at every step and ε → 0, in an MDP Q → q\* and the policy → π\*. The formal GLIE proof (Singh et al., 2000) uses a **per-state** decaying ε. With the per-episode decay used here, states that need several exploratory actions in a row to reach may stop being explored, so the guarantee holds only approximately.

In the figures, **"EpsilonGulosa"** and **"EpsilonDecaindo"** are the policies that are **greedy (ε = 0) with respect to the Q** learned by each algorithm. The name says how each one was trained, not how it acts during evaluation.
- The state does not contain the true ability `h` (at most a band of it). The problem is therefore **partially observable**: the Markov property holds only approximately.

### Layout

| File | Contents |
|---|---|
| `main.py` | The paper's experiment. The main parameters are at the top of the file. |
| `its_rl/config.py` | `Config`: every environment parameter (λ, episode length, `k`, γ, …). |
| `its_rl/aluno.py` | Student model: probability of a correct answer, reward and dropout. |
| `its_rl/ambiente.py` | MDP state and episode generation (`gerar_episodio`). |
| `its_rl/politicas.py` | `TabelaQ`, the ε-greedy policy and the baselines (`Simple`, `Random`, `Easy`, `Medium`, `Hard`). |
| `its_rl/treino.py` | `mc_on_policy` (fixed ε) and `glie_mc_on_policy` (GLIE): training. |
| `its_rl/avaliacao.py` | `avaliar_politicas`: mean return and mean session length per ability. |
| `its_rl/graficos.py` | Figures and the LaTeX table. |
| `código antigo/RL_OFFICIAL.py` | Original pre-refactor version. Kept for reference only. |

Identifiers and comments in the code are in Portuguese.

**Suggested reading order:** `config` → `aluno` → `ambiente` → `politicas` → `treino` → `avaliacao` → `main`.

### Installation

Requires Python 3.9 or later. The commands below create a virtual environment inside the project folder. See [Why a virtual environment](#why-a-virtual-environment).

```bash
cd "path/to/Artigo ITS + RL"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run `source .venv/bin/activate` **in every new terminal**. When the environment is active, the prompt starts with `(.venv)`. In VS Code, pick the interpreter with `Cmd/Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

### Running

```bash
python main.py
```

The script runs these steps in order:

1. **Training:** `N_ALUNOS` students with `h ~ Uniform(0, 1)`, `SESSOES_POR_ALUNO` sessions each, first with a fixed ε and then with GLIE.
2. **Evaluation:** every policy, for `N_ALUNOS` evenly spaced values of `h` in [0, 1], with `SESSOES_POR_ALUNO` sessions each.
3. **Results:** the LaTeX table is printed to the terminal, files are saved to `results/<run>/` and the figures are shown. The figure windows **block** the end of the script until they are closed; to run without windows, use `MPLBACKEND=Agg python main.py`.

**Runtime.** With the default values (`N_ALUNOS = 2000`, `SESSOES_POR_ALUNO = 1000`), the run has 4 million training episodes (2 algorithms) and 14 million evaluation episodes (7 policies), which takes **about 10 to 15 minutes per value of `k`** on a MacBook (≈ 30 to 45 minutes for `K_VALORES = [0, 2, 4]`). With `k = 4`, peak memory is ≈ 2 GB. Runtime grows linearly with `N_ALUNOS` and `SESSOES_POR_ALUNO`. For a quick check, lower the values at the top of `main.py`, for example to `N_ALUNOS = 20` and `SESSOES_POR_ALUNO = 50`.

### Parameters

At the top of `main.py`:

| Name | Default | Meaning |
|---|---|---|
| `SEED` | 0 | Random seed. The same seed reproduces exactly the same results. |
| `N_ALUNOS` | 2000 | Number of students in training and of `h` points in evaluation. |
| `SESSOES_POR_ALUNO` | 1000 | Sessions per student in training and in evaluation. |
| `K_VALORES` | [0, 2, 4] | The whole experiment runs once per `k`, each with a fresh generator seeded with `SEED` (same students for every `k`). |
| `EPSILON` | 0.7 | Fixed ε (`mc_on_policy`). |
| `EPSILON_INICIAL` | 1.0 | GLIE: ε₀, the ε of the first episode. |
| `TAU` | 20,000 | GLIE: τ in ε_k = ε₀ / (1 + k/τ). After τ episodes, ε has halved. With 2 million episodes, the final ε is ≈ 0.01. |
| `PASTA_RESULTADOS` | `results` | Output folder. |

In `Config` (`its_rl/config.py`), passed as `Config(...)` in `main.py`:

| Name | Default | Meaning |
|---|---|---|
| `k` | 0 | Number of ability bands in the state (0 = the state does not include ability). In `main.py`, it comes from `K_VALORES`. |
| `lambdas` | (0.05, 0.025) | `(λ_f, λ_vf)` of the continuation probability. |
| `max_episode_length` | 12 | Maximum number of questions per session. |
| `min_episode_length` | 0 | The student cannot leave before this. In practice, an episode has at least `min_episode_length + 1` questions. |
| `gamma` | 1.0 | Discount factor. |
| `sigmoid_shape` | 15 | Steepness of the correct-answer sigmoids. |
| `beta_ab` | (100, 110) | Beta distribution of `h` for students sampled without a given `h` (used only by `curva_aprendizado`). |

### Outputs (`results/<run>/`)

Each run goes into a subfolder named after its configuration, for example `results/k0_eps0.7_glie_eps1.0_tau20000/`, so runs with different parameters do not overwrite each other.

| File | Contents |
|---|---|
| `tabela_q_eps_fixo.npz`, `tabela_q_glie.npz` | Arrays `q` (Q(s, a)) and `n` (visit counts for each (s, a)) for each algorithm. They can be reloaded with `TabelaQ.carregar`. |
| `results_min{min}.json` | Return and session length (mean and standard deviation) per policy and per `h`, plus the total number of questions of each difficulty. |
| `retorno_*.png`, `duracao_*.png` | Mean return and mean session length per ability. |
| `distribuicao_questoes.png` | Frequency of each difficulty per policy. |

Each figure is saved twice: with a white background and with a transparent one (`*_transparente.png`).

### Learning curve (optional)

Uncomment `curva_aprendizado(cfg, rng)` in `main.py`. It shows how Q at the initial state and the per-episode return evolve during training, with students sampled from `Beta(100, 110)`. It uses a fixed ε. The plots are saved to `results/curva_aprendizado/`.

### Why a virtual environment

A **virtual environment** (`venv`) is a folder (`.venv/`) with an isolated copy of Python and the project's libraries. **It is not a virtual machine**: it does not emulate another computer, it only keeps the libraries separate.

- Each project gets its own libraries and versions, with no conflicts with other projects or with the system Python.
- On macOS, the system Python may block or discourage global `pip install`. A venv avoids that.
- `requirements.txt` + `venv` let anyone recreate the same environment with two commands.
- To remove it, delete the `.venv/` folder. It is listed in `.gitignore` and is not committed.
