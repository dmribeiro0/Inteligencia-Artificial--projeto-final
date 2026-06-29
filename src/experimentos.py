"""
experimentos.py
---------------
Comparação experimental entre Q-Learning e DQN no ambiente MDP
de cidade sustentável calibrado com dados reais (Atlas IDHM, UDH SP).

ODS 11 — Cidades e Comunidades Sustentáveis.

Experimentos realizados:
  1. Treinamento com 30 seeds diferentes (requisito do trabalho)
  2. Curvas de aprendizado comparativas (recompensa por episódio)
  3. Comparação de recompensa média e desvio-padrão
  4. Avaliação em UDHs reais do CSV (estado inicial real → melhora/piora)
  5. Distribuição de ações escolhidas por cada agente
  6. Comparação do estado final da cidade (antes × depois)
  7. Tempo de execução de cada algoritmo

Saídas geradas em data/:
  - fig_curvas_aprendizado.png
  - fig_boxplot_recompensas.png
  - fig_distribuicao_acoes.png
  - fig_estado_final.png
  - fig_avaliacao_udhs.png
  - resultados.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from ambiente   import CidadeSustentavelEnv, ACOES, INDICADORES
from qlearning  import QLearningAgent
from dqn        import DQNAgent

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
MODELOS_DIR = os.path.join(os.path.dirname(__file__), "..", "modelos")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Hiperparâmetros fixos — iguais para os dois algoritmos
NUM_EPISODIOS  = 500
NUM_SEEDS      = 30
NUM_AVALIACOES = 10   # episódios de avaliação por seed
RUIDO_AMB      = 0.05

# Hiperparâmetros Q-Learning
QL_ALPHA         = 0.1
QL_GAMMA         = 0.95
QL_EPS_INI       = 1.0
QL_EPS_FIM       = 0.05
QL_EPS_DECAY     = 0.995

# Hiperparâmetros DQN
DQN_ALPHA              = 1e-3
DQN_GAMMA              = 0.95
DQN_EPS_INI            = 1.0
DQN_EPS_FIM            = 0.05
DQN_EPS_DECAY          = 0.995
DQN_BATCH              = 64
DQN_BUFFER             = 10_000
DQN_ATUALIZAR_ALVO     = 10
DQN_DIM_OCULTA         = 64

# Cores consistentes nos gráficos
COR_QL  = "#2196F3"   # azul
COR_DQN = "#FF5722"   # laranja


# ---------------------------------------------------------------------------
# 1. Treinamento com múltiplas seeds
# ---------------------------------------------------------------------------

def treinar_multiplas_seeds(verbose: bool = True) -> dict:
    """
    Treina Q-Learning e DQN com NUM_SEEDS seeds diferentes.
    Registra recompensas por episódio, tempo de treino e recompensa de avaliação.

    Retorna
    -------
    dict com resultados completos de ambos os algoritmos.
    """
    resultados = {
        "ql":  {"recompensas_treino": [], "recompensas_aval": [], "tempos": []},
        "dqn": {"recompensas_treino": [], "recompensas_aval": [], "tempos": []},
    }

    print("=" * 65)
    print(f"  Treinamento — {NUM_SEEDS} seeds × {NUM_EPISODIOS} episódios")
    print("=" * 65)

    for seed in range(NUM_SEEDS):
        if verbose:
            print(f"\n[Seed {seed+1:>2}/{NUM_SEEDS}]")

        # Ambiente de treino — mesmo estado inicial para os dois
        env_ql  = CidadeSustentavelEnv.from_json(ruido=RUIDO_AMB, seed=seed)
        env_dqn = CidadeSustentavelEnv.from_json(ruido=RUIDO_AMB, seed=seed)

        # ── Q-Learning ──────────────────────────────────────────
        t0 = time.time()
        ql = QLearningAgent(
            alpha=QL_ALPHA, gamma=QL_GAMMA,
            epsilon_ini=QL_EPS_INI, epsilon_fim=QL_EPS_FIM,
            epsilon_decay=QL_EPS_DECAY, seed=seed,
        )
        ql.treinar(env_ql, episodios=NUM_EPISODIOS, verbose=False)
        tempo_ql = time.time() - t0

        # Avaliação Q-Learning
        aval_ql = ql.avaliar(env_ql, episodios=NUM_AVALIACOES, verbose=False)

        resultados["ql"]["recompensas_treino"].append(ql.historico["recompensas"])
        resultados["ql"]["recompensas_aval"].append(aval_ql["media"])
        resultados["ql"]["tempos"].append(tempo_ql)

        # ── DQN ─────────────────────────────────────────────────
        t0 = time.time()
        dqn = DQNAgent(
            alpha=DQN_ALPHA, gamma=DQN_GAMMA,
            epsilon_ini=DQN_EPS_INI, epsilon_fim=DQN_EPS_FIM,
            epsilon_decay=DQN_EPS_DECAY, batch_size=DQN_BATCH,
            buffer_capacidade=DQN_BUFFER,
            atualizar_alvo_a_cada=DQN_ATUALIZAR_ALVO,
            dim_oculta=DQN_DIM_OCULTA, seed=seed,
        )
        dqn.treinar(env_dqn, episodios=NUM_EPISODIOS, verbose=False)
        tempo_dqn = time.time() - t0

        # Avaliação DQN
        aval_dqn = dqn.avaliar(env_dqn, episodios=NUM_AVALIACOES, verbose=False)

        resultados["dqn"]["recompensas_treino"].append(dqn.historico["recompensas"])
        resultados["dqn"]["recompensas_aval"].append(aval_dqn["media"])
        resultados["dqn"]["tempos"].append(tempo_dqn)

        if verbose:
            print(
                f"  Q-L  | Aval: {aval_ql['media']:>8.4f} | "
                f"Tempo: {tempo_ql:>6.2f}s"
            )
            print(
                f"  DQN  | Aval: {aval_dqn['media']:>8.4f} | "
                f"Tempo: {tempo_dqn:>6.2f}s"
            )

    # Salva modelos da última seed para avaliação qualitativa
    ql.salvar(os.path.join(MODELOS_DIR, "qtable_final.pkl"))
    dqn.salvar(os.path.join(MODELOS_DIR, "dqn_final.pt"))

    # Resumo estatístico
    print("\n" + "=" * 65)
    print("  RESUMO — 30 seeds")
    print("=" * 65)
    for nome, chave in [("Q-Learning", "ql"), ("DQN", "dqn")]:
        avals  = resultados[chave]["recompensas_aval"]
        tempos = resultados[chave]["tempos"]
        print(f"\n  {nome}")
        print(f"    Recomp. avaliação : {np.mean(avals):.4f} ± {np.std(avals):.4f}")
        print(f"    Mín / Máx         : {np.min(avals):.4f} / {np.max(avals):.4f}")
        print(f"    Tempo médio/seed  : {np.mean(tempos):.2f}s")
        print(f"    Tempo total       : {np.sum(tempos):.2f}s")

    return resultados


# ---------------------------------------------------------------------------
# 2. Gráfico — Curvas de aprendizado
# ---------------------------------------------------------------------------

def plot_curvas_aprendizado(resultados: dict) -> None:
    """
    Plota a recompensa média ± desvio-padrão por episódio para cada algoritmo,
    calculada sobre as 30 execuções com seeds diferentes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for nome, chave, cor in [("Q-Learning", "ql", COR_QL), ("DQN", "dqn", COR_DQN)]:
        matriz   = np.array(resultados[chave]["recompensas_treino"])  # (30, 500)
        media    = matriz.mean(axis=0)
        std      = matriz.std(axis=0)
        episodios = np.arange(1, len(media) + 1)

        ax.plot(episodios, media, color=cor, linewidth=2, label=nome)
        ax.fill_between(episodios, media - std, media + std,
                        color=cor, alpha=0.15)

    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel("Recompensa total", fontsize=12)
    ax.set_title(
        f"Curvas de Aprendizado — Q-Learning vs DQN\n"
        f"(média ± desvio-padrão sobre {NUM_SEEDS} seeds, {NUM_EPISODIOS} episódios)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=12)
    plt.tight_layout()
    caminho = os.path.join(DATA_DIR, "fig_curvas_aprendizado.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# 3. Gráfico — Boxplot de recompensas de avaliação
# ---------------------------------------------------------------------------

def plot_boxplot_recompensas(resultados: dict) -> None:
    """
    Boxplot das recompensas médias de avaliação das 30 seeds.
    Permite comparar distribuição e variabilidade entre os algoritmos.
    """
    dados = {
        "Q-Learning": resultados["ql"]["recompensas_aval"],
        "DQN":        resultados["dqn"]["recompensas_aval"],
    }
    df_box = pd.DataFrame(dados)

    fig, ax = plt.subplots(figsize=(8, 6))
    bplot = ax.boxplot(
        [df_box["Q-Learning"], df_box["DQN"]],
        tick_labels=["Q-Learning", "DQN"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        notch=False,
    )
    bplot["boxes"][0].set_facecolor(COR_QL)
    bplot["boxes"][1].set_facecolor(COR_DQN)
    for patch in bplot["boxes"]:
        patch.set_alpha(0.7)

    ax.set_ylabel("Recompensa média de avaliação", fontsize=12)
    ax.set_title(
        f"Distribuição das Recompensas de Avaliação\n"
        f"({NUM_SEEDS} seeds, {NUM_AVALIACOES} episódios de avaliação por seed)",
        fontsize=13, fontweight="bold"
    )

    # Anota média e desvio
    for i, (nome, chave) in enumerate([("Q-Learning", "ql"), ("DQN", "dqn")], 1):
        vals = resultados[chave]["recompensas_aval"]
        ax.text(i, np.max(vals) + 0.5,
                f"μ={np.mean(vals):.2f}\nσ={np.std(vals):.2f}",
                ha="center", fontsize=10, color="black")

    plt.tight_layout()
    caminho = os.path.join(DATA_DIR, "fig_boxplot_recompensas.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# 4. Gráfico — Distribuição de ações escolhidas
# ---------------------------------------------------------------------------

def plot_distribuicao_acoes(resultados: dict) -> None:
    """
    Plota a frequência de cada ação escolhida pelos agentes treinados
    durante a avaliação (modo greedy, sem exploração).
    Detecta visualmente se há vício em uma única ação.
    """
    # Recarrega modelos finais para coletar ações em modo greedy
    from qlearning import QLearningAgent
    from dqn       import DQNAgent

    ql  = QLearningAgent.carregar(os.path.join(MODELOS_DIR, "qtable_final.pkl"))
    dqn = DQNAgent.carregar(os.path.join(MODELOS_DIR, "dqn_final.pt"))

    env = CidadeSustentavelEnv.from_json(ruido=RUIDO_AMB, seed=0)

    contagem_ql  = {a: 0 for a in ACOES}
    contagem_dqn = {a: 0 for a in ACOES}

    # Coleta ações em 30 episódios de avaliação
    N_EP = 30
    for _ in range(N_EP):
        estado = env.reset()
        done = False
        while not done:
            a_ql  = ql.agir(estado, explorar=False)
            a_dqn = dqn.agir(estado, explorar=False)
            contagem_ql[ACOES[a_ql]]   += 1
            contagem_dqn[ACOES[a_dqn]] += 1
            _, _, done, _ = env.step(a_ql)   # avança o ambiente com ql

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    acoes_curtas = [a.replace("Programa de ", "Prog. ")
                      .replace("Construção de ", "Constr. ")
                      .replace("Expandir ", "Exp. ")
                      .replace("Controle de Emissões", "Ctrl. Emissões")
                    for a in ACOES]

    for ax, contagem, nome, cor in [
        (axes[0], contagem_ql,  "Q-Learning", COR_QL),
        (axes[1], contagem_dqn, "DQN",        COR_DQN),
    ]:
        total = sum(contagem.values())
        vals  = [contagem[a] / total * 100 for a in ACOES]
        bars  = ax.barh(acoes_curtas, vals, color=cor, alpha=0.8, edgecolor="white")
        ax.set_xlabel("% de escolhas (modo greedy)", fontsize=11)
        ax.set_title(f"{nome} — Distribuição de Ações", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=9)

    plt.suptitle(
        f"Frequência de Ações em Modo Greedy ({N_EP} episódios de avaliação)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    caminho = os.path.join(DATA_DIR, "fig_distribuicao_acoes.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# 5. Gráfico — Estado final da cidade (antes × depois)
# ---------------------------------------------------------------------------

def plot_estado_final() -> None:
    """
    Compara o estado inicial médio das UDHs com o estado final
    após 20 turnos do agente treinado (Q-Learning e DQN).
    """
    from qlearning import QLearningAgent
    from dqn       import DQNAgent

    ql  = QLearningAgent.carregar(os.path.join(MODELOS_DIR, "qtable_final.pkl"))
    dqn = DQNAgent.carregar(os.path.join(MODELOS_DIR, "dqn_final.pt"))

    env = CidadeSustentavelEnv.from_json(ruido=0.0, seed=0)  # sem ruído para comparação limpa
    estado_inicial = env.estado.copy()

    # Coleta estado final médio em 30 episódios
    N = 30
    finais_ql  = {ind: [] for ind in INDICADORES}
    finais_dqn = {ind: [] for ind in INDICADORES}

    for _ in range(N):
        # Q-Learning
        estado = env.reset()
        done = False
        while not done:
            acao = ql.agir(estado, explorar=False)
            estado, _, done, _ = env.step(acao)
        for ind in INDICADORES:
            finais_ql[ind].append(env.estado[ind])

        # DQN
        estado = env.reset()
        done = False
        while not done:
            acao = dqn.agir(estado, explorar=False)
            estado, _, done, _ = env.step(acao)
        for ind in INDICADORES:
            finais_dqn[ind].append(env.estado[ind])

    # Médias finais
    media_ql  = {ind: np.mean(finais_ql[ind])  for ind in INDICADORES}
    media_dqn = {ind: np.mean(finais_dqn[ind]) for ind in INDICADORES}

    x     = np.arange(len(INDICADORES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, [estado_inicial[i] for i in INDICADORES],
           width, label="Inicial (real)", color="#9E9E9E", alpha=0.8)
    ax.bar(x,         [media_ql[i]  for i in INDICADORES],
           width, label="Q-Learning",    color=COR_QL,  alpha=0.8)
    ax.bar(x + width, [media_dqn[i] for i in INDICADORES],
           width, label="DQN",           color=COR_DQN, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(INDICADORES, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Valor do indicador (0–10)", fontsize=12)
    ax.set_ylim(0, 11)
    ax.set_title(
        "Estado da Cidade: Inicial × Após 20 Turnos\n"
        f"(média de {N} episódios, sem ruído)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.axhline(y=10, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    caminho = os.path.join(DATA_DIR, "fig_estado_final.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    print(f"Salvo: {caminho}")

    # Imprime tabela comparativa
    print(f"\n{'='*65}")
    print(f"  Estado da cidade: inicial × Q-Learning × DQN")
    print(f"{'='*65}")
    print(f"  {'Indicador':<14} {'Inicial':>8}  {'Q-Learning':>10}  {'DQN':>8}  {'ΔQL':>7}  {'ΔDQN':>7}")
    print(f"  {'─'*60}")
    for ind in INDICADORES:
        ini = estado_inicial[ind]
        ql_ = media_ql[ind]
        dqn_= media_dqn[ind]
        print(
            f"  {ind:<14} {ini:>8.2f}  {ql_:>10.2f}  {dqn_:>8.2f}  "
            f"{ql_-ini:>+7.2f}  {dqn_-ini:>+7.2f}"
        )
    print(f"{'='*65}")

    return estado_inicial, media_ql, media_dqn


# ---------------------------------------------------------------------------
# 6. Avaliação em UDHs reais
# ---------------------------------------------------------------------------

def avaliar_udhs_reais(n_udhs: int = 5) -> None:
    """
    Avalia os agentes treinados em UDHs reais do CSV.
    Mostra a melhora/piora de cada indicador para cada UDH.
    """
    from qlearning import QLearningAgent
    from dqn       import DQNAgent

    csv_path = os.path.join(DATA_DIR, "udh_normalizado.csv")
    if not os.path.exists(csv_path):
        print("⚠️  udh_normalizado.csv não encontrado — pulando avaliação em UDHs reais.")
        return

    ql  = QLearningAgent.carregar(os.path.join(MODELOS_DIR, "qtable_final.pkl"))
    dqn = DQNAgent.carregar(os.path.join(MODELOS_DIR, "dqn_final.pt"))

    df   = pd.read_csv(csv_path)
    # Seleciona UDHs do ano 2010 para evitar NaN de Mobility
    df10 = df[df["ANO"] == 2010].reset_index(drop=True)

    # Seleciona UDHs com Economy variada (baixa, média, alta)
    indices = np.linspace(0, len(df10) - 1, n_udhs, dtype=int).tolist()

    resultados_udh = []

    print(f"\n{'='*70}")
    print(f"  Avaliação em {n_udhs} UDHs reais (ano 2010)")
    print(f"{'='*70}")

    fig, axes = plt.subplots(n_udhs, 1, figsize=(14, 4 * n_udhs))
    if n_udhs == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        env_udh = CidadeSustentavelEnv.from_csv(csv_path, idx=int(df10.index[idx]),
                                                 ruido=0.0, seed=0)
        estado_ini = env_udh.estado.copy()
        nome_udh   = env_udh.nome

        # Q-Learning
        estado = env_udh.reset()
        done = False
        while not done:
            acao = ql.agir(estado, explorar=False)
            estado, _, done, _ = env_udh.step(acao)
        final_ql = env_udh.estado.copy()

        # DQN
        estado = env_udh.reset()
        done = False
        while not done:
            acao = dqn.agir(estado, explorar=False)
            estado, _, done, _ = env_udh.step(acao)
        final_dqn = env_udh.estado.copy()

        resultados_udh.append({
            "UDH":       nome_udh,
            "Inicial":   estado_ini,
            "QL_final":  final_ql,
            "DQN_final": final_dqn,
        })

        # Plot
        x     = np.arange(len(INDICADORES))
        width = 0.25
        ax.bar(x - width, [estado_ini[i] for i in INDICADORES],
               width, color="#9E9E9E", alpha=0.8, label="Inicial")
        ax.bar(x,         [final_ql[i]  for i in INDICADORES],
               width, color=COR_QL,  alpha=0.8, label="Q-Learning")
        ax.bar(x + width, [final_dqn[i] for i in INDICADORES],
               width, color=COR_DQN, alpha=0.8, label="DQN")
        ax.set_xticks(x)
        ax.set_xticklabels(INDICADORES, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 11)
        ax.set_ylabel("Valor (0–10)", fontsize=10)
        ax.set_title(f"UDH: {nome_udh}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        print(f"\n  UDH: {nome_udh}")
        print(f"  {'Indicador':<14} {'Inicial':>8}  {'QL':>8}  {'DQN':>8}  {'ΔQL':>7}  {'ΔDQN':>7}")
        print(f"  {'─'*58}")
        for ind in INDICADORES:
            ini = estado_ini[ind]
            ql_ = final_ql[ind]
            dqn_= final_dqn[ind]
            print(f"  {ind:<14} {ini:>8.2f}  {ql_:>8.2f}  {dqn_:>8.2f}  "
                  f"{ql_-ini:>+7.2f}  {dqn_-ini:>+7.2f}")

    plt.suptitle(
        f"Avaliação em UDHs Reais — Inicial × Q-Learning × DQN\n"
        f"(Região Metropolitana de São Paulo, 2010)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    caminho = os.path.join(DATA_DIR, "fig_avaliacao_udhs.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.show()
    print(f"\nSalvo: {caminho}")


# ---------------------------------------------------------------------------
# 7. Salvar tabela de resultados
# ---------------------------------------------------------------------------

def salvar_resultados_csv(resultados: dict) -> None:
    """
    Salva tabela resumo com métricas de todas as seeds.
    Inclui recompensa de avaliação e tempo por algoritmo e seed.
    """
    linhas = []
    for seed in range(NUM_SEEDS):
        linhas.append({
            "seed":           seed,
            "algoritmo":      "Q-Learning",
            "recomp_aval":    resultados["ql"]["recompensas_aval"][seed],
            "tempo_s":        resultados["ql"]["tempos"][seed],
            "recomp_treino_media": np.mean(resultados["ql"]["recompensas_treino"][seed]),
            "recomp_treino_final": resultados["ql"]["recompensas_treino"][seed][-1],
        })
        linhas.append({
            "seed":           seed,
            "algoritmo":      "DQN",
            "recomp_aval":    resultados["dqn"]["recompensas_aval"][seed],
            "tempo_s":        resultados["dqn"]["tempos"][seed],
            "recomp_treino_media": np.mean(resultados["dqn"]["recompensas_treino"][seed]),
            "recomp_treino_final": resultados["dqn"]["recompensas_treino"][seed][-1],
        })

    df = pd.DataFrame(linhas)
    caminho = os.path.join(DATA_DIR, "resultados.csv")
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"\nSalvo: {caminho}")

    # Resumo estatístico por algoritmo
    print(f"\n{'='*65}")
    print("  Resumo estatístico (30 seeds)")
    print(f"{'='*65}")
    print(df.groupby("algoritmo")[["recomp_aval", "tempo_s"]].agg(["mean", "std"]).round(4).to_string())
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  EXPERIMENTOS — Q-Learning vs DQN")
    print("  Cidade Sustentável (ODS 11) — UDH São Paulo")
    print("=" * 65 + "\n")

    # ── Experimento 1: 30 seeds ──────────────────────────────────
    resultados = treinar_multiplas_seeds(verbose=True)

    # ── Experimento 2: Curvas de aprendizado ─────────────────────
    print("\n[Gráfico] Curvas de aprendizado...")
    plot_curvas_aprendizado(resultados)

    # ── Experimento 3: Boxplot recompensas ───────────────────────
    print("\n[Gráfico] Boxplot de recompensas...")
    plot_boxplot_recompensas(resultados)

    # ── Experimento 4: Distribuição de ações ─────────────────────
    print("\n[Gráfico] Distribuição de ações...")
    plot_distribuicao_acoes(resultados)

    # ── Experimento 5: Estado final da cidade ────────────────────
    print("\n[Gráfico] Estado final da cidade...")
    plot_estado_final()

    # ── Experimento 6: Avaliação em UDHs reais ───────────────────
    print("\n[Gráfico] Avaliação em UDHs reais...")
    avaliar_udhs_reais(n_udhs=5)

    # ── Experimento 7: Salvar CSV de resultados ──────────────────
    salvar_resultados_csv(resultados)

    print("\n" + "=" * 65)
    print("  Experimentos concluídos!")
    print(f"  Arquivos salvos em: {os.path.abspath(DATA_DIR)}")
    print(f"  Modelos salvos em : {os.path.abspath(MODELOS_DIR)}")
    print("=" * 65)