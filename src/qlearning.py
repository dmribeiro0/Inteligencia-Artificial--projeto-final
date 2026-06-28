"""
qlearning.py
------------
Agente Q-Learning para o ambiente MDP de cidade sustentável.

Discretização baseada nos quartis reais da base Atlas IDHM (UDH São Paulo):
  - 5 bins por indicador, definidos pelos cortes Q1, Q2 (mediana) e Q3
  - Bins: [min, Q1) → 0 | [Q1, Q2) → 1 | [Q2, Q3) → 2 | [Q3, max) → 3 | max → 4
  - Q-table implementada como defaultdict (lazy) — evita pré-alocar 5^8 × 8 entradas

Integração com ambiente.py:
    from ambiente import CidadeSustentavelEnv, INDICADORES
    from qlearning import QLearningAgent

    env   = CidadeSustentavelEnv.from_json()
    agent = QLearningAgent()
    agent.treinar(env, episodios=500)
    agent.salvar("qtable.pkl")

Referências:
    Watkins & Dayan (1992) — Q-Learning
    Sutton & Barto (2018) — Reinforcement Learning: An Introduction, cap. 6
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Quartis reais da base Atlas IDHM (UDH São Paulo, normalizados 0–10)
# Fonte: eda.py sobre udh_sp.xlsx
# Ordem: mesma de INDICADORES em ambiente.py
# ---------------------------------------------------------------------------

QUARTIS = {
    #              Q1       Q2 (med)  Q3
    "CleanEnergy": [9.817,  10.000,  10.000],
    "Mobility":    [7.465,   8.871,   9.677],
    "AirQuality":  [9.420,   9.696,   9.849],
    "Economy":     [1.433,   2.172,   3.570],
    "Employment":  [4.892,   6.108,   7.196],
    "Housing":     [9.464,   9.711,   9.848],
    "Education":   [3.801,   5.445,   6.878],
    "Healthcare":  [2.197,   4.280,   6.553],
}

# Ordem canônica — deve ser idêntica à de INDICADORES em ambiente.py
INDICADORES = [
    "CleanEnergy", "Mobility", "AirQuality", "Economy",
    "Employment",  "Housing",  "Education",  "Healthcare",
]

# Pré-computa array de cortes na ordem canônica (shape: 8 × 3)
_CORTES = np.array([QUARTIS[ind] for ind in INDICADORES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Funções de discretização
# ---------------------------------------------------------------------------

def discretizar_estado(vetor: np.ndarray) -> tuple:
    """
    Converte o vetor contínuo do ambiente (shape 8,) em uma tupla discreta
    de 8 inteiros no intervalo [0, 4], usando os quartis reais como cortes.

    Bins resultantes por indicador:
        0 → valor < Q1           (abaixo do 1º quartil)
        1 → Q1 ≤ valor < Q2     (entre Q1 e mediana)
        2 → Q2 ≤ valor < Q3     (entre mediana e Q3)
        3 → Q3 ≤ valor < 10.0   (acima do 3º quartil)
        4 → valor == 10.0        (máximo absoluto)

    np.digitize(x, bins) retorna o índice do bin à direita do valor,
    produzindo naturalmente índices 0–3. O clipe em 4 cobre o caso
    em que o valor atinge exatamente 10.0 (o máximo da escala).

    Parâmetros
    ----------
    vetor : np.ndarray, shape (8,)
        Estado contínuo retornado por env.reset() ou env.step().

    Retorna
    -------
    tuple de 8 ints, cada um em {0, 1, 2, 3, 4}
    """
    bins = np.array([
        np.digitize(vetor[i], _CORTES[i]) for i in range(len(INDICADORES))
    ], dtype=np.int8)
    # Garante que valores == 10.0 fiquem no bin 4 (não extrapolam para 5)
    bins = np.clip(bins, 0, 4)
    return tuple(bins.tolist())


def bin_para_descricao(estado_discreto: tuple) -> dict:
    """
    Retorna um dict legível mapeando cada indicador ao seu bin e rótulo.
    Útil para depuração e análise da política aprendida.

    Exemplo:
        {'CleanEnergy': {'bin': 3, 'label': 'Alto'},
         'Economy':     {'bin': 0, 'label': 'Crítico'}, ...}
    """
    rotulos = ["Crítico", "Baixo", "Médio", "Alto", "Máximo"]
    return {
        ind: {"bin": estado_discreto[i], "label": rotulos[estado_discreto[i]]}
        for i, ind in enumerate(INDICADORES)
    }


# ---------------------------------------------------------------------------
# Agente Q-Learning
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Agente Q-Learning tabular com política ε-greedy e decaimento linear de ε.

    A Q-table é um defaultdict: estados nunca visitados retornam Q=0 para
    todas as ações, evitando pré-alocação de 5^8 × 8 ≈ 3,1 M entradas.

    Parâmetros
    ----------
    num_acoes : int
        Número de ações disponíveis no ambiente. Default=8.
    alpha : float
        Taxa de aprendizado α ∈ (0, 1]. Default=0.1.
    gamma : float
        Fator de desconto γ ∈ [0, 1]. Default=0.95.
    epsilon_ini : float
        Valor inicial de ε (exploração). Default=1.0.
    epsilon_fim : float
        Valor mínimo de ε após decaimento. Default=0.05.
    epsilon_decay : float
        Decaimento multiplicativo de ε após cada episódio. Default=0.995.
    seed : int, opcional
        Semente para reprodutibilidade.

    Atributos públicos
    ------------------
    q_table : defaultdict
        Mapeamento (estado_discreto) → np.ndarray(num_acoes,).
    historico : dict
        Recompensas, tamanho da Q-table e ε por episódio (preenchido em treinar).
    epsilon : float
        Valor atual de ε (atualizado durante o treino).
    """

    def __init__(
        self,
        num_acoes: int = 8,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_ini: float = 1.0,
        epsilon_fim: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.num_acoes     = num_acoes
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_ini
        self.epsilon_fim   = epsilon_fim
        self.epsilon_decay = epsilon_decay

        # Q-table lazy: cria vetor de zeros ao acessar estado inédito
        self.q_table = defaultdict(lambda: np.zeros(self.num_acoes, dtype=np.float64))

        # Histórico de treino
        self.historico: dict = {
            "recompensas":   [],   # recompensa total por episódio
            "qtable_size":   [],   # nº de estados visitados por episódio
            "epsilons":      [],   # ε ao início de cada episódio
        }

    # ------------------------------------------------------------------
    # Política
    # ------------------------------------------------------------------

    def agir(self, estado: np.ndarray, explorar: bool = True) -> int:
        """
        Seleciona uma ação via política ε-greedy (treino) ou greedy (avaliação).

        Parâmetros
        ----------
        estado : np.ndarray
            Vetor contínuo retornado pelo ambiente.
        explorar : bool
            Se True, aplica ε-greedy. Se False, apenas greedy (avaliação).

        Retorna
        -------
        int — índice da ação escolhida.
        """
        if explorar and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acoes)

        s = discretizar_estado(estado)
        return int(np.argmax(self.q_table[s]))

    # ------------------------------------------------------------------
    # Atualização de Bellman
    # ------------------------------------------------------------------

    def _atualizar(
        self,
        estado:         np.ndarray,
        acao:           int,
        recompensa:     float,
        prox_estado:    np.ndarray,
        done:           bool,
    ) -> float:
        """
        Aplica a regra de atualização Q-Learning (off-policy):

            Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

        O termo `max_a' Q(s', a')` é zerado quando o episódio termina (done=True),
        pois não há próximo estado a considerar.

        Retorna
        -------
        float — erro TD (|alvo − Q(s,a)|) para monitoramento.
        """
        s  = discretizar_estado(estado)
        s_ = discretizar_estado(prox_estado)

        q_atual = self.q_table[s][acao]
        q_max   = 0.0 if done else float(np.max(self.q_table[s_]))
        alvo    = recompensa + self.gamma * q_max
        td_erro = alvo - q_atual

        self.q_table[s][acao] = q_atual + self.alpha * td_erro
        return abs(td_erro)

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def treinar(
        self,
        env,
        episodios:   int = 500,
        verbose:     bool = True,
        log_intervalo: int = 50,
    ) -> dict:
        """
        Executa o loop principal de treinamento Q-Learning.

        A cada episódio:
          1. Reseta o ambiente e obtém estado inicial.
          2. Escolhe ação via ε-greedy, executa, obtém (s', r, done).
          3. Atualiza Q-table via regra de Bellman.
          4. Decai ε multiplicativamente (mínimo: epsilon_fim).
          5. Registra métricas no histórico.

        Parâmetros
        ----------
        env : CidadeSustentavelEnv
            Instância do ambiente (ambiente.py).
        episodios : int
            Número de episódios de treino. Default=500.
        verbose : bool
            Imprime métricas a cada `log_intervalo` episódios.
        log_intervalo : int
            Frequência de impressão quando verbose=True.

        Retorna
        -------
        dict — histórico com chaves 'recompensas', 'qtable_size', 'epsilons'.
        """
        for ep in range(1, episodios + 1):
            estado = env.reset()
            recompensa_total = 0.0
            done = False

            self.historico["epsilons"].append(self.epsilon)

            while not done:
                acao = self.agir(estado, explorar=True)
                prox_estado, recompensa, done, _ = env.step(acao)
                self._atualizar(estado, acao, recompensa, prox_estado, done)
                estado = prox_estado
                recompensa_total += recompensa

            # Decaimento de ε
            self.epsilon = max(self.epsilon_fim, self.epsilon * self.epsilon_decay)

            # Registro
            self.historico["recompensas"].append(recompensa_total)
            self.historico["qtable_size"].append(len(self.q_table))

            if verbose and (ep % log_intervalo == 0 or ep == 1):
                media = np.mean(self.historico["recompensas"][-log_intervalo:])
                print(
                    f"Ep {ep:>5}/{episodios} | "
                    f"Recomp. média: {media:>7.4f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"Estados visitados: {len(self.q_table)}"
                )

        if verbose:
            print(f"\n[Treino concluído] Estados únicos na Q-table: {len(self.q_table)}")

        return self.historico

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def avaliar(self, env, episodios: int = 10, verbose: bool = True) -> dict:
        """
        Avalia o agente treinado em modo greedy (sem exploração).

        Parâmetros
        ----------
        env : CidadeSustentavelEnv
            Ambiente de avaliação (pode ser carregado via from_csv para UDH real).
        episodios : int
            Número de episódios de avaliação.
        verbose : bool
            Imprime resultado de cada episódio.

        Retorna
        -------
        dict com 'recompensas', 'media', 'std', 'acoes_escolhidas'.

        Exemplo:
            env2 = CidadeSustentavelEnv.from_csv(idx=42)
            resultado = agent.avaliar(env2, episodios=5)
        """
        recompensas = []
        acoes_log   = []

        for ep in range(1, episodios + 1):
            estado = env.reset()
            recompensa_total = 0.0
            acoes_ep = []
            done = False

            while not done:
                acao = self.agir(estado, explorar=False)
                estado, recompensa, done, info = env.step(acao)
                recompensa_total += recompensa
                acoes_ep.append(info["acao_nome"])

            recompensas.append(recompensa_total)
            acoes_log.append(acoes_ep)

            if verbose:
                print(f"Avaliação {ep:>2} | Recompensa total: {recompensa_total:.4f}")

        resultado = {
            "recompensas":      recompensas,
            "media":            float(np.mean(recompensas)),
            "std":              float(np.std(recompensas)),
            "acoes_escolhidas": acoes_log,
        }

        if verbose:
            print(f"\nMédia: {resultado['media']:.4f}  ±  {resultado['std']:.4f}")

        return resultado

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def salvar(self, caminho: str = "qtable.pkl") -> None:
        """
        Serializa o agente (Q-table + hiperparâmetros + histórico) em disco.

        Exemplo:
            agent.salvar("modelos/qtable_500ep.pkl")
        """
        payload = {
            "q_table":       dict(self.q_table),   # defaultdict → dict para pickle
            "num_acoes":     self.num_acoes,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_fim":   self.epsilon_fim,
            "epsilon_decay": self.epsilon_decay,
            "historico":     self.historico,
        }
        with open(caminho, "wb") as f:
            pickle.dump(payload, f)
        print(f"[Q-Learning] Agente salvo em: {caminho}")

    @classmethod
    def carregar(cls, caminho: str = "qtable.pkl") -> "QLearningAgent":
        """
        Carrega um agente previamente salvo com salvar().

        Exemplo:
            agent = QLearningAgent.carregar("modelos/qtable_500ep.pkl")
            resultado = agent.avaliar(env)
        """
        with open(caminho, "rb") as f:
            payload = pickle.load(f)

        agente = cls(
            num_acoes     = payload["num_acoes"],
            alpha         = payload["alpha"],
            gamma         = payload["gamma"],
            epsilon_ini   = payload["epsilon"],
            epsilon_fim   = payload["epsilon_fim"],
            epsilon_decay  = payload["epsilon_decay"],
        )
        # Restaura Q-table como defaultdict
        agente.q_table = defaultdict(
            lambda: np.zeros(agente.num_acoes, dtype=np.float64),
            payload["q_table"],
        )
        agente.historico = payload["historico"]
        print(f"[Q-Learning] Agente carregado de: {caminho}")
        print(f"             Estados na Q-table: {len(agente.q_table)}")
        return agente

    # ------------------------------------------------------------------
    # Inspeção da política aprendida
    # ------------------------------------------------------------------

    def politica_aprendida(self, top_n: int = 10) -> None:
        """
        Imprime os `top_n` estados mais visitados e a ação greedy associada.
        Útil para interpretar o que o agente aprendeu.

        Parâmetros
        ----------
        top_n : int
            Quantos estados exibir, ordenados pelo valor Q máximo. Default=10.
        """
        from ambiente import ACOES  # importação local para evitar dependência circular

        if not self.q_table:
            print("[Q-Learning] Q-table vazia — treine o agente primeiro.")
            return

        # Ordena estados pelo valor Q máximo (melhor situação conhecida)
        ranking = sorted(
            self.q_table.items(),
            key=lambda kv: np.max(kv[1]),
            reverse=True,
        )[:top_n]

        print(f"\n{'='*65}")
        print(f"  Política aprendida — Top {top_n} estados por Q máximo")
        print(f"{'='*65}")

        rotulos = ["Crít", "Baix", "Méd", "Alto", "Máx"]
        header  = "  " + "  ".join(f"{ind[:6]:>6}" for ind in INDICADORES)
        print(header)
        print(f"  {'─'*60}")

        for estado, q_vals in ranking:
            bins_str = "  ".join(f"{rotulos[b]:>6}" for b in estado)
            acao_idx = int(np.argmax(q_vals))
            q_max    = np.max(q_vals)
            print(f"  {bins_str}  →  {ACOES[acao_idx]:<35} (Q={q_max:.3f})")

        print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from ambiente import CidadeSustentavelEnv, INDICADORES as IND_AMB

    print("=== Teste do Agente Q-Learning ===\n")

    # Carrega ambiente de treino
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "estado_inicial_medio.json")
    if os.path.exists(json_path):
        env = CidadeSustentavelEnv.from_json(ruido=0.05, seed=42)
    else:
        print("⚠️  estado_inicial_medio.json não encontrado — usando estado padrão.\n")
        env = CidadeSustentavelEnv(ruido=0.05, seed=42)

    # Instancia agente
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon_ini=1.0,
        epsilon_fim=0.05,
        epsilon_decay=0.995,
        seed=42,
    )

    # Treino
    print("--- Treinamento (500 episódios) ---")
    agent.treinar(env, episodios=500, verbose=True, log_intervalo=100)

    # Mostra política
    agent.politica_aprendida(top_n=5)

    # Avaliação
    print("\n--- Avaliação (10 episódios, modo greedy) ---")
    agent.avaliar(env, episodios=10)

    # Salva e recarrega
    agent.salvar("/tmp/qtable_teste.pkl")
    agent2 = QLearningAgent.carregar("/tmp/qtable_teste.pkl")
    print(f"\nAgente recarregado — Q-table com {len(agent2.q_table)} estados.")

    # Teste de discretização
    print("\n--- Teste de discretização ---")
    estado_exemplo = np.array([10.0, 5.0, 9.5, 2.5, 6.0, 9.7, 4.0, 3.0], dtype=np.float32)
    discreto = discretizar_estado(estado_exemplo)
    descricao = bin_para_descricao(discreto)
    print(f"Vetor contínuo : {estado_exemplo}")
    print(f"Estado discreto: {discreto}")
    for ind, info in descricao.items():
        print(f"  {ind:<14} bin={info['bin']}  ({info['label']})")