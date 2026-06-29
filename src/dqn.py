"""
dqn.py
------
Agente DQN (Deep Q-Network) para o ambiente MDP de cidade sustentável.

Diferenças metodológicas em relação ao Q-Learning (qlearning.py):
  - Estado contínuo (8 floats) entra direto na rede — sem discretização
  - A função Q(s,a) é aproximada por uma rede neural (MLP 3 camadas)
  - Replay Buffer: armazena experiências (s, a, r, s', done) e treina
    com mini-batches aleatórios, quebrando correlação temporal
  - Rede-alvo (target network): cópia da rede principal atualizada a cada
    C passos, estabilizando o treinamento (Mnih et al., 2015)

Arquitetura da rede:
    input(8) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(8)
    (8 entradas = indicadores do estado | 8 saídas = Q por ação)

Integração com ambiente.py:
    from ambiente import CidadeSustentavelEnv
    from dqn import DQNAgent

    env   = CidadeSustentavelEnv.from_json()
    agent = DQNAgent()
    agent.treinar(env, episodios=500)
    agent.salvar("modelos/dqn_500ep.pt")

Referências:
    Mnih et al. (2015) — Human-level control through deep reinforcement learning
    Sutton & Barto (2018) — Reinforcement Learning: An Introduction, cap. 9
"""

import os
import random
import numpy as np
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Rede neural Q
# ---------------------------------------------------------------------------

class RedeQ(nn.Module):
    """
    MLP com duas camadas ocultas que aproxima Q(s, a) para todas as ações.

    Entrada : vetor de estado contínuo (dim_estado,)
    Saída   : vetor de valores Q, um por ação (num_acoes,)

    Parâmetros
    ----------
    dim_estado : int   — número de indicadores (8)
    num_acoes  : int   — número de ações disponíveis (8)
    dim_oculta : int   — neurônios por camada oculta (default 64)
    """

    def __init__(self, dim_estado: int = 8, num_acoes: int = 8, dim_oculta: int = 64):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_estado, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, num_acoes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Armazena experiências (s, a, r, s', done) e fornece mini-batches aleatórios.

    O sampling aleatório quebra a correlação temporal entre experiências
    consecutivas, estabilizando o gradiente da rede Q.

    Parâmetros
    ----------
    capacidade : int   — número máximo de experiências armazenadas (FIFO)
    """

    def __init__(self, capacidade: int = 10_000):
        self.buffer = deque(maxlen=capacidade)

    def adicionar(
        self,
        estado:      np.ndarray,
        acao:        int,
        recompensa:  float,
        prox_estado: np.ndarray,
        done:        bool,
    ) -> None:
        self.buffer.append((estado, acao, recompensa, prox_estado, done))

    def amostrar(self, batch_size: int):
        """
        Retorna `batch_size` experiências aleatórias como tensores PyTorch.

        Retorna
        -------
        estados, acoes, recompensas, proximos_estados, dones
        """
        batch = random.sample(self.buffer, batch_size)
        estados, acoes, recompensas, proximos_estados, dones = zip(*batch)

        return (
            torch.tensor(np.array(estados),        dtype=torch.float32),
            torch.tensor(acoes,                    dtype=torch.long),
            torch.tensor(recompensas,              dtype=torch.float32),
            torch.tensor(np.array(proximos_estados), dtype=torch.float32),
            torch.tensor(dones,                    dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agente DQN
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Agente DQN com replay buffer, rede-alvo e política ε-greedy.

    Parâmetros
    ----------
    dim_estado : int
        Dimensão do vetor de estado (número de indicadores). Default=8.
    num_acoes : int
        Número de ações disponíveis. Default=8.
    dim_oculta : int
        Neurônios por camada oculta da rede Q. Default=64.
    alpha : float
        Taxa de aprendizado do otimizador Adam. Default=1e-3.
    gamma : float
        Fator de desconto γ. Default=0.95.
    epsilon_ini : float
        Valor inicial de ε (exploração). Default=1.0.
    epsilon_fim : float
        Valor mínimo de ε. Default=0.05.
    epsilon_decay : float
        Decaimento multiplicativo de ε por episódio. Default=0.995.
    batch_size : int
        Tamanho do mini-batch do replay buffer. Default=64.
    buffer_capacidade : int
        Capacidade máxima do replay buffer. Default=10_000.
    atualizar_alvo_a_cada : int
        A cada quantos episódios copia os pesos para a rede-alvo. Default=10.
    seed : int, opcional
        Semente para reprodutibilidade.

    Atributos públicos
    ------------------
    rede_q      : RedeQ   — rede principal (treinada a cada passo)
    rede_alvo   : RedeQ   — rede-alvo (atualizada periodicamente)
    historico   : dict    — métricas de treino por episódio
    epsilon     : float   — valor atual de ε
    """

    def __init__(
        self,
        dim_estado:           int   = 8,
        num_acoes:            int   = 8,
        dim_oculta:           int   = 64,
        alpha:                float = 1e-3,
        gamma:                float = 0.95,
        epsilon_ini:          float = 1.0,
        epsilon_fim:          float = 0.05,
        epsilon_decay:        float = 0.995,
        batch_size:           int   = 64,
        buffer_capacidade:    int   = 10_000,
        atualizar_alvo_a_cada: int  = 10,
        seed: Optional[int]         = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.dim_estado            = dim_estado
        self.num_acoes             = num_acoes
        self.gamma                 = gamma
        self.epsilon               = epsilon_ini
        self.epsilon_fim           = epsilon_fim
        self.epsilon_decay         = epsilon_decay
        self.batch_size            = batch_size
        self.atualizar_alvo_a_cada = atualizar_alvo_a_cada

        # Dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Redes Q principal e alvo
        self.rede_q    = RedeQ(dim_estado, num_acoes, dim_oculta).to(self.device)
        self.rede_alvo = RedeQ(dim_estado, num_acoes, dim_oculta).to(self.device)
        self._sincronizar_alvo()   # alvo começa igual à principal

        # Otimizador e função de perda
        self.otimizador = optim.Adam(self.rede_q.parameters(), lr=alpha)
        self.criterio   = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacidade)

        # Histórico de treino
        self.historico: dict = {
            "recompensas": [],   # recompensa total por episódio
            "perdas":      [],   # perda média por episódio
            "epsilons":    [],   # ε ao início de cada episódio
        }

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _sincronizar_alvo(self) -> None:
        """Copia os pesos da rede principal para a rede-alvo."""
        self.rede_alvo.load_state_dict(self.rede_q.state_dict())

    def _estado_tensor(self, estado: np.ndarray) -> torch.Tensor:
        """Converte vetor numpy em tensor float32 no device correto."""
        return torch.tensor(estado, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Política
    # ------------------------------------------------------------------

    def agir(self, estado: np.ndarray, explorar: bool = True) -> int:
        """
        Seleciona uma ação via política ε-greedy (treino) ou greedy (avaliação).

        Diferente do Q-Learning, o estado contínuo entra direto na rede —
        sem nenhuma discretização.

        Parâmetros
        ----------
        estado : np.ndarray
            Vetor contínuo retornado pelo ambiente (shape 8,).
        explorar : bool
            Se True aplica ε-greedy. Se False apenas greedy (avaliação).

        Retorna
        -------
        int — índice da ação escolhida.
        """
        if explorar and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acoes)

        with torch.no_grad():
            q_vals = self.rede_q(self._estado_tensor(estado))
        return int(q_vals.argmax().item())

    # ------------------------------------------------------------------
    # Atualização da rede
    # ------------------------------------------------------------------

    def _aprender(self) -> float:
        """
        Treina a rede Q com um mini-batch do replay buffer.

        Algoritmo:
          1. Amostra `batch_size` experiências do buffer.
          2. Calcula alvo de Bellman usando a rede-alvo:
                alvo = r  (se done)
                alvo = r + γ · max_a' Q_alvo(s', a')  (caso contrário)
          3. Calcula perda MSE entre Q_principal(s, a) e alvo.
          4. Backpropagation + passo do otimizador.

        A rede-alvo é usada no cálculo do alvo (passo 2) para estabilizar
        o treinamento — sem ela, o alvo muda a cada passo junto com a rede,
        criando instabilidade.

        Retorna
        -------
        float — valor da perda (para monitoramento).
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        estados, acoes, recompensas, prox_estados, dones = self.buffer.amostrar(self.batch_size)

        # Move para o device
        estados      = estados.to(self.device)
        acoes        = acoes.to(self.device)
        recompensas  = recompensas.to(self.device)
        prox_estados = prox_estados.to(self.device)
        dones        = dones.to(self.device)

        # Q(s, a) da rede principal para as ações tomadas
        q_atuais = self.rede_q(estados).gather(1, acoes.unsqueeze(1)).squeeze(1)

        # Alvo de Bellman com rede-alvo (sem gradiente)
        with torch.no_grad():
            q_prox = self.rede_alvo(prox_estados).max(1).values
            alvos  = recompensas + self.gamma * q_prox * (1.0 - dones)

        # Perda MSE e backprop
        perda = self.criterio(q_atuais, alvos)
        self.otimizador.zero_grad()
        perda.backward()
        # Gradient clipping — evita explosão de gradiente
        nn.utils.clip_grad_norm_(self.rede_q.parameters(), max_norm=1.0)
        self.otimizador.step()

        return float(perda.item())

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def treinar(
        self,
        env,
        episodios:     int  = 500,
        verbose:       bool = True,
        log_intervalo: int  = 50,
    ) -> dict:
        """
        Executa o loop principal de treinamento DQN.

        A cada episódio:
          1. Reseta o ambiente.
          2. A cada turno: age via ε-greedy, armazena no buffer, aprende.
          3. A cada `atualizar_alvo_a_cada` episódios: sincroniza rede-alvo.
          4. Decai ε multiplicativamente.
          5. Registra métricas.

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
        dict — histórico com chaves 'recompensas', 'perdas', 'epsilons'.
        """
        for ep in range(1, episodios + 1):
            estado = env.reset()
            recompensa_total = 0.0
            perdas_ep        = []
            done             = False

            self.historico["epsilons"].append(self.epsilon)

            while not done:
                acao = self.agir(estado, explorar=True)
                prox_estado, recompensa, done, _ = env.step(acao)

                # Armazena experiência
                self.buffer.adicionar(estado, acao, recompensa, prox_estado, done)

                # Aprende com mini-batch (só após buffer ter dados suficientes)
                perda = self._aprender()
                if perda > 0:
                    perdas_ep.append(perda)

                estado = prox_estado
                recompensa_total += recompensa

            # Sincroniza rede-alvo periodicamente
            if ep % self.atualizar_alvo_a_cada == 0:
                self._sincronizar_alvo()

            # Decaimento de ε
            self.epsilon = max(self.epsilon_fim, self.epsilon * self.epsilon_decay)

            # Registro
            self.historico["recompensas"].append(recompensa_total)
            self.historico["perdas"].append(np.mean(perdas_ep) if perdas_ep else 0.0)

            if verbose and (ep % log_intervalo == 0 or ep == 1):
                media_r = np.mean(self.historico["recompensas"][-log_intervalo:])
                media_p = np.mean(self.historico["perdas"][-log_intervalo:])
                print(
                    f"Ep {ep:>5}/{episodios} | "
                    f"Recomp. média: {media_r:>7.4f} | "
                    f"Perda média: {media_p:>7.4f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)}"
                )

        if verbose:
            print(f"\n[Treino concluído] Buffer final: {len(self.buffer)} experiências")

        return self.historico

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def avaliar(self, env, episodios: int = 10, verbose: bool = True) -> dict:
        """
        Avalia o agente em modo greedy (sem exploração, rede em eval mode).

        Interface idêntica ao QLearningAgent.avaliar() para facilitar
        comparação direta no experimentos.py.

        Parâmetros
        ----------
        env : CidadeSustentavelEnv
            Ambiente de avaliação (pode ser UDH real via from_csv).
        episodios : int
            Número de episódios de avaliação.
        verbose : bool
            Imprime resultado de cada episódio.

        Retorna
        -------
        dict com 'recompensas', 'media', 'std', 'acoes_escolhidas'.
        """
        self.rede_q.eval()
        recompensas = []
        acoes_log   = []

        for ep in range(1, episodios + 1):
            estado = env.reset()
            recompensa_total = 0.0
            acoes_ep         = []
            done             = False

            while not done:
                acao = self.agir(estado, explorar=False)
                estado, recompensa, done, info = env.step(acao)
                recompensa_total += recompensa
                acoes_ep.append(info["acao_nome"])

            recompensas.append(recompensa_total)
            acoes_log.append(acoes_ep)

            if verbose:
                print(f"Avaliação {ep:>2} | Recompensa total: {recompensa_total:.4f}")

        self.rede_q.train()

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

    def salvar(self, caminho: str = "dqn.pt") -> None:
        """
        Salva os pesos das redes, hiperparâmetros e histórico em disco.

        Exemplo:
            agent.salvar("modelos/dqn_500ep.pt")
        """
        os.makedirs(os.path.dirname(caminho) if os.path.dirname(caminho) else ".", exist_ok=True)
        torch.save({
            "rede_q_state":    self.rede_q.state_dict(),
            "rede_alvo_state": self.rede_alvo.state_dict(),
            "dim_estado":      self.dim_estado,
            "num_acoes":       self.num_acoes,
            "gamma":           self.gamma,
            "epsilon":         self.epsilon,
            "epsilon_fim":     self.epsilon_fim,
            "epsilon_decay":   self.epsilon_decay,
            "batch_size":      self.batch_size,
            "atualizar_alvo_a_cada": self.atualizar_alvo_a_cada,
            "historico":       self.historico,
        }, caminho)
        print(f"[DQN] Agente salvo em: {caminho}")

    @classmethod
    def carregar(cls, caminho: str = "dqn.pt") -> "DQNAgent":
        """
        Carrega um agente previamente salvo com salvar().

        Exemplo:
            agent = DQNAgent.carregar("modelos/dqn_500ep.pt")
            resultado = agent.avaliar(env)
        """
        payload = torch.load(caminho, map_location="cpu", weights_only=False)

        agente = cls(
            dim_estado            = payload["dim_estado"],
            num_acoes             = payload["num_acoes"],
            gamma                 = payload["gamma"],
            epsilon_ini           = payload["epsilon"],
            epsilon_fim           = payload["epsilon_fim"],
            epsilon_decay         = payload["epsilon_decay"],
            batch_size            = payload["batch_size"],
            atualizar_alvo_a_cada = payload["atualizar_alvo_a_cada"],
        )
        agente.rede_q.load_state_dict(payload["rede_q_state"])
        agente.rede_alvo.load_state_dict(payload["rede_alvo_state"])
        agente.historico = payload["historico"]

        print(f"[DQN] Agente carregado de: {caminho}")
        print(f"      Buffer de treino tinha {len(agente.historico['recompensas'])} episódios.")
        return agente

    # ------------------------------------------------------------------
    # Inspeção da política aprendida
    # ------------------------------------------------------------------

    def politica_aprendida(self, estados_teste: Optional[list] = None) -> None:
        """
        Mostra a ação greedy escolhida pela rede para estados de referência.

        Diferente do Q-Learning, não há tabela para inspecionar — a política
        é implícita nos pesos da rede. Esta função avalia a rede em estados
        representativos para dar interpretabilidade.

        Parâmetros
        ----------
        estados_teste : list de np.ndarray, opcional
            Estados a avaliar. Se None, usa estados canônicos (mín, médio, máx).
        """
        from ambiente import ACOES

        if estados_teste is None:
            # Estados canônicos: todos baixos, todos médios, todos altos
            estados_teste = [
                ("Todos críticos (2.0)",  np.full(self.dim_estado, 2.0, dtype=np.float32)),
                ("Todos médios  (5.0)",   np.full(self.dim_estado, 5.0, dtype=np.float32)),
                ("Todos altos   (8.0)",   np.full(self.dim_estado, 8.0, dtype=np.float32)),
                ("Economy baixo",         np.array([8,8,8,1,8,8,8,8], dtype=np.float32)),
                ("Education baixo",       np.array([8,8,8,8,8,8,1,8], dtype=np.float32)),
                ("Healthcare baixo",      np.array([8,8,8,8,8,8,8,1], dtype=np.float32)),
            ]

        self.rede_q.eval()
        print(f"\n{'='*65}")
        print(f"  Política DQN — ação greedy por estado de referência")
        print(f"{'='*65}")

        with torch.no_grad():
            for descricao, estado in estados_teste:
                t     = torch.tensor(estado, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_vals = self.rede_q(t).squeeze(0)
                acao   = int(q_vals.argmax().item())
                q_max  = float(q_vals.max().item())
                print(f"  {descricao:<25} →  {ACOES[acao]:<35} (Q={q_max:.3f})")

        print(f"{'='*65}")
        self.rede_q.train()


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from ambiente import CidadeSustentavelEnv

    print("=== Teste do Agente DQN ===\n")
    print(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}\n")

    # Carrega ambiente de treino
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "estado_inicial_medio.json")
    if os.path.exists(json_path):
        env = CidadeSustentavelEnv.from_json(ruido=0.05, seed=42)
    else:
        print("⚠️  estado_inicial_medio.json não encontrado — usando estado padrão.\n")
        env = CidadeSustentavelEnv(ruido=0.05, seed=42)

    # Instancia agente
    agent = DQNAgent(
        dim_estado            = env.num_estados,
        num_acoes             = env.num_acoes,
        dim_oculta            = 64,
        alpha                 = 1e-3,
        gamma                 = 0.95,
        epsilon_ini           = 1.0,
        epsilon_fim           = 0.05,
        epsilon_decay         = 0.995,
        batch_size            = 64,
        buffer_capacidade     = 10_000,
        atualizar_alvo_a_cada = 10,
        seed                  = 42,
    )

    # Treino
    print("--- Treinamento (500 episódios) ---")
    agent.treinar(env, episodios=500, verbose=True, log_intervalo=100)

    # Política aprendida
    agent.politica_aprendida()

    # Avaliação
    print("\n--- Avaliação (10 episódios, modo greedy) ---")
    agent.avaliar(env, episodios=10)

    # Salva e recarrega
    agent.salvar("/tmp/dqn_teste.pt")
    agent2 = DQNAgent.carregar("/tmp/dqn_teste.pt")
    print(f"\nAgente recarregado — {len(agent2.historico['recompensas'])} episódios de histórico.")