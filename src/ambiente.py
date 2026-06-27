"""
ambiente.py
-----------
Ambiente MDP para simulação de cidade sustentável.
Baseado na especificação: MDP = (S, A, R, P, γ)

Estados calibrados com dados reais do Atlas IDHM (UDH São Paulo).
ODS 11 — Cidades e Comunidades Sustentáveis.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NUM_TURNOS = 20  # duração de cada episódio

# Nomes das ações
ACOES = [
    "Construir Usina Solar",           # A1
    "Expandir Transporte Público",     # A2
    "Programa de Controle de Emissões",# A3
    "Incentivo Industrial",            # A4
    "Programa de Geração de Empregos", # A5
    "Construção de Habitação Popular", # A6
    "Construção de Escolas",           # A7
    "Construção de Hospitais",         # A8
]

NUM_ACOES = len(ACOES)

# Nomes dos indicadores do estado (mesma ordem usada internamente)
INDICADORES = [
    "CleanEnergy",
    "Mobility",
    "AirQuality",
    "Economy",
    "Employment",
    "Housing",
    "Education",
    "Healthcare",
]

# ---------------------------------------------------------------------------
# Valores iniciais padrão
# Média normalizada dos UDHs do estado de São Paulo (escala 0–10).
# Estes valores devem ser substituídos pela média real calculada no eda.py.
# ---------------------------------------------------------------------------
ESTADO_INICIAL_PADRAO = {
    "CleanEnergy": 5.0,
    "Mobility":    5.0,
    "AirQuality":  5.0,
    "Economy":     5.0,
    "Employment":  5.0,
    "Housing":     5.0,
    "Education":   5.0,
    "Healthcare":  5.0,
}

# ---------------------------------------------------------------------------
# Matriz de transição
# Cada ação define o delta aplicado a cada indicador.
# Valores positivos melhoram o indicador; negativos pioram.
# Calibrados com base nas correlações observadas nos dados do Atlas IDHM.
# ---------------------------------------------------------------------------
TRANSICOES = {
    # A1 — Construir Usina Solar
    0: {
        "CleanEnergy": +1.5,
        "AirQuality":  +0.8,
        "Economy":     -0.5,
        "Employment":  +0.3,
        "Mobility":     0.0,
        "Housing":      0.0,
        "Education":    0.0,
        "Healthcare":  +0.1,
    },
    # A2 — Expandir Transporte Público
    1: {
        "Mobility":    +1.5,
        "AirQuality":  +0.5,
        "Economy":     -0.4,
        "Employment":  +0.2,
        "CleanEnergy":  0.0,
        "Housing":      0.0,
        "Education":    0.0,
        "Healthcare":  +0.1,
    },
    # A3 — Programa de Controle de Emissões
    2: {
        "AirQuality":  +1.5,
        "CleanEnergy": +0.3,
        "Economy":     -0.3,
        "Employment":  -0.2,
        "Mobility":     0.0,
        "Housing":      0.0,
        "Education":    0.0,
        "Healthcare":  +0.2,
    },
    # A4 — Incentivo Industrial
    3: {
        "Economy":     +1.5,
        "Employment":  +0.8,
        "AirQuality":  -0.8,
        "CleanEnergy": -0.3,
        "Mobility":    -0.2,
        "Housing":      0.0,
        "Education":    0.0,
        "Healthcare":  -0.1,
    },
    # A5 — Programa de Geração de Empregos
    4: {
        "Employment":  +1.5,
        "Economy":     +0.5,
        "Education":   +0.2,
        "Housing":     +0.1,
        "CleanEnergy":  0.0,
        "Mobility":     0.0,
        "AirQuality":  -0.1,
        "Healthcare":   0.0,
    },
    # A6 — Construção de Habitação Popular
    5: {
        "Housing":     +1.5,
        "Healthcare":  +0.3,
        "Economy":     -0.4,
        "Employment":  +0.2,
        "CleanEnergy": -0.1,
        "Mobility":    -0.1,
        "AirQuality":   0.0,
        "Education":    0.0,
    },
    # A7 — Construção de Escolas
    6: {
        "Education":   +1.5,
        "Employment":  +0.3,
        "Economy":     -0.3,
        "Housing":     +0.1,
        "CleanEnergy":  0.0,
        "Mobility":     0.0,
        "AirQuality":   0.0,
        "Healthcare":  +0.1,
    },
    # A8 — Construção de Hospitais
    7: {
        "Healthcare":  +1.5,
        "Housing":     +0.2,
        "Economy":     -0.3,
        "Employment":  +0.3,
        "CleanEnergy":  0.0,
        "Mobility":     0.0,
        "AirQuality":  +0.1,
        "Education":    0.0,
    },
}


# ---------------------------------------------------------------------------
# Classe do ambiente
# ---------------------------------------------------------------------------

class CidadeSustentavelEnv:
    """
    Ambiente MDP de cidade sustentável.

    Parâmetros
    ----------
    estado_inicial : dict, opcional
        Dicionário com os valores iniciais de cada indicador (escala 0–10).
        Se None, usa ESTADO_INICIAL_PADRAO.
    ruido : float
        Desvio padrão do ruído gaussiano adicionado às transições.
        Simula incerteza do mundo real. Default=0.1.
    seed : int, opcional
        Semente para reprodutibilidade.
    """

    def __init__(self, estado_inicial=None, ruido=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.estado_inicial = estado_inicial if estado_inicial else ESTADO_INICIAL_PADRAO.copy()
        self.ruido = ruido
        self.estado = None
        self.turno = 0
        self.historico = []
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, estado_inicial=None):
        """
        Reinicia o ambiente.
        Se estado_inicial for fornecido, usa ele (ex: valores reais de um município).
        Caso contrário, usa o estado_inicial definido na construção.
        """
        base = estado_inicial if estado_inicial else self.estado_inicial
        self.estado = {k: float(v) for k, v in base.items()}
        self.turno = 0
        self.historico = [self.estado.copy()]
        return self._estado_vetor()

    # ------------------------------------------------------------------
    def step(self, acao):
        """
        Executa uma ação no ambiente.

        Parâmetros
        ----------
        acao : int
            Índice da ação (0 a 7).

        Retorna
        -------
        proximo_estado : np.ndarray
            Vetor com os valores dos indicadores após a ação.
        recompensa : float
            Recompensa calculada com base no novo estado.
        done : bool
            True se o episódio terminou (20 turnos).
        info : dict
            Informações adicionais (estado legível, turno atual).
        """
        assert 0 <= acao < NUM_ACOES, f"Ação inválida: {acao}"

        # Aplica transição com ruído
        deltas = TRANSICOES[acao]
        for indicador, delta in deltas.items():
            ruido = np.random.normal(0, self.ruido)
            novo_valor = self.estado[indicador] + delta + ruido
            # Clipa entre 0 e 10
            self.estado[indicador] = float(np.clip(novo_valor, 0.0, 10.0))

        self.turno += 1
        self.historico.append(self.estado.copy())

        recompensa = self._calcular_recompensa()
        done = self.turno >= NUM_TURNOS

        info = {
            "turno": self.turno,
            "estado": self.estado.copy(),
            "acao_nome": ACOES[acao],
            "dimensoes": self._dimensoes(),
        }

        return self._estado_vetor(), recompensa, done, info

    # ------------------------------------------------------------------
    def _calcular_recompensa(self):
        """
        Reward = 0.33 * Environmental + 0.33 * Economic + 0.34 * Social
        Conforme especificação do MDP.
        """
        env, eco, soc = self._dimensoes().values()
        return 0.33 * env + 0.33 * eco + 0.34 * soc

    # ------------------------------------------------------------------
    def _dimensoes(self):
        """Calcula as três dimensões de sustentabilidade."""
        s = self.estado
        environmental = (
            0.4 * s["CleanEnergy"] +
            0.3 * s["Mobility"] +
            0.3 * s["AirQuality"]
        )
        economic = (
            0.6 * s["Economy"] +
            0.4 * s["Employment"]
        )
        social = (
            0.34 * s["Housing"] +
            0.33 * s["Education"] +
            0.33 * s["Healthcare"]
        )
        return {
            "Environmental": round(environmental, 4),
            "Economic":      round(economic, 4),
            "Social":        round(social, 4),
        }

    # ------------------------------------------------------------------
    def _estado_vetor(self):
        """Retorna o estado como vetor numpy (ordem de INDICADORES)."""
        return np.array([self.estado[k] for k in INDICADORES], dtype=np.float32)

    # ------------------------------------------------------------------
    def estado_legivel(self):
        """Imprime o estado atual de forma legível."""
        print(f"\n{'='*45}")
        print(f"  Turno {self.turno}/{NUM_TURNOS}")
        print(f"{'='*45}")
        for k, v in self.estado.items():
            barra = "█" * int(v) + "░" * (10 - int(v))
            print(f"  {k:<14} {barra}  {v:.2f}")
        dims = self._dimensoes()
        print(f"{'─'*45}")
        for nome, val in dims.items():
            print(f"  {nome:<14} {val:.4f}")
        print(f"  {'Recompensa':<14} {self._calcular_recompensa():.4f}")
        print(f"{'='*45}")

    # ------------------------------------------------------------------
    @property
    def num_acoes(self):
        return NUM_ACOES

    @property
    def num_estados(self):
        return len(INDICADORES)


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste do Ambiente MDP ===\n")

    # Estado inicial médio (será substituído pela média real do eda.py)
    estado_medio = {
        "CleanEnergy": 9.2,   # T_LUZ médio SP (alto — quase universal)
        "Mobility":    7.3,   # inverso de T_OCUPDESLOC_1
        "AirQuality":  5.0,   # virá do SEEG
        "Economy":     5.8,   # IDHM_R * 10
        "Employment":  6.1,   # P_FORMAL / 10
        "Housing":     8.4,   # T_BANAGUA / 10
        "Education":   7.2,   # IDHM_E * 10
        "Healthcare":  8.0,   # IDHM_L * 10
    }

    env = CidadeSustentavelEnv(estado_inicial=estado_medio, ruido=0.05, seed=42)

    print("Estado inicial:")
    env.estado_legivel()

    # Simula um episódio com ações aleatórias
    done = False
    recompensa_total = 0

    while not done:
        acao = np.random.randint(0, NUM_ACOES)
        estado, recompensa, done, info = env.step(acao)
        recompensa_total += recompensa
        print(f"Turno {info['turno']:>2} | Ação: {info['acao_nome']:<35} | Recompensa: {recompensa:.4f}")

    print(f"\nRecompensa total acumulada: {recompensa_total:.4f}")
    print("\nEstado final:")
    env.estado_legivel()