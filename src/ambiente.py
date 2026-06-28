"""
ambiente.py
-----------
Ambiente MDP para simulação de cidade sustentável.
Baseado na especificação: MDP = (S, A, R, P, γ)

Estados calibrados com dados reais do Atlas IDHM (UDH São Paulo).
ODS 11 — Cidades e Comunidades Sustentáveis.

Integração com eda.py:
    O eda.py gera dois arquivos em data/ que este módulo consome:
      - data/estado_inicial_medio.json  : estado inicial médio das UDHs (treino)
      - data/udh_normalizado.csv        : todas as UDHs individuais (avaliação)

Uso típico:
    # Treinamento (estado médio das UDHs)
    env = CidadeSustentavelEnv.from_json("../data/estado_inicial_medio.json")

    # Avaliação em UDH real específica
    env = CidadeSustentavelEnv.from_csv("../data/udh_normalizado.csv", idx=42)
"""

import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos padrão dos arquivos gerados pelo eda.py
# ---------------------------------------------------------------------------
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
JSON_PATH_PADRAO = os.path.join(DATA_DIR, "estado_inicial_medio.json")
CSV_PATH_PADRAO  = os.path.join(DATA_DIR, "udh_normalizado.csv")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NUM_TURNOS = 20  # duração de cada episódio

ACOES = [
    "Construir Usina Solar",            # A1
    "Expandir Transporte Público",      # A2
    "Programa de Controle de Emissões", # A3
    "Incentivo Industrial",             # A4
    "Programa de Geração de Empregos",  # A5
    "Construção de Habitação Popular",  # A6
    "Construção de Escolas",            # A7
    "Construção de Hospitais",          # A8
]

NUM_ACOES = len(ACOES)

# Nomes dos indicadores — mesma ordem do eda.py / udh_normalizado.csv
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
# Desvio-padrão dos indicadores (obtido do Atlas IDHM)
# Utilizado para calibrar a intensidade das transições.
# ---------------------------------------------------------------------------

STD_INDICADORES = {
    "CleanEnergy": 0.601,
    "Mobility":    1.904,
    "AirQuality":  0.622,
    "Economy":     1.624,
    "Employment":  1.575,
    "Housing":     0.740,
    "Education":   2.090,
    "Healthcare":  2.612,
}

# Fallback neutro (usado apenas se nenhum arquivo for encontrado)
ESTADO_INICIAL_PADRAO = {ind: 5.0 for ind in INDICADORES}

# ---------------------------------------------------------------------------
# Intensidades das transições
# ---------------------------------------------------------------------------

PESOS_IMPACTO = {
    "forte": 0.50,
    "medio": 0.25,
    "fraco": 0.10,
}

def calcular_delta(indicador, intensidade, sinal):
    """
    Calcula o delta aplicado ao indicador.

    intensidade:
        "forte"
        "medio"
        "fraco"

    sinal:
        +1  -> melhora o indicador
        -1  -> piora o indicador
    """

    std = STD_INDICADORES[indicador]

    return sinal * PESOS_IMPACTO[intensidade] * std

# ---------------------------------------------------------------------------
# Matriz de transição
# Cada ação define o delta aplicado a cada indicador.
# Valores positivos melhoram; negativos pioram.
# Calibrados com base nas correlações observadas nos dados do Atlas IDHM.
# ---------------------------------------------------------------------------
TRANSICOES = {
    
    # -------------------------------------------------------------
    # A1 — Construir Usina Solar
    # -------------------------------------------------------------
    0: {
        "CleanEnergy": ("forte", +1),
        "AirQuality":  ("medio", +1),
        "Economy":     ("medio", -1),
        "Employment":  ("fraco", +1),
        "Healthcare":  ("fraco", +1),
    },

    # -------------------------------------------------------------
    # A2 — Expandir Transporte Público
    # -------------------------------------------------------------
    1: {
        "Mobility":     ("forte", +1),
        "AirQuality":   ("medio", +1),
        "Economy":      ("medio", -1),
        "Employment":   ("fraco", +1),
        "Healthcare":   ("fraco", +1),
    },

    # -------------------------------------------------------------
    # A3 — Programa de Controle de Emissões
    # -------------------------------------------------------------
    2: {
        "AirQuality":   ("forte", +1),
        "CleanEnergy":  ("fraco", +1),
        "Economy":      ("fraco", -1),
        "Employment":   ("fraco", -1),
        "Healthcare":   ("fraco", +1),
    },

    # -------------------------------------------------------------
    # A4 — Incentivo Industrial
    # -------------------------------------------------------------
    3: {
        "Economy":      ("forte", +1),
        "Employment":   ("medio", +1),
        "AirQuality":   ("medio", -1),
        "CleanEnergy":  ("fraco", -1),
        "Mobility":     ("fraco", -1),
        "Healthcare":   ("fraco", -1),
    },

    # -------------------------------------------------------------
    # A5 — Programa de Geração de Empregos
    # -------------------------------------------------------------
    4: {
        "Employment":   ("forte", +1),
        "Economy":      ("medio", +1),
        "Education":    ("fraco", +1),
        "Housing":      ("fraco", +1),
        "AirQuality":   ("fraco", -1),
    },

    # -------------------------------------------------------------
    # A6 — Construção de Habitação Popular
    # -------------------------------------------------------------
    5: {
        "Housing":      ("forte", +1),
        "Healthcare":   ("medio", +1),
        "Economy":      ("medio", -1),
        "Employment":   ("fraco", +1),
        "CleanEnergy":  ("fraco", -1),
        "Mobility":     ("fraco", -1),
    },

    # -------------------------------------------------------------
    # A7 — Construção de Escolas
    # -------------------------------------------------------------
    6: {
        "Education":    ("forte", +1),
        "Employment":   ("medio", +1),
        "Economy":      ("fraco", -1),
        "Housing":      ("fraco", +1),
        "Healthcare":   ("fraco", +1),
    },

    # -------------------------------------------------------------
    # A8 — Construção de Hospitais
    # -------------------------------------------------------------
    7: {
        "Healthcare":   ("forte", +1),
        "Housing":      ("fraco", +1),
        "Economy":      ("fraco", -1),
        "Employment":   ("fraco", +1),
        "AirQuality":   ("fraco", +1),
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
        Valores iniciais de cada indicador (escala 0-10).
        Se None, usa ESTADO_INICIAL_PADRAO.
    ruido : float
        Desvio padrão do ruído gaussiano nas transições. Default=0.1.
    seed : int, opcional
        Semente para reprodutibilidade.
    nome : str, opcional
        Nome da UDH/município (para identificação nos experimentos).
    """

    def __init__(self, estado_inicial=None, ruido=0.1, seed=None, nome="Cidade"):
        if seed is not None:
            np.random.seed(seed)

        self.estado_inicial = estado_inicial if estado_inicial else ESTADO_INICIAL_PADRAO.copy()
        self.ruido  = ruido
        self.nome   = nome
        self.estado = None
        self.turno  = 0
        self.historico = []
        self.reset()

    # ------------------------------------------------------------------
    # Construtores alternativos — integração com eda.py
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_path: str = JSON_PATH_PADRAO, **kwargs):
        """
        Cria o ambiente a partir do estado_inicial_medio.json gerado pelo eda.py.
        Usado na fase de TREINAMENTO.

        Exemplo:
            env = CidadeSustentavelEnv.from_json()
            env = CidadeSustentavelEnv.from_json("../data/estado_inicial_medio.json")
        """
        with open(json_path, "r") as f:
            estado = json.load(f)
        print(f"[Ambiente] Estado inicial carregado de: {json_path}")
        return cls(estado_inicial=estado, nome="Media_UDH_SP", **kwargs)

    @classmethod
    def from_csv(cls, csv_path: str = CSV_PATH_PADRAO, idx: int = 0, **kwargs):
        """
        Cria o ambiente a partir de uma linha do udh_normalizado.csv.
        Usado na fase de AVALIAÇÃO com UDHs reais individuais.

        Mobility é NaN nas UDHs do ano 2000 (T_OCUPDESLOC_1 só existe em 2010).
        Nesses casos usa a mediana de 2010 como fallback.

        Exemplo:
            env = CidadeSustentavelEnv.from_csv(idx=42)
            env = CidadeSustentavelEnv.from_csv("../data/udh_normalizado.csv", idx=100)
        """
        df    = pd.read_csv(csv_path)
        linha = df.iloc[idx]

        # Fallback de Mobility para UDHs do ano 2000
        mobility_fallback = (
            df[df["ANO"] == 2010]["Mobility"].median()
            if "ANO" in df.columns else 5.0
        )

        estado = {}
        for ind in INDICADORES:
            if ind not in linha or pd.isna(linha[ind]):
                estado[ind] = float(mobility_fallback if ind == "Mobility" else 5.0)
            else:
                estado[ind] = float(linha[ind])

        nome = str(linha.get("NOME_UDH", f"UDH_{idx}"))
        ano  = int(linha["ANO"]) if "ANO" in linha and not pd.isna(linha["ANO"]) else "?"
        print(f"[Ambiente] UDH carregada: {nome}  (índice {idx}, ano {ano})")
        return cls(estado_inicial=estado, nome=nome, **kwargs)

    @classmethod
    def from_dict(cls, estado: dict, **kwargs):
        """
        Cria o ambiente diretamente de um dicionário. Útil para testes rápidos.

        Exemplo:
            env = CidadeSustentavelEnv.from_dict({"CleanEnergy": 8.0, ...})
        """
        return cls(estado_inicial=estado, **kwargs)

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def reset(self, estado_inicial=None):
        """
        Reinicia o ambiente para o estado inicial.
        Se estado_inicial for fornecido, usa ele no lugar do padrão.
        """
        base = estado_inicial if estado_inicial else self.estado_inicial
        self.estado    = {k: float(v) for k, v in base.items()}
        self.turno     = 0
        self.historico = [self.estado.copy()]
        return self._estado_vetor()

    def step(self, acao: int):
        """
        Executa uma ação no ambiente.

        Parâmetros
        ----------
        acao : int  — índice da ação (0 a 7)

        Retorna
        -------
        proximo_estado : np.ndarray  — vetor com os 8 indicadores
        recompensa     : float       — recompensa calculada
        done           : bool        — True se chegou no turno 20
        info           : dict        — detalhes do turno
        """
        assert 0 <= acao < NUM_ACOES, f"Ação inválida: {acao}"

        # Aplica deltas com ruído gaussiano e clipa em [0, 10]
        for indicador, (intensidade, sinal) in TRANSICOES[acao].items():
            delta = calcular_delta(indicador, intensidade, sinal)
            ruido     = np.random.normal(0, self.ruido)
            novo_val  = self.estado[indicador] + delta + ruido
            self.estado[indicador] = float(np.clip(novo_val, 0.0, 10.0))

        self.turno += 1
        self.historico.append(self.estado.copy())

        recompensa = self._calcular_recompensa()
        done       = self.turno >= NUM_TURNOS

        info = {
            "turno":     self.turno,
            "estado":    self.estado.copy(),
            "acao_nome": ACOES[acao],
            "dimensoes": self._dimensoes(),
        }

        return self._estado_vetor(), recompensa, done, info

    # ------------------------------------------------------------------
    # Cálculos internos
    # ------------------------------------------------------------------

    def _calcular_recompensa(self) -> float:
        """Reward = 0.33*Environmental + 0.33*Economic + 0.34*Social"""
        d = self._dimensoes()
        return 0.33 * d["Environmental"] + 0.33 * d["Economic"] + 0.34 * d["Social"]

    def _dimensoes(self) -> dict:
        """Calcula as três dimensões de sustentabilidade."""
        s = self.estado
        return {
            "Environmental": round(0.4*s["CleanEnergy"] + 0.3*s["Mobility"]   + 0.3*s["AirQuality"], 4),
            "Economic":      round(0.6*s["Economy"]     + 0.4*s["Employment"], 4),
            "Social":        round(0.34*s["Housing"]    + 0.33*s["Education"] + 0.33*s["Healthcare"], 4),
        }

    def _estado_vetor(self) -> np.ndarray:
        """Retorna o estado como vetor numpy na ordem de INDICADORES."""
        return np.array([self.estado[k] for k in INDICADORES], dtype=np.float32)

    # ------------------------------------------------------------------
    # Utilitários de visualização
    # ------------------------------------------------------------------

    def estado_legivel(self):
        """Imprime o estado atual formatado no terminal."""
        print(f"\n{'='*48}")
        print(f"  {self.nome}  —  Turno {self.turno}/{NUM_TURNOS}")
        print(f"{'='*48}")
        for k, v in self.estado.items():
            # Trata NaN residual com segurança
            v_safe = 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            barra  = "█" * int(v_safe) + "░" * (10 - int(v_safe))
            print(f"  {k:<14} {barra}  {v_safe:.2f}")
        dims = self._dimensoes()
        print(f"{'─'*48}")
        for nome, val in dims.items():
            print(f"  {nome:<14} {val:.4f}")
        print(f"  {'Recompensa':<14} {self._calcular_recompensa():.4f}")
        print(f"{'='*48}")

    def comparar_estados(self, estado_antes: dict):
        """Mostra tabela de comparação antes × depois por indicador."""
        print(f"\n{'='*55}")
        print(f"  Comparação: antes × depois ({self.nome})")
        print(f"{'='*55}")
        print(f"  {'Indicador':<14} {'Antes':>7}  {'Depois':>7}  {'Δ':>7}")
        print(f"  {'─'*44}")
        for k in INDICADORES:
            antes  = estado_antes.get(k, 0.0)
            depois = self.estado[k]
            delta  = depois - antes
            sinal  = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
            print(f"  {k:<14} {antes:>7.2f}  {depois:>7.2f}  {sinal} {abs(delta):>5.2f}")
        print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def num_acoes(self) -> int:
        return NUM_ACOES

    @property
    def num_estados(self) -> int:
        return len(INDICADORES)


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste do Ambiente MDP ===\n")

    # Treino: carrega média real das UDHs gerada pelo eda.py
    if os.path.exists(JSON_PATH_PADRAO):
        env = CidadeSustentavelEnv.from_json(ruido=0.05, seed=42)
    else:
        print(f"⚠️  {JSON_PATH_PADRAO} não encontrado.")
        print("   Execute primeiro: python eda.py ../data/udh_sp.xlsx\n")
        env = CidadeSustentavelEnv(ruido=0.05, seed=42)

    print("\nEstado inicial:")
    env.estado_legivel()
    estado_antes = env.estado.copy()

    # Episódio com ações aleatórias
    done = False
    recompensa_total = 0.0
    while not done:
        acao = np.random.randint(0, NUM_ACOES)
        _, recompensa, done, info = env.step(acao)
        recompensa_total += recompensa
        print(f"Turno {info['turno']:>2} | {info['acao_nome']:<35} | Recompensa: {recompensa:.4f}")

    print(f"\nRecompensa total acumulada: {recompensa_total:.4f}")
    print("\nEstado final:")
    env.estado_legivel()
    env.comparar_estados(estado_antes)

    # Avaliação: carrega UDH real do CSV
    if os.path.exists(CSV_PATH_PADRAO):
        print("\n\n=== Teste: avaliação com UDH real (linha 0) ===")
        env2 = CidadeSustentavelEnv.from_csv(idx=0, ruido=0.05, seed=7)
        env2.estado_legivel()
    else:
        print(f"\n⚠️  {CSV_PATH_PADRAO} não encontrado — pulando teste de avaliação.")