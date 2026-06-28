"""
eda.py
------
Análise Exploratória dos Dados UDH — Região Metropolitana de São Paulo.
Projeto: Aprendizado por Reforço para Cidades Sustentáveis (ODS 11)

Entrada : arquivo .xlsx da base UDH (Atlas do Desenvolvimento Humano, PNUD)
Saída   : udh_normalizado.csv  — indicadores normalizados para escala 0-10
          fig1_histogramas.png
          fig2_correlacao.png
          fig3_boxplots_ano.png
          fig4_radar_2000_2010.png
          fig5_scatter_economy_healthcare.png
          fig6_heatmap_municipios.png

Mapeamento de indicadores:
  CleanEnergy  <- T_LUZ            (% domicílios com energia elétrica)
  Mobility     <- T_OCUPDESLOC_1   (% deslocamento >1h — invertido, só 2010)
  AirQuality   <- média(T_LIXO, T_BANAGUA)
  Economy      <- RDPC             (renda domiciliar per capita)
  Employment   <- P_FORMAL         (% emprego formal)
  Housing      <- média(T_AGUA, T_BANAGUA)
  Education    <- IDHM_E           (índice de educação, ×100)
  Healthcare   <- IDHM_L           (índice de longevidade, ×100)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configurações de visualização
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# Colunas utilizadas
# ---------------------------------------------------------------------------
META_COLS = ["Cod_ID", "NOME_UDH", "CODMUN6", "NOME_MUN", "ANO", "IDHM"]

RAW_COLS = [
    "T_LUZ",           # CleanEnergy
    "T_OCUPDESLOC_1",  # Mobility (só 2010)
    "T_LIXO",          # AirQuality (componente 1)
    "T_BANAGUA",       # AirQuality (componente 2) + Housing (componente 2)
    "RDPC",            # Economy
    "P_FORMAL",        # Employment
    "T_AGUA",          # Housing (componente 1)
    "IDHM_E",          # Education
    "IDHM_L",          # Healthcare
]

RAW_INDICATORS = [
    "RAW_CleanEnergy", "RAW_Mobility",   "RAW_AirQuality",
    "RAW_Economy",     "RAW_Employment", "RAW_Housing",
    "RAW_Education",   "RAW_Healthcare",
]

MDP_INDICATORS = [c.replace("RAW_", "") for c in RAW_INDICATORS]


# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
def carregar_dados(xlsx_path: str) -> pd.DataFrame:
    """Carrega o .xlsx e seleciona as colunas relevantes."""
    df_raw = pd.read_excel(xlsx_path, sheet_name=0)

    print(f"Shape bruto       : {df_raw.shape}")
    print(f"Anos disponíveis  : {sorted(df_raw['ANO'].unique())}")
    print(f"Municípios        : {df_raw['NOME_MUN'].nunique()}")
    print(f"UDHs únicas       : {df_raw['NOME_UDH'].nunique()}")

    df = df_raw[META_COLS + RAW_COLS].copy()
    print(f"Shape após seleção: {df.shape}\n")
    return df


# ---------------------------------------------------------------------------
# 2. Tratar valores nulos
# ---------------------------------------------------------------------------
def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa mediana por ano nos indicadores (exceto Mobility, só em 2010)."""
    print("Nulos antes do tratamento:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    cols_impute = ["T_LUZ", "T_LIXO", "T_BANAGUA", "RDPC",
                   "P_FORMAL", "T_AGUA", "IDHM_E", "IDHM_L", "IDHM"]

    for col in cols_impute:
        medians = df.groupby("ANO")[col].transform("median")
        n = df[col].isnull().sum()
        df[col] = df[col].fillna(medians)
        if n:
            print(f"  {col}: {n} nulos → mediana por ano")

    print("\nNulos restantes (T_OCUPDESLOC_1 em 2000 é esperado):")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print()
    return df


# ---------------------------------------------------------------------------
# 3. Construir indicadores compostos
# ---------------------------------------------------------------------------
def construir_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as colunas RAW_ para cada indicador do MDP."""

    # Mobility: T_OCUPDESLOC_1 é negativo → inverter (100 - x)
    df["RAW_Mobility"]    = np.where(df["T_OCUPDESLOC_1"].notna(),
                                     100 - df["T_OCUPDESLOC_1"], np.nan)

    # AirQuality: média de coleta de lixo e saneamento
    df["RAW_AirQuality"]  = (df["T_LIXO"] + df["T_BANAGUA"]) / 2

    # Housing: média de abastecimento de água e banheiro com água
    df["RAW_Housing"]     = (df["T_AGUA"] + df["T_BANAGUA"]) / 2

    # Demais indicadores diretos
    df["RAW_CleanEnergy"] = df["T_LUZ"]
    df["RAW_Economy"]     = df["RDPC"]
    df["RAW_Employment"]  = df["P_FORMAL"]
    df["RAW_Education"]   = df["IDHM_E"] * 100   # IDHM já em 0-1 → escala %
    df["RAW_Healthcare"]  = df["IDHM_L"] * 100

    print("Indicadores compostos criados:")
    for raw, ind in zip(RAW_INDICATORS, MDP_INDICATORS):
        print(f"  {ind:15s} <- {raw}")
    print()
    return df


# ---------------------------------------------------------------------------
# 4. Normalização para escala 0–10
# Economy usa log antes do min-max (distribuição muito assimétrica com outliers).
# Demais indicadores usam min-max linear padrão.
# ---------------------------------------------------------------------------
def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada indicador RAW_ para [0, 10]."""
    df_norm = df[META_COLS + RAW_INDICATORS].copy()

    print("Normalização (0–10):")
    for raw, ind in zip(RAW_INDICATORS, MDP_INDICATORS):
        col_data = df_norm[raw].dropna()

        if ind == "Economy":
            # Log-normalização: suaviza outliers de RDPC
            log_vals      = np.log1p(df_norm[raw])
            lmin, lmax    = np.log1p(col_data).min(), np.log1p(col_data).max()
            df_norm[ind]  = ((log_vals - lmin) / (lmax - lmin)) * 10
            print(f"  {ind:15s}: log([{col_data.min():.2f}, {col_data.max():.2f}]) → [0, 10]  (log-norm)")
        else:
            vmin, vmax    = col_data.min(), col_data.max()
            df_norm[ind]  = ((df_norm[raw] - vmin) / (vmax - vmin)) * 10
            print(f"  {ind:15s}: [{vmin:.2f}, {vmax:.2f}] → [0, 10]")

    df_out = df_norm[META_COLS + MDP_INDICATORS].copy()
    print()
    print("Estatísticas dos indicadores normalizados:")
    print(df_out[MDP_INDICATORS].describe().round(3).to_string())
    print()
    return df_out


# ---------------------------------------------------------------------------
# 5. Salvar CSV e gráficos em data/
# ---------------------------------------------------------------------------
def salvar_csv(df_out: pd.DataFrame, out_path: str = "../data/udh_normalizado.csv"):
    """Salva o DataFrame normalizado em CSV na pasta data/."""
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Salvo: {out_path}  ({df_out.shape[0]} linhas, {df_out.shape[1]} colunas)")


# ---------------------------------------------------------------------------
# 6. Gráficos
# ---------------------------------------------------------------------------
def plot_histogramas(df_out: pd.DataFrame):
    """Histogramas dos 8 indicadores normalizados."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = sns.color_palette("muted", len(MDP_INDICATORS))

    for i, (ind, color) in enumerate(zip(MDP_INDICATORS, colors)):
        data = df_out[ind].dropna()
        axes[i].hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
        axes[i].set_title(ind, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Valor (0–10)")
        axes[i].set_ylabel("Frequência")
        axes[i].axvline(data.mean(), color="black", linestyle="--", linewidth=1.2,
                        label=f"Média: {data.mean():.2f}")
        axes[i].legend(fontsize=9)

    plt.suptitle("Distribuição dos Indicadores do MDP\n(UDH São Paulo, 2000+2010)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("../data/fig1_histogramas.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig1_histogramas.png")


def plot_correlacao(df_out: pd.DataFrame):
    """Mapa de calor de correlação entre indicadores."""
    corr = df_out[MDP_INDICATORS].dropna().corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("Correlação entre Indicadores do MDP\n(UDH São Paulo)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../data/fig2_correlacao.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig2_correlacao.png")


def plot_boxplots_ano(df_out: pd.DataFrame):
    """Boxplots comparando 2000 vs 2010 para cada indicador."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, ind in enumerate(MDP_INDICATORS):
        ax = axes[i]
        data_2000 = df_out[df_out["ANO"] == 2000][ind].dropna()
        data_2010 = df_out[df_out["ANO"] == 2010][ind].dropna()

        bplot = ax.boxplot(
            [data_2000, data_2010],
            tick_labels=["2000", "2010"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bplot["boxes"][0].set_facecolor("#AEC6CF")
        if len(bplot["boxes"]) > 1:
            bplot["boxes"][1].set_facecolor("#FFD700")

        title = ind if ind != "Mobility" else f"{ind}\n(apenas 2010)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Valor (0–10)")

    plt.suptitle("Evolução dos Indicadores: 2000 vs 2010\n(UDH São Paulo)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("../data/fig3_boxplots_ano.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig3_boxplots_ano.png")


def plot_radar(df_out: pd.DataFrame):
    """Radar chart comparando perfil médio 2000 vs 2010."""
    # Mobility excluído pois só existe em 2010
    indicadores_radar = [i for i in MDP_INDICATORS if i != "Mobility"]
    medias_2000 = df_out[df_out["ANO"] == 2000][indicadores_radar].mean()
    medias_2010 = df_out[df_out["ANO"] == 2010][indicadores_radar].mean()

    N = len(indicadores_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for medias, label, color in [
        (medias_2000, "2000", "#AEC6CF"),
        (medias_2010, "2010", "#FFD700"),
    ]:
        vals = medias.values.tolist() + [medias.values[0]]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.2, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicadores_radar, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_title("Perfil Médio dos Indicadores\n2000 vs 2010 (UDH São Paulo)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig("../data/fig4_radar_2000_2010.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig4_radar_2000_2010.png")


def plot_scatter_economy_healthcare(df_out: pd.DataFrame):
    """Scatter Economy × Healthcare para visualizar correlação renda-longevidade."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ano, color, marker in [(2000, "#AEC6CF", "o"), (2010, "#FFD700", "s")]:
        sub = df_out[df_out["ANO"] == ano]
        ax.scatter(sub["Economy"], sub["Healthcare"],
                   alpha=0.35, s=15, color=color, marker=marker, label=str(ano))

    ax.set_xlabel("Economy (normalizado, 0–10)", fontsize=12)
    ax.set_ylabel("Healthcare (normalizado, 0–10)", fontsize=12)
    ax.set_title("Economy × Healthcare por UDH\n(correlação entre renda e longevidade)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Ano", fontsize=11)
    plt.tight_layout()
    plt.savefig("../data/fig5_scatter_economy_healthcare.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig5_scatter_economy_healthcare.png")


def plot_heatmap_municipios(df_out: pd.DataFrame):
    """Heatmap do perfil médio dos 15 municípios com maior Economy."""
    inds_sem_mobility = [i for i in MDP_INDICATORS if i != "Mobility"]

    mun_mean = (
        df_out.groupby("NOME_MUN")[MDP_INDICATORS].mean()
        .dropna(subset=inds_sem_mobility)
    )
    top15 = mun_mean.sort_values("Economy", ascending=False).head(15)[inds_sem_mobility]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(top15, annot=True, fmt=".1f", cmap="YlGnBu",
                vmin=0, vmax=10, linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Perfil Médio por Município (Top 15 por Economy)\n(UDH São Paulo, 2000+2010)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig("../data/fig6_heatmap_municipios.png", bbox_inches="tight")
    plt.show()
    print("Salvo: ../data/fig6_heatmap_municipios.png")


# ---------------------------------------------------------------------------
# 7. Resumo final e cálculo da média para o MDP
# ---------------------------------------------------------------------------
def resumo_final(df_out: pd.DataFrame) -> dict:
    """
    Imprime resumo e retorna o estado inicial médio para o MDP.
    Usa dados de 2010 para Mobility (não disponível em 2000).
    """
    print("=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"Registros totais : {len(df_out)}")
    print(f"UDHs únicas      : {df_out['NOME_UDH'].nunique()}")
    print(f"Municípios       : {df_out['NOME_MUN'].nunique()}")
    print(f"Anos             : {sorted(df_out['ANO'].unique())}")
    print()
    print("Médias dos indicadores normalizados (0–10):")
    print(df_out[MDP_INDICATORS].mean().round(3).to_string())
    print()
    print("⚠️  Observações para o MDP:")
    print("  • Economy tem distribuição assimétrica (RDPC com outliers).")
    print("  • Mobility só disponível em 2010 → inicializado com mediana 2010.")
    print("=" * 60)

    # Estado inicial médio para o ambiente MDP
    # Mobility: mediana de 2010 (único ano disponível)
    mobility_2010 = df_out[df_out["ANO"] == 2010]["Mobility"].median()

    estado_medio = {}
    for ind in MDP_INDICATORS:
        if ind == "Mobility":
            estado_medio[ind] = round(float(mobility_2010), 4)
        else:
            estado_medio[ind] = round(float(df_out[ind].mean()), 4)

    print("\nEstado inicial médio para o ambiente MDP:")
    for k, v in estado_medio.items():
        print(f"  {k:15s}: {v}")

    return estado_medio


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Caminho do arquivo .xlsx — passar como argumento ou editar aqui
    XLSX_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/udh_sp.xlsx"

    print(f"Carregando: {XLSX_PATH}\n")

    df        = carregar_dados(XLSX_PATH)
    df        = tratar_nulos(df)
    df        = construir_indicadores(df)
    df_out    = normalizar(df)

    salvar_csv(df_out)

    plot_histogramas(df_out)
    plot_correlacao(df_out)
    plot_boxplots_ano(df_out)
    plot_radar(df_out)
    plot_scatter_economy_healthcare(df_out)
    plot_heatmap_municipios(df_out)

    estado_medio = resumo_final(df_out)

    # Exporta o estado médio como JSON para o ambiente carregar
    import json
    with open("../data/estado_inicial_medio.json", "w") as f:
        json.dump(estado_medio, f, indent=2)
    print("\nSalvo: ../data/estado_inicial_medio.json  (usado pelo ambiente.py)")