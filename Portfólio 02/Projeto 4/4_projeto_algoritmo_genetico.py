"""
Projeto 4 — Algoritmo Genético (GA) para o Problema da Mochila 0/1
Autor: Maciel Ferreira Custódio Júnior
Matrícula: 190100087
Disciplina: FGA0221 – Inteligência Artificial

Objetivo
---------
Simular um processo evolutivo para resolver o problema clássico da Mochila 0/1 (Knapsack),
usando o paradigma de busca populacional e heurística estocástica dos Algoritmos Genéticos.

Descrição
---------
- Cada indivíduo representa uma solução candidata (vetor binário), onde 1 significa
  que o item está incluído na mochila e 0 que foi excluído.
- A função de fitness avalia o valor total dos itens selecionados, penalizando
  soluções que ultrapassam a capacidade máxima.
- A população evolui por várias gerações, aplicando operadores de seleção,
  crossover, mutação e elitismo, até convergir para uma solução de alto valor.

Características:
- Seleção: torneio
- Crossover: 1 ponto
- Mutação: bit-flip (troca aleatória de 0 ↔ 1)
- Elitismo: mantém o melhor indivíduo de cada geração
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random

# -----------------------------
# Estruturas básicas
# -----------------------------
@dataclass(frozen=True)
class Item:
    """Representa um item com peso e valor."""
    weight: int
    value: int

# Um indivíduo é uma lista binária (0 = não pega o item, 1 = pega)
Individual = List[int]

# -----------------------------
# Funções auxiliares
# -----------------------------
def random_individual(n: int) -> Individual:
    """Cria um indivíduo aleatório de tamanho n (solução inicial)."""
    return [random.randint(0, 1) for _ in range(n)]

def decode(ind: Individual, items: List[Item]) -> Tuple[int, int]:
    """
    Calcula o (peso_total, valor_total) do indivíduo.
    Soma o peso e valor de todos os itens selecionados (bit = 1).
    """
    total_weight = sum(bit * item.weight for bit, item in zip(ind, items))
    total_value  = sum(bit * item.value  for bit, item in zip(ind, items))
    return total_weight, total_value

def fitness(ind: Individual, items: List[Item], capacity: int, penalty: int = 10) -> int:
    """
    Função de avaliação (fitness):
    - Se o peso total for menor ou igual à capacidade, retorna o valor total.
    - Se exceder, aplica uma penalidade proporcional ao excesso.
    """
    weight, value = decode(ind, items)
    excess = max(0, weight - capacity)
    score = value - penalty * excess
    return max(0, score)  # fitness nunca é negativo

# -----------------------------
# Operadores genéticos
# -----------------------------
def tournament_selection(pop: List[Individual], k: int, items: List[Item], capacity: int) -> Individual:
    """
    Seleção por torneio:
    - Sorteia k indivíduos da população.
    - Retorna o melhor deles (maior fitness).
    """
    contenders = random.sample(pop, k)
    return max(contenders, key=lambda ind: fitness(ind, items, capacity))

def one_point_crossover(a: Individual, b: Individual) -> Tuple[Individual, Individual]:
    """
    Crossover de um ponto:
    - Divide os pais em um ponto aleatório e troca as partes.
    - Retorna dois filhos.
    """
    n = len(a)
    if n < 2:
        return a[:], b[:]  # sem corte possível
    cut = random.randint(1, n - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]

def mutate(ind: Individual, pm: float) -> None:
    """
    Mutação bit-flip:
    - Cada gene (bit) tem chance pm de ser invertido (0 ↔ 1).
    """
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] ^= 1  # operador XOR com 1 inverte o bit

# -----------------------------
# Algoritmo Genético principal
# -----------------------------
def genetic_algorithm(
    items: List[Item],
    capacity: int,
    pop_size: int = 80,
    generations: int = 200,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = None,
    seed: int | None = 42,
) -> Tuple[Individual, int]:
    """
    Implementação do Algoritmo Genético para o problema da mochila.

    Parâmetros:
    - items: lista de itens disponíveis.
    - capacity: peso máximo da mochila.
    - pop_size: número de indivíduos por geração.
    - generations: número de iterações (gerações evolutivas).
    - tournament_k: tamanho do torneio para seleção.
    - crossover_rate: probabilidade de realizar crossover.
    - mutation_rate: probabilidade de mutação por gene (default = 1/n).
    - seed: inicialização para reprodutibilidade.

    Retorna:
    - (melhor_indivíduo_encontrado, fitness_desse_indivíduo)
    """
    # Define semente aleatória para resultados reproduzíveis
    if seed is not None:
        random.seed(seed)

    n = len(items)
    if mutation_rate is None:
        mutation_rate = 1.0 / max(1, n)

    # População inicial gerada aleatoriamente
    population = [random_individual(n) for _ in range(pop_size)]

    # Avalia o melhor indivíduo inicial
    best = max(population, key=lambda x: fitness(x, items, capacity))
    best_fit = fitness(best, items, capacity)

    # Loop de evolução
    for _ in range(generations):
        new_population: List[Individual] = []

        # Mantém o melhor indivíduo (elitismo)
        new_population.append(best[:])

        # Gera novos indivíduos até preencher a população
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, tournament_k, items, capacity)
            parent2 = tournament_selection(population, tournament_k, items, capacity)

            # Crossover com probabilidade dada
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Aplica mutação em ambos os filhos
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Garante que a população não exceda o tamanho esperado
        population = new_population[:pop_size]

        # Atualiza melhor indivíduo global
        gen_best = max(population, key=lambda x: fitness(x, items, capacity))
        gen_fit = fitness(gen_best, items, capacity)
        if gen_fit > best_fit:
            best, best_fit = gen_best[:], gen_fit

    return best, best_fit

# -----------------------------
# Execução de exemplo
# -----------------------------
def run_demo() -> None:
    """
    Exemplo simples de execução do algoritmo:
    - 15 itens com pesos e valores variados.
    - Capacidade de 25 unidades.
    """
    items = [
        Item(12,  4), Item( 2,  2), Item( 1,  1), Item( 1,  2), Item( 4, 10),
        Item( 1,  2), Item( 2,  1), Item( 1,  1), Item(10, 12), Item( 3,  6),
        Item( 6,  7), Item( 4,  5), Item( 5,  8), Item( 7,  9), Item( 3,  5),
    ]
    capacity = 25

    best, best_fit = genetic_algorithm(
        items=items,
        capacity=capacity,
        pop_size=80,
        generations=200,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=None,  # usa 1/n padrão
        seed=42,
    )

    # Decodifica a solução encontrada
    weight, value = decode(best, items)
    chosen = [i for i, bit in enumerate(best) if bit == 1]

    print("=== ALGORITMO GENÉTICO — MOCHILA 0/1 ===")
    print(f"Itens escolhidos (índices): {chosen}")
    print(f"Peso total: {weight} | Valor total: {value} | Fitness: {best_fit}")
    print(f"Capacidade máxima: {capacity}")
    print("Solução binária:", best)

# Execução direta
if __name__ == "__main__":
    run_demo()
