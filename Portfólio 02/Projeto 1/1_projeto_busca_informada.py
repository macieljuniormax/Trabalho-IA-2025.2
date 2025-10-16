"""
Projeto 1 — Busca Informada (A* e Greedy) no 8-puzzle
Autor: Maciel Ferreira Custódio Júnior
Matrícula: 190100087
Disciplina: FGA0221 – Inteligência Artificial

Descrição
---------
Resolvendo o quebra-cabeça deslizante (quebra-cabeça 3x3) usando duas buscas informadas:
- A* (expande por custo f = g + h)
- Greedy Best-First (expande por custo f = h)

Heurísticas disponíveis:
- misplaced  : número de peças fora do lugar
- manhattan  : soma das distâncias Manhattan das peças até a posição objetivo
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set
import heapq

State = Tuple[int, ...] # estado do quebra-cabeça como tupla de 9 inteiros
Coordinate = Tuple[int, int] # (linha, coluna) no tabuleiro 3x3

GOAL: State = (1,2,3,4,5,6,7,8,0)  # objetivo padrão

# -----------------------------
# Utilitários
# -----------------------------
def index_to_row_col(index: int) -> Coordinate:
    """Converte índice [0..8] em (linha, coluna) no tabuleiro 3x3."""
    return divmod(index, 3)

def row_col_to_index(row: int, col: int) -> int:
    """Converte (linha, coluna) em índice [0..8] no tabuleiro 3x3."""
    return row * 3 + col

def print_board(state: State) -> str:
    """
    Retorna uma string formatada representando o estado 3x3 do tabuleiro.
    Substitui o valor 0 por '·' para indicar o espaço vazio.
    """
    rows = []
    for r in range(3):
        row = state[r*3:(r+1)*3]
        rows.append(' '.join('{:>2}'.format(x if x != 0 else '·') for x in row))
    return '\n'.join(rows)

def is_solvable(state: State) -> bool:
    """
    Verifica se o quebra-cabeça 3x3 é solúvel.
    Regra: contagem de inversões (ignorando 0) deve ser par.
    """
    arr = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions % 2 == 0

# -----------------------------
# Heurísticas
# -----------------------------
def h_misplaced(state: State, goal: State = GOAL) -> int:
    """Número de peças fora do lugar"""

    return sum(1 for i, v in enumerate(state) if v != 0 and v != goal[i])

def h_manhattan(state: State, goal: State = GOAL) -> int:
    """Soma das distâncias Manhattan de cada peça à posição alvo"""

    # cria um dicionário que mapeia cada número a sua posição correta
    goal_pos: Dict[int, Coordinate] = {val: index_to_row_col(i) for i, val in enumerate(goal)}

    dist = 0

    for i, value in enumerate(state):
        if value == 0:
            continue
        row, col = index_to_row_col(i) # posição atual
        goal_row, goal_col = goal_pos[value] # posição alvo
        dist += abs(row - goal_row) + abs(col - goal_col)  # distância Manhattan
    return dist

# -----------------------------
# Estruturas de busca
# -----------------------------
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    node: "Node" = field(compare=False)

class Node:
    """Nó da busca no espaço de estados do quebra-cabeça 3x3."""

    def __init__(self, state: State, parent: Optional["Node"], g: int, h: int) -> None:
        self.state = state
        self.parent = parent
        self.g = g  # custo acumulado (número de movimentos)
        self.h = h  # heurística

    @property
    def f(self) -> int:
        return self.g + self.h

    def path(self) -> List[State]:
        """Reconstrói o caminho percorrido."""
        node: Optional["Node"] = self
        rev: List[State] = []
        while node is not None:
            rev.append(node.state)
            node = node.parent
        return list(reversed(rev))

def neighbors(state: State) -> Iterable[State]:
    """
    Gera estados vizinhos (possíveis jogadas) movendo o espaço (0) para cima/baixo/esq/dir.
    """
    zero_index = state.index(0) # posição do espaço vazio
    zero_row, zero_col = index_to_row_col(zero_index) # converte para (linha, coluna)
    moves = [(-1,0),(1,0),(0,-1),(0,1)] # movimentos possíveis: cima, baixo, esquerda, direita
    for move_row, move_col in moves:
        new_row, new_col = zero_row + move_row, zero_col + move_col # nova posição do espaço vazio
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = row_col_to_index(new_row, new_col) # converte de volta para índice
            new_state = list(state)  # cria uma nova lista mutável

            # troca o espaço (0) com a peça na nova posição
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]

            yield tuple(new_state) # retorna como tupla imutável

# -----------------------------
# Algoritmos de busca
# -----------------------------
def astar(start: State,
          goal: State,
          heuristic: Callable[[State, State], int]
    ) -> Tuple[Optional[Node], int]:
    """
    A* no quebra-cabeça 3x3.
    Retorna o nó objetivo e nós expandidos.
    """

    frontier: List[PrioritizedItem] = [] # fila de prioridade ordenada por f = g + h
    start_node = Node(start, None, g=0, h=heuristic(start, goal)) # nó inicial
    heapq.heappush(frontier, PrioritizedItem(start_node.f, start_node)) # adiciona à frontier

    best_g: Dict[State, int] = {start: 0} # melhor custo g encontrado para cada estado
    expanded = 0 # contador de nós expandidos
    visited: Set[State] = set() # estados já visitados

    # Loop principal (enquanto houver nós no frontier)
    while frontier:
        current = heapq.heappop(frontier).node # nó com menor f = g + h

        if current.state in visited: # ignora estados já visitados
            continue

        visited.add(current.state) # marca estado como visitado
        expanded += 1

        # Verifica se alcançou o objetivo
        if current.state == goal:
            return current, expanded

        # Expande os vizinhos
        for neighbor in neighbors(current.state):
            new_g = current.g + 1
            if neighbor not in best_g or new_g < best_g[neighbor]:
                best_g[neighbor] = new_g # atualiza melhor g para este estado
                neighbor_node = Node(neighbor, current, g=new_g, h=heuristic(neighbor, goal)) # cria o nó vizinho
                heapq.heappush(frontier, PrioritizedItem(neighbor_node.f, neighbor_node)) # adiciona à frontier

    return None, expanded

def greedy_best_first(start: State,
                      goal: State,
                      heuristic: Callable[[State, State], int]
    ) -> Tuple[Optional[Node], int]:
    """
    Implementação do algoritmo Greedy Best-First Search para o quebra-cabeça 3x3.
    Prioriza apenas o valor da heurística (h), não o custo acumulado (g).

    Retorna o nó objetivo e nós expandidos.
    Pode encontrar soluções mais rápidas, mas não necessariamente ótimas.
    """

    frontier: List[PrioritizedItem] = []
    start_node = Node(start, None, g=0, h=heuristic(start, goal))
    heapq.heappush(frontier, PrioritizedItem(start_node.h, start_node))

    expanded = 0
    visited: Set[State] = set()

    while frontier:
        current = heapq.heappop(frontier).node

        if current.state in visited:
            continue

        visited.add(current.state)
        expanded += 1

        if current.state == goal:
            return current, expanded

        for neighbor in neighbors(current.state):
            # Ignora estados já visitados
            if neighbor in visited:
                continue

            neighbor_node = Node(neighbor, current, g=current.g + 1, h=heuristic(neighbor, goal))
            heapq.heappush(frontier, PrioritizedItem(neighbor_node.h, neighbor_node))

    return None, expanded

# -----------------------------
# Runner / Menu interativo
# -----------------------------
DEFAULT_START: State = (1, 2, 3, 4, 0, 6, 7, 5, 8)

def parse_state_string(s: str) -> State:
    """
    Converte uma string como "1,2,3,4,0,6,7,5,8" para uma tupla de 9 inteiros (State).
    Valida quantidade (9) e conteúdo (dígitos 0..8 sem repetição).
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 9:
        raise ValueError("Estado deve ter 9 números (0..8) separados por vírgula.")
    try:
        vals = tuple(int(x) for x in parts)
    except ValueError:
        raise ValueError("Todos os valores devem ser inteiros de 0 a 8.")
    if sorted(vals) != list(range(9)):
        raise ValueError("Estado deve conter exatamente os dígitos 0..8, sem repetição.")
    return vals

def read_state_from_input(prompt: str = "Digite o estado inicial (ex: 1,2,3,4,0,6,7,5,8): ") -> State:
    s = input(prompt).strip()
    if not s:
        return DEFAULT_START
    try:
        st = parse_state_string(s)
        return st
    except Exception as e:
        print(f"Entrada inválida: {e}")
        return read_state_from_input(prompt)

def choose_option(title: str, options: list[str], default_idx: int = 0) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        mark = " (padrão)" if i-1 == default_idx else ""
        print(f"  {i}. {opt}{mark}")
    choice = input("Escolha uma opção: ").strip()
    if not choice:
        return options[default_idx]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except:
        pass
    print("Opção inválida. Tente novamente.")
    return choose_option(title, options, default_idx)

def interactive_menu() -> None:
    print("======= QUEBRA-CABEÇA — BUSCA INFORMADA (A* e Greedy) =======")

    # Escolher estratégia
    strategy = choose_option("Estratégia de busca", ["astar", "greedy"], default_idx=0)

    # Escolher heurística
    heuristic_name = choose_option("Heurística", ["manhattan", "misplaced"], default_idx=0)

    # 3) Estado inicial
    start = read_state_from_input()

    # 4) Valida se possui solução
    if not is_solvable(start):
        print("\nATENÇÃO: este estado não é solúvel (paridade de inversões ímpar).")
        print("Estado informado:\n" + print_board(start))
        retry = input("Deseja tentar outro estado? (s/N): ").strip().lower()
        if retry == "s":
            return interactive_menu()
        # Se insistir, seguimos mesmo assim (algoritmo não encontrará solução)
        print("Prosseguindo mesmo assim...\n")

    # Rodar
    print("\nResumo da execução:")
    print(f"  Estratégia : {strategy}")
    print(f"  Heurística : {heuristic_name}")
    print("  Inicial:\n" + print_board(start))
    print("  Objetivo:\n" + print_board(GOAL))
    print()

    # Mapeia heurística
    h = {"manhattan": h_manhattan, "misplaced": h_misplaced}[heuristic_name]

    if strategy == "greedy":
        result, expanded = greedy_best_first(start, GOAL, h)
        label = "Greedy Best-First"
    else:
        result, expanded = astar(start, GOAL, h)
        label = "A*"

    print(f"Estratégia : {label}")
    print(f"Heurística : {heuristic_name}")

    if result is None:
        print(f"Nenhuma solução encontrada. Nós expandidos: {expanded}")
        return

    path = result.path()
    print(f"Solução em {len(path)-1} movimentos. Nós expandidos: {expanded}\n")
    for i, st in enumerate(path):
        print(f"Passo {i}:\n{print_board(st)}\n")

if __name__ == "__main__":
    interactive_menu()
