"""
Projeto 3 — Busca COMPLEXA (IDA*) no N-puzzle (8 / 15-puzzle)
Autor: Maciel Ferreira Custódio Júnior
Matrícula: 190100087
Disciplina: FGA0221 – Inteligência Artificial

Descrição
---------
Resolvendo o quebra-cabeça deslizante (3x3 ou 4x4) usando busca complexa IDA*:
- IDA* (Iterative Deepening A*): faz aprofundamentos iterativos em limites de f(n) = g(n) + h(n).
- Heurísticas implementadas:
    * manhattan        : soma das distâncias Manhattan de cada peça até a posição objetivo
    * linear_conflict  : manhattan + 2 * (conflitos lineares em linhas/colunas)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set
import math

# -----------------------------
# Tipos e Constantes
# -----------------------------
State = Tuple[int, ...]         # Estado do tabuleiro (tupla imutável com N*N inteiros)
Coordinate = Tuple[int, int]    # Coordenada (linha, coluna)

# -----------------------------
# Utilitários gerais (NxN)
# -----------------------------
def side_len(state: State) -> int:
    """
    Retorna N tal que N*N == len(state).
    Ex.: len=9 -> N=3; len=16 -> N=4.
    """
    n2 = len(state)
    n = int(math.isqrt(n2))
    if n * n != n2:
        raise ValueError("O estado deve ter tamanho N*N (quadrado perfeito).")
    return n


def goal_of(n: int) -> State:
    """Retorna o estado objetivo padrão (1..N*N-1, 0)."""
    # Monta [1,2,...,N*N-1,0] e converte para tupla imutável
    return tuple(list(range(1, n*n)) + [0])


def index_to_row_col(index: int, n: int) -> Coordinate:
    """Converte índice [0..N*N-1] em (linha, coluna)."""
    return divmod(index, n)


def row_col_to_index(row: int, col: int, n: int) -> int:
    """Converte (linha, coluna) em índice linear [0..N*N-1]."""
    return row * n + col


def print_board(state: State) -> str:
    """
    Retorna uma string legível representando o tabuleiro NxN.
    Substitui o valor 0 por '·' (vazio) e alinha as colunas.
    """
    n = side_len(state)
    width = len(str(n*n - 1))          # largura mínima para alinhar números (15 -> 2 dígitos, etc.)
    lines: List[str] = []
    for row_index in range(n):
        row_values = state[row_index*n:(row_index+1)*n]  # fatia a linha corrente
        # Para cada célula, imprime número alinhado; se 0 (vazio), imprime '·'
        line = ' '.join(
            f"{value:>{width}}" if value != 0 else " "*(width-1) + "·"
            for value in row_values
        )
        lines.append(line)
    return '\n'.join(lines)

# -----------------------------
# Verificação de Solubilidade
# -----------------------------
def is_solvable(state: State) -> bool:
    """
    Verifica se o estado é solúvel.
    - N ímpar: número de inversões (ignorando 0) deve ser par.
    - N par:   regra depende da linha do 0 (contada de baixo para cima).
    """
    n = side_len(state)
    # Remove o zero e conta inversões no restante
    arr = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1

    if n % 2 == 1:
        # Em tabuleiro ímpar (3x3), deve ser par
        return inversions % 2 == 0
    else:
        # Em tabuleiro par (4x4), depende da linha do 0 a partir da base
        zero_index = state.index(0)
        zero_row, _ = index_to_row_col(zero_index, n)
        row_from_bottom = n - zero_row
        if row_from_bottom % 2 == 1:
            return inversions % 2 == 0
        else:
            return inversions % 2 == 1

# -----------------------------
# Geração de Vizinhos
# -----------------------------
def neighbors(state: State) -> Iterable[State]:
    """
    Gera estados vizinhos movendo o espaço (0) para cima/baixo/esquerda/direita.
    """
    n = side_len(state)
    zero_index = state.index(0)
    zero_row, zero_col = index_to_row_col(zero_index, n)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # linha, coluna: cima, baixo, esq, dir

    for move_row, move_col in moves:
        new_row, new_col = zero_row + move_row, zero_col + move_col
        if 0 <= new_row < n and 0 <= new_col < n:
            # Índice linear do destino do 0
            new_index = row_col_to_index(new_row, new_col, n)
            # Clona estado em lista (para trocar valores) e faz o swap 0 <-> peça
            new_state = list(state)
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            # Retorna tupla imutável (hashable p/ sets e dicionários)
            yield tuple(new_state)

# -----------------------------
# Heurísticas (admissíveis)
# -----------------------------
def h_manhattan(state: State, goal: State) -> int:
    """
    Soma das distâncias Manhattan de cada peça até sua posição no objetivo.
    - Para cada peça v != 0, soma |linha_atual - linha_objetivo| + |col_atual - col_objetivo|.
    """
    n = side_len(state)
    # Mapa: valor -> (linha_objetivo, col_objetivo) no estado goal
    goal_pos: Dict[int, Coordinate] = {
        value: index_to_row_col(i, n) for i, value in enumerate(goal)
    }
    distance_sum = 0
    for index, value in enumerate(state):
        if value == 0:
            continue  # ignora o vazio
        row, col = index_to_row_col(index, n)
        goal_row, goal_col = goal_pos[value]
        distance_sum += abs(row - goal_row) + abs(col - goal_col)
    return distance_sum


def _linear_conflicts_in_row(row_values: List[int],
                             row_index: int,
                             goal_pos: Dict[int, Coordinate]) -> int:
    """
    Conta conflitos lineares em uma linha:
    - Considera apenas peças que pertencem a essa MESMA linha no objetivo.
    - Se sua ordem de colunas-objetivo está invertida, há um conflito (cada par invertido = 1).
    """
    goal_cols_in_this_row: List[int] = []
    for value in row_values:
        if value == 0:
            continue
        goal_row, goal_col = goal_pos[value]
        if goal_row == row_index:
            goal_cols_in_this_row.append(goal_col)

    # Contagem de pares invertidos na lista de colunas-objetivo
    conflicts = 0
    for i in range(len(goal_cols_in_this_row)):
        for j in range(i + 1, len(goal_cols_in_this_row)):
            if goal_cols_in_this_row[i] > goal_cols_in_this_row[j]:
                conflicts += 1
    return conflicts


def _linear_conflicts_in_col(col_values: List[int],
                             col_index: int,
                             goal_pos: Dict[int, Coordinate]) -> int:
    """
    Conta conflitos lineares em uma coluna:
    - Considera apenas peças que pertencem a essa MESMA coluna no objetivo.
    - Ordem vertical invertida => conflito.
    """
    goal_rows_in_this_col: List[int] = []
    for value in col_values:
        if value == 0:
            continue
        goal_row, goal_col = goal_pos[value]
        if goal_col == col_index:
            goal_rows_in_this_col.append(goal_row)

    conflicts = 0
    for i in range(len(goal_rows_in_this_col)):
        for j in range(i + 1, len(goal_rows_in_this_col)):
            if goal_rows_in_this_col[i] > goal_rows_in_this_col[j]:
                conflicts += 1
    return conflicts


def h_linear_conflict(state: State, goal: State) -> int:
    """
    Heurística Linear Conflict:
    h = Manhattan + 2 * (conflitos em linhas + conflitos em colunas).
    - Adiciona 2 por conflito porque, no mínimo, um movimento extra é necessário para resolver
      cada par invertido além do custo Manhattan.
    """
    n = side_len(state)
    base = h_manhattan(state, goal)  # parte Manhattan
    # Pré-calcula posições objetivo para checar se a peça pertence à linha/coluna atual
    goal_pos: Dict[int, Coordinate] = {value: index_to_row_col(i, n) for i, value in enumerate(goal)}

    conflicts = 0
    # Conta conflitos linha a linha
    for row_index in range(n):
        row_values = list(state[row_index*n:(row_index+1)*n])
        conflicts += _linear_conflicts_in_row(row_values, row_index, goal_pos)

    # Conta conflitos coluna a coluna
    for col_index in range(n):
        col_values = [state[row*n + col_index] for row in range(n)]
        conflicts += _linear_conflicts_in_col(col_values, col_index, goal_pos)

    return base + 2 * conflicts

# -----------------------------
# Estrutura de Nó
# -----------------------------
@dataclass
class Node:
    """Representa um nó da busca (estado + ponteiro para o pai)."""
    state: State
    parent: Optional["Node"]

    def path(self) -> List[State]:
        """
        Reconstrói o caminho da raiz até este nó.
        - Sobe pelos pais acumulando estados (rev), depois reverte para ordem do início ao fim.
        """
        node = self
        rev: List[State] = []
        while node is not None:
            rev.append(node.state)
            node = node.parent
        return list(reversed(rev))

# -----------------------------
# Implementação do IDA*
# -----------------------------
def ida_star(start: State,
             goal: State,
             heuristic: Callable[[State, State], int]) -> Tuple[Optional[Node], int]:
    """
    Implementa o algoritmo IDA* (Iterative Deepening A*):
    - Usa uma DFS com limite de f = g + h (bound).
    - Se nenhuma solução estiver com f <= bound, aumenta-se o bound para o menor f que excedeu (next_bound).
    Retorna (nó objetivo ou None, número total de nós expandidos).
    """
    # Limite inicial de f é a heurística no estado inicial
    bound = heuristic(start, goal)

    # Estruturas para manter o caminho corrente (stack implícita) e evitar ciclos no caminho
    start_node = Node(start, None)
    path_nodes: List[Node] = [start_node]
    path_set: Set[State] = {start}

    total_expanded = 0

    while True:
        # Executa uma DFS limitada por 'bound' a partir do caminho atual
        found, next_bound, expanded = _ida_dfs(
            path_nodes, path_set, g=0, bound=bound, goal=goal, h=heuristic
        )
        total_expanded += expanded

        if found is not None:
            # Solução encontrada dentro do limite de f atual
            return found, total_expanded

        if next_bound == math.inf:
            # Nenhum caminho possível (ex.: estado insolúvel)
            return None, total_expanded

        # Aumenta o limite para o menor f que estourou o bound na iteração anterior
        bound = next_bound


def _ida_dfs(path_nodes: List[Node],
             path_set: Set[State],
             g: int,
             bound: int,
             goal: State,
             h: Callable[[State, State], int]) -> Tuple[Optional[Node], int, int]:
    """
    Função auxiliar recursiva: DFS limitada por f = g + h.
    Retorna (nó_objetivo_ou_None, próximo_bound_sugerido, nós_expandidos_nesta_chamada).
    """
    expanded = 0

    # Nó e estado correntes estão no topo de path_nodes
    current_node = path_nodes[-1]
    state_at_node = current_node.state

    # Custo f = g + h no nó atual
    f_cost = g + h(state_at_node, goal)
    if f_cost > bound:
        # Se f estourou o limite, devolve esse f para sugerir um novo bound (candidato a next_bound)
        return None, f_cost, expanded

    if state_at_node == goal:
        # Objetivo alcançado
        return current_node, bound, expanded

    # Se não estourou e não é objetivo, expande vizinhos
    min_excess = math.inf  # menor f que excedeu o bound, visto nos filhos (para o próximo bound)
    for neighbor_state in neighbors(state_at_node):
        # Evita laço: não visite um estado que já está no caminho atual
        if neighbor_state in path_set:
            continue

        # Cria nó filho e adiciona ao caminho
        child_node = Node(neighbor_state, current_node)
        path_nodes.append(child_node)
        path_set.add(neighbor_state)
        expanded += 1  # contamos a expansão ao seguir para o filho

        # Chamada recursiva para o filho com g+1 (um movimento a mais)
        found, next_bound, expanded_child = _ida_dfs(
            path_nodes, path_set, g + 1, bound, goal, h
        )
        expanded += expanded_child  # acumula expansões do nível de baixo

        if found is not None:
            # Se o filho encontrou solução, propaga o sucesso
            return found, bound, expanded

        # Se não achou, mas sugeriu um novo bound menor que o min_excess atual, atualiza
        if next_bound < min_excess:
            min_excess = next_bound

        # BACKTRACK: remove o filho e seu estado do caminho atual
        path_nodes.pop()
        path_set.remove(neighbor_state)

    # Se nenhum filho levou a solução, devolve None e o menor f que excedeu (ou inf se nada expandiu)
    return None, min_excess, expanded

# -----------------------------
# Execução / Menu interativo
# -----------------------------
DEFAULT_START_3x3: State = (1, 2, 3,
                            4, 0, 6,
                            7, 5, 8)

DEFAULT_START_4x4: State = ( 5,  1,  2,  4,
                             9,  6,  3,  8,
                            13, 10,  7, 12,
                             0, 14, 11, 15)

def parse_state_string(s: str, n: int) -> State:
    """
    Converte string '1,2,3,...' em tupla (0..N*N-1).
    - Valida quantidade de números e o conjunto {0..N*N-1} sem repetição.
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != n*n:
        raise ValueError(f"Estado deve ter {n*n} números (0..{n*n-1}).")
    try:
        values = tuple(int(x) for x in parts)
    except ValueError:
        raise ValueError("Todos os valores devem ser inteiros.")
    if sorted(values) != list(range(n*n)):
        raise ValueError(f"O estado deve conter exatamente 0..{n*n-1}, sem repetição.")
    return values


def read_state_from_input(prompt: str, n: int, default_state: State) -> State:
    """
    Lê o estado inicial do usuário (ou usa o padrão se ENTER).
    Em caso de erro, informa e pergunta novamente.
    """
    s = input(prompt).strip()
    if not s:
        return default_state
    try:
        return parse_state_string(s, n)
    except Exception as e:
        print(f"Entrada inválida: {e}")
        return read_state_from_input(prompt, n, default_state)


def choose_option(title: str, options: List[str], default_idx: int = 0) -> str:
    """
    Menu simples para seleção de opções.
    - Mostra lista numerada; ENTER escolhe a opção padrão.
    """
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        mark = " (padrão)" if i-1 == default_idx else ""
        print(f"  {i}. {opt}{mark}")
    choice = input("Escolha (ENTER p/ padrão): ").strip()
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
    """
    Menu principal (formato idêntico aos Projetos 1 e 2).
    - Escolhe tamanho (3x3 ou 4x4), heurística, estado inicial e executa IDA*.
    """
    print("======= QUEBRA-CABEÇA — BUSCA COMPLEXA (IDA*) =======")

    # 1) Tamanho do tabuleiro
    size_choice = choose_option(
        "Tamanho do tabuleiro",
        ["3x3 (8-puzzle)", "4x4 (15-puzzle)"],
        default_idx=0
    )
    if size_choice.startswith("3x3"):
        n = 3
        GOAL = goal_of(3)
        start = read_state_from_input(
            "Digite o estado inicial (ex: 1,2,3,4,0,6,7,5,8) ou ENTER p/ padrão: ",
            n, DEFAULT_START_3x3
        )
    else:
        n = 4
        GOAL = goal_of(4)
        start = read_state_from_input(
            "Digite o estado inicial (ex: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0) ou ENTER p/ padrão: ",
            n, DEFAULT_START_4x4
        )

    # 2) Heurística
    heuristic_name = choose_option("Heurística", ["manhattan", "linear_conflict"], default_idx=0)
    heuristic_map = {"manhattan": h_manhattan, "linear_conflict": h_linear_conflict}
    heuristic = heuristic_map[heuristic_name]

    # 3) Solubilidade
    if not is_solvable(start):
        print("\nATENÇÃO: este estado NÃO é solúvel.")
        print(print_board(start))
        cont = input("Prosseguir mesmo assim? (s/N): ").strip().lower()
        if cont != "s":
            return

    # 4) Resumo e execução
    print("\nResumo da execução:")
    print(f"  Tamanho: {n}x{n}")
    print(f"  Heurística: {heuristic_name}")
    print("  Inicial:\n" + print_board(start))
    print("  Objetivo:\n" + print_board(GOAL))
    print("\nExecutando IDA*...\n")

    result, expanded = ida_star(start, GOAL, heuristic)

    print("\n===== RESULTADO =====")
    print(f"Heurística: {heuristic_name}")
    if result is None:
        print(f"Nenhuma solução encontrada. Nós expandidos: {expanded}")
        return

    path = result.path()
    print(f"Solução encontrada em {len(path)-1} movimentos.")
    print(f"Nós expandidos (total): {expanded}\n")
    for i, state in enumerate(path):
        print(f"Passo {i}:\n{print_board(state)}\n")


if __name__ == "__main__":
    interactive_menu()
