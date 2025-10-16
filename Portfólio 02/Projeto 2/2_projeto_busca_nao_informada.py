"""
Projeto 2 — Busca NÃO informada (IDDFS) no quebra-cabeça
Autor: Maciel Ferreira Custódio Júnior
Matrícula: 190100087
Disciplina: FGA0221 – Inteligência Artificial

Descrição
---------
Resolvendo o quebra-cabeça deslizante (quebra-cabeça 3x3) usando duas busca não informada
implementando Iterative Deepening Depth-First Search (IDDFS):
- Executa DFS limitada a uma profundidade L (DLS), aumentando L=0,1,2,...
- É busca não informada e encontra solução de menor profundidade
- Usa pouca memória (como DFS), mas com sobrecusto de reexplorar níveis rasos
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Set, Dict

State = Tuple[int, ...]            # estado: tupla de 9 inteiros (0 é o espaço vazio)
Coord = Tuple[int, int]            # (linha, coluna)
GOAL: State = (1,2,3,4,5,6,7,8,0)  # objetivo padrão


# -----------------------------
# Utilitários
# -----------------------------
def index_to_row_col(index: int) -> Coord:
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
    Verifica se o quebra-cabeça é solúvel.
    Regra: número de inversões (ignorando 0) deve ser par.
    """
    arr = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions % 2 == 0

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
# Estruturas de busca
# -----------------------------
@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    depth: int  # profundidade do nó (g = número de movimentos)

    def path(self) -> List[State]:
        """Reconstrói o caminho desde a raiz até este nó."""
        node: Optional["Node"] = self
        rev: List[State] = []
        while node is not None:
            rev.append(node.state)
            node = node.parent
        return list(reversed(rev))


# -----------------------------------
# DLS (Depth-Limited Search) e IDDFS
# -----------------------------------
def depth_limited_search(start: State, goal: State, limit: int) -> Tuple[Optional[Node], bool, int]:
    """
    Executa uma busca em profundidade limitada (DFS com profundidade máxima 'limit').

    A função retorna uma tupla contendo:
      - o nó objetivo (caso encontrado) ou None se não houver solução dentro do limite;
      - um valor booleano 'atingiu_corte', que indica se a busca foi interrompida por atingir o limite de profundidade;
      - o número total de nós expandidos durante essa chamada.
    """
    expanded = 0

    start_node = Node(start, None, depth=0)
    stack: List[Tuple[Node, Optional[Iterable[State]]]] = [(start_node, None)] # Pilha iterativa: (nó, iterador de vizinhos ou None)

    path_set: Set[State] = {start} # estados no caminho atual

    reached_cutoff = False

    while stack:
        node, it = stack[-1]

        # Checagem de objetivo
        if node.state == goal:
            return node, reached_cutoff, expanded

        if it is None:
            # Limite de profundidade
            if node.depth == limit: # atingiu corte
                reached_cutoff = True # marca que o corte foi atingido
                stack.pop() # desempilha
                path_set.remove(node.state) # remove do caminho atual
                continue

            it = iter(neighbors(node.state)) # cria iterador de vizinhos
            stack[-1] = (node, it) # atualiza o topo da pilha com o iterador

        try:
            nb = next(it) # obtém próximo vizinho

            if nb in path_set: # evita ciclos
                continue
            expanded += 1

            child = Node(nb, node, node.depth + 1) # cria nó filho
            stack.append((child, None)) # empilha filho
            path_set.add(nb) # adiciona ao caminho atual

        except StopIteration:
            # Todos os vizinhos foram explorados
            stack.pop()
            path_set.remove(node.state)

    return None, reached_cutoff, expanded


def iddfs(start: State, goal: State, max_depth: int = 40) -> Tuple[Optional[Node], int]:
    """
    Iterative Deepening DFS: executa DLS com limites 0,1,2,... até encontrar solução
    ou atingir max_depth. Retorna (nó objetivo ou None, total de nós expandidos).
    """
    total_expanded = 0
    for limit in range(max_depth + 1):
        result, cutoff, expanded = depth_limited_search(start, goal, limit)
        total_expanded += expanded
        if result is not None:
            return result, total_expanded
        if not cutoff:
            # não atingimos corte e não achamos solução => não há solução em profundidade > limit
            break
    return None, total_expanded


# -----------------------------
# Runner simples (menu)
# -----------------------------
DEFAULT_START: State = (1, 4, 2, 3, 7, 5, 6, 0, 8)

def parse_state_string(s: str) -> State:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 9:
        raise ValueError("Forneça 9 números (0..8) separados por vírgula.")
    vals = tuple(int(x) for x in parts)
    if sorted(vals) != list(range(9)):
        raise ValueError("O estado deve conter exatamente os dígitos 0..8, sem repetição.")
    return vals

def interactive_menu() -> None:
    print("======= QUEBRA-CABEÇA — BUSCA NÃO INFORMADA (DLS e IDDFS) =======")
    s = input('Digite o estado inicial (ex: 1,2,3,4,0,6,7,5,8) ou ENTER p/ padrão: ').strip()
    start = parse_state_string(s) if s else DEFAULT_START

    if not is_solvable(start):
        print("\nATENÇÃO: este estado NÃO é solúvel (paridade de inversões ímpar).")
        print(print_board(start))
        cont = input("Prosseguir mesmo assim? (s/N): ").strip().lower()
        if cont != "s":
            return  # sai do menu sem reabrir

    print("\nEstratégia:\n  1) dls   (profundidade limitada)\n  2) iddfs (aprofundamento iterativo) [padrão]")
    choice = input("Escolha (1/2, ENTER=padrão): ").strip()
    strategy = "dls" if choice == "1" else "iddfs"

    max_depth = 40
    if strategy == "dls":
        md = input("Profundidade máxima (ENTER p/ 40): ").strip()
        if md:
            max_depth = int(md)

    print("\nResumo:")
    print("Inicial:\n" + print_board(start))
    print("\nObjetivo:\n" + print_board(GOAL))

    if strategy == "dls":
        print("\nEstratégia: DLS (profundidade limitada)")
        print(f"Profundidade máxima: {max_depth}")
        result, cutoff, expanded = depth_limited_search(start, GOAL, max_depth)
        if result is None:
            print(f"\nNenhuma solução dentro do limite {max_depth}. Nós expandidos: {expanded}")
            return
        path = result.path()
        print(f"\nSolução em {len(path)-1} movimentos.")
        print(f"Nós expandidos (nesta busca): {expanded}\n")
        for i, st in enumerate(path):
            print(f"Passo {i}:\n{print_board(st)}\n")
    else:
        print("\nExecutando IDDFS...")
        result, expanded = iddfs(start, GOAL, max_depth=40)
        print("\nResumo:")
        print("Inicial:\n" + print_board(start))
        print("\nObjetivo:\n" + print_board(GOAL))
        if result is None:
            print(f"\nNenhuma solução até profundidade 40. Nós expandidos: {expanded}")
            return
        path = result.path()
        print(f"\nSolução em {len(path)-1} movimentos.")
        print(f"Nós expandidos (total): {expanded}\n")
        for i, st in enumerate(path):
            print(f"Passo {i}:\n{print_board(st)}\n")

if __name__ == "__main__":
    interactive_menu()
