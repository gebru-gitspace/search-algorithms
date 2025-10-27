#!/usr/bin/env python3
"""
pathfinder_pacman.py

A simple pygame grid environment (Pacman-style top-down view) that
lets you draw walls and run pathfinding algorithms:
 - BFS
 - DFS
 - A* (Manhattan)
 - UCS (Dijkstra / uniform-cost)

Controls:
 - Left-click: toggle wall on that cell
 - Press S then left-click: set Start
 - Press G then left-click: set Goal
 - 1: select BFS
 - 2: select DFS
 - 3: select A*
 - 4: select UCS
 - Space: run/step the selected algorithm
 - A: toggle auto-run (continuous)
 - R: reset grid (clear walls, start/goal remain)
 - + / - : speed up / slow down visualization

Requires: pygame
    pip install pygame
Run:
    python pathfinder_pacman.py
"""
import pygame
import sys
from collections import deque
import heapq
import math
import time

# ----------------- Configuration -----------------
CELL_SIZE = 24        # pixel size of each grid cell
GRID_COLS = 28        # like Pacman width-ish
GRID_ROWS = 20        # height
MARGIN = 20           # margin around grid
FPS = 30

# Colors
COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_WALL = (30, 60, 200)
COLOR_START = (50, 200, 50)
COLOR_GOAL = (200, 50, 50)
COLOR_FRONTIER = (255, 180, 30)
COLOR_EXPLORED = (180, 180, 180)
COLOR_PATH = (240, 240, 100)
COLOR_TEXT = (230, 230, 230)
COLOR_AGENT = (255, 200, 50)

# --------------------------------------------------

# Helper positions conversion
def cell_to_pixel(c, r):
    x = MARGIN + c * CELL_SIZE
    y = MARGIN + r * CELL_SIZE
    return (x, y)

def pixel_to_cell(x, y):
    cx = (x - MARGIN) // CELL_SIZE
    cy = (y - MARGIN) // CELL_SIZE
    if 0 <= cx < GRID_COLS and 0 <= cy < GRID_ROWS:
        return int(cx), int(cy)
    return None

# Neighbors (4-directional)
def neighbors(node):
    c, r = node
    for dc, dr in ((1,0),(-1,0),(0,1),(0,-1)):
        nc, nr = c+dc, r+dr
        if 0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS:
            yield (nc, nr)

# Heuristic for A* (Manhattan)
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# Path reconstruct
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# BFS generator (yields state for visualization)
def bfs(start, goal, walls):
    frontier = deque([start])
    came_from = {}
    explored = set([start])
    while frontier:
        current = frontier.popleft()
        yield {'current': current, 'frontier': list(frontier), 'explored': set(explored), 'status': 'searching'}
        if current == goal:
            path = reconstruct_path(came_from, current)
            yield {'path': path, 'status': 'found'}
            return
        for n in neighbors(current):
            if n in explored or n in walls: continue
            came_from[n] = current
            explored.add(n)
            frontier.append(n)
    yield {'status': 'no_path'}
    return

# DFS generator (stack)
def dfs(start, goal, walls):
    frontier = [start]  # stack
    came_from = {}
    explored = set([start])
    while frontier:
        current = frontier.pop()
        yield {'current': current, 'frontier': list(frontier), 'explored': set(explored), 'status': 'searching'}
        if current == goal:
            path = reconstruct_path(came_from, current)
            yield {'path': path, 'status': 'found'}
            return
        for n in neighbors(current):
            if n in explored or n in walls: continue
            came_from[n] = current
            explored.add(n)
            frontier.append(n)
    yield {'status': 'no_path'}
    return

# Uniform Cost Search (Dijkstra) generator
def ucs(start, goal, walls, cost_fn=None):
    if cost_fn is None:
        cost_fn = lambda a, b: 1
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    explored = set()
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        explored.add(current)
        frontier_list = [item[1] for item in frontier]
        yield {'current': current, 'frontier': frontier_list, 'explored': set(explored), 'costs': dict(cost_so_far), 'status': 'searching'}
        if current == goal:
            path = reconstruct_path(came_from, current)
            yield {'path': path, 'status': 'found', 'cost': current_cost}
            return
        for n in neighbors(current):
            if n in walls: continue
            new_cost = cost_so_far[current] + cost_fn(current, n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                came_from[n] = current
                heapq.heappush(frontier, (new_cost, n))
    yield {'status': 'no_path'}
    return

# A* generator
def astar(start, goal, walls, heuristic=manhattan, cost_fn=None):
    if cost_fn is None:
        cost_fn = lambda a, b: 1
    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), 0, start))  # (f, g, node)
    came_from = {}
    g_score = {start: 0}
    explored = set()
    while frontier:
        f, g, current = heapq.heappop(frontier)
        explored.add(current)
        frontier_list = [item[2] for item in frontier]
        yield {'current': current, 'frontier': frontier_list, 'explored': set(explored), 'g': dict(g_score), 'status': 'searching'}
        if current == goal:
            path = reconstruct_path(came_from, current)
            yield {'path': path, 'status': 'found', 'cost': g_score[current]}
            return
        for n in neighbors(current):
            if n in walls: continue
            tentative_g = g_score[current] + cost_fn(current, n)
            if n not in g_score or tentative_g < g_score[n]:
                g_score[n] = tentative_g
                came_from[n] = current
                fscore = tentative_g + heuristic(n, goal)
                heapq.heappush(frontier, (fscore, tentative_g, n))
    yield {'status': 'no_path'}
    return

# ----------------- Pygame UI -----------------
def draw_grid(screen, walls, start, goal, frontier, explored, path, current, font, alg_name, speed, auto_mode):
    screen.fill(COLOR_BG)

    # Draw cells
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rect = pygame.Rect(MARGIN + c*CELL_SIZE, MARGIN + r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (c, r) in walls:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                pygame.draw.rect(screen, COLOR_BG, rect)

    # Explored
    for n in explored or []:
        x, y = cell_to_pixel(*n)
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_EXPLORED, rect)

    # Frontier
    for n in frontier or []:
        x, y = cell_to_pixel(*n)
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_FRONTIER, rect)

    # Path
    if path:
        for n in path:
            x, y = cell_to_pixel(*n)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_PATH, rect)

    # Start & Goal
    if goal:
        x, y = cell_to_pixel(*goal)
        pygame.draw.rect(screen, COLOR_GOAL, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))
    if start:
        x, y = cell_to_pixel(*start)
        pygame.draw.rect(screen, COLOR_START, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

    # Agent at current
    if current:
        x, y = cell_to_pixel(*current)
        pygame.draw.circle(screen, COLOR_AGENT, (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)

    # Grid lines
    for r in range(GRID_ROWS + 1):
        y = MARGIN + r * CELL_SIZE
        pygame.draw.line(screen, COLOR_GRID, (MARGIN, y), (MARGIN + GRID_COLS*CELL_SIZE, y))
    for c in range(GRID_COLS + 1):
        x = MARGIN + c * CELL_SIZE
        pygame.draw.line(screen, COLOR_GRID, (x, MARGIN), (x, MARGIN + GRID_ROWS*CELL_SIZE))

    # Text info
    info_lines = [
        f"Algo: {alg_name}    Speed: {speed:.2f}s step    Auto: {'ON' if auto_mode else 'OFF'}",
        "Controls: Left-click toggle wall | S then click: set Start | G then click: set Goal",
        "1:BFS  2:DFS  3:A*  4:UCS   Space: step/run   A: toggle auto-run   R: reset",
        "+ / - : faster/slower"
    ]
    for i, line in enumerate(info_lines):
        surf = font.render(line, True, COLOR_TEXT)
        screen.blit(surf, (MARGIN, MARGIN + GRID_ROWS*CELL_SIZE + 10 + i*20))

    pygame.display.flip()


def main():
    pygame.init()
    width = MARGIN*2 + CELL_SIZE * GRID_COLS
    height = MARGIN*2 + CELL_SIZE * GRID_ROWS + 90
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pacman-style Pathfinding Visualizer - BFS/DFS/A*/UCS")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    # state
    walls = set()
    start = (1, 1)
    goal = (GRID_COLS-2, GRID_ROWS-2)
    selected_alg = 'BFS'  # options: BFS, DFS, ASTAR, UCS
    algo_gen = None
    frontier = []
    explored = set()
    path = []
    current = None
    auto_mode = False
    stepping = False
    speed = 0.05  # seconds between automatic steps

    waiting_for_set = None  # 'S' or 'G' for next click

    last_step_time = time.time()

    # main loop
    running = True
    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_r:
                    walls.clear()
                    path = []
                    frontier = []
                    explored = set()
                    algo_gen = None
                if event.key == pygame.K_s:
                    waiting_for_set = 'S'
                if event.key == pygame.K_g:
                    waiting_for_set = 'G'
                if event.key == pygame.K_1:
                    selected_alg = 'BFS'
                    algo_gen = None
                if event.key == pygame.K_2:
                    selected_alg = 'DFS'
                    algo_gen = None
                if event.key == pygame.K_3:
                    selected_alg = 'A*'
                    algo_gen = None
                if event.key == pygame.K_4:
                    selected_alg = 'UCS'
                    algo_gen = None
                if event.key == pygame.K_SPACE:
                    # Start or step the generator
                    if algo_gen is None:
                        if start is None or goal is None:
                            print("Set start and goal first.")
                        else:
                            path = []
                            frontier = []
                            explored = set()
                            current = None
                            if selected_alg == 'BFS':
                                algo_gen = bfs(start, goal, walls)
                            elif selected_alg == 'DFS':
                                algo_gen = dfs(start, goal, walls)
                            elif selected_alg == 'A*':
                                algo_gen = astar(start, goal, walls)
                            elif selected_alg == 'UCS':
                                algo_gen = ucs(start, goal, walls)
                            stepping = True
                    else:
                        stepping = True
                if event.key == pygame.K_a:
                    auto_mode = not auto_mode
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    speed = max(0.01, speed * 0.7)
                if event.key == pygame.K_MINUS:
                    speed = min(1.0, speed / 0.7)

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                cell = pixel_to_cell(*pos)
                if cell:
                    if waiting_for_set == 'S':
                        start = cell
                        waiting_for_set = None
                        algo_gen = None
                    elif waiting_for_set == 'G':
                        goal = cell
                        waiting_for_set = None
                        algo_gen = None
                    else:
                        # toggle wall
                        if cell == start:
                            start = None
                        elif cell == goal:
                            goal = None
                        else:
                            if cell in walls:
                                walls.remove(cell)
                            else:
                                walls.add(cell)
                        algo_gen = None

        # Step the algorithm generator if active and stepping/auto and time passed
        if algo_gen is not None and (stepping or auto_mode):
            if now - last_step_time >= speed:
                last_step_time = now
                try:
                    result = next(algo_gen)
                    # Interpret generator result
                    if result.get('status') == 'searching':
                        frontier = result.get('frontier', frontier)
                        explored = result.get('explored', explored)
                        current = result.get('current', current)
                        path = []
                    elif result.get('status') == 'found':
                        path = result.get('path', [])
                        # stop stepping
                        stepping = False
                        algo_gen = None
                        frontier = []
                        explored = set()
                        current = None
                    elif result.get('status') == 'no_path':
                        print("No path found.")
                        stepping = False
                        algo_gen = None
                        frontier = []
                        explored = set()
                        current = None
                except StopIteration:
                    stepping = False
                    algo_gen = None

        # draw
        draw_grid(screen, walls, start, goal, frontier, explored, path, current, font, selected_alg, speed, auto_mode)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
