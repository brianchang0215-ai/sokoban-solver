// Sokoban solver, v2: push-based A* with room decomposition.
//
// The drawn levels are a hallway plus rooms off it (one door each, a
// box pushed through a door can never come back), so the push-optimal
// total is the sum of the rooms' optima. v2 therefore:
//   1. finds rooms as components cut off by a single door cell
//      (an articulation point of the floor graph),
//   2. solves each room on its own with push A* (Hungarian heuristic,
//      dead-square + 2x2-freeze pruning, exit-through-the-door check),
//   3. stitches the rooms together with hallway walks,
//   4. falls back to whole-level push A* when no decomposition covers
//      every box and goal.
//
// Local test mode only for now:
//   node main.ts <level.txt>
// prints the full move string (lowercase = walk, uppercase = push).
// The grader I/O adapter (how the 5 instances arrive / where answers
// go) is still to come, once the kit's competition.yaml is read; the
// search core here is format-independent and will be reused as-is.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

type Cell = number; // y * width + x

const DIRS = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
] as const;

export interface ParsedLevel {
  width: number;
  height: number;
  walls: Set<Cell>;
  goals: Cell[];
  goalSet: Set<Cell>;
  boxes: Cell[];
  player: Cell;
  floor: boolean[];
}

export function parseLevel(text: string): ParsedLevel {
  const lines = text.replace(/\r/g, '').split('\n');
  while (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  const height = lines.length;
  const width = Math.max(...lines.map((l) => l.length));
  const walls = new Set<Cell>();
  const goals: Cell[] = [];
  const goalSet = new Set<Cell>();
  const boxes: Cell[] = [];
  let player = -1;
  const floor: boolean[] = new Array(width * height).fill(false);
  for (let y = 0; y < height; y++) {
    const line = lines[y];
    for (let x = 0; x < width; x++) {
      const ch = x < line.length ? line[x] : ' ';
      const idx = y * width + x;
      if (ch === '#') {
        walls.add(idx);
      } else {
        floor[idx] = true;
        if (ch === '.' || ch === '*' || ch === '+') {
          goals.push(idx);
          goalSet.add(idx);
        }
        if (ch === '$' || ch === '*') boxes.push(idx);
        if (ch === '@' || ch === '+') player = idx;
      }
    }
  }
  boxes.sort((a, b) => a - b);
  return { width, height, walls, goals, goalSet, boxes, player, floor };
}

// ---- generic search board (whole level, or one room) ----------------

interface Board {
  width: number;
  height: number;
  floor: boolean[];
  goals: Cell[];
  goalSet: Set<Cell>;
  dead: boolean[];
  dists: number[][]; // dists[goalIndex][cell], walls-only BFS
}

function buildBoard(
  width: number,
  height: number,
  floor: boolean[],
  goals: Cell[],
): Board {
  const goalSet = new Set(goals);
  const dists: number[][] = [];
  for (const g of goals) {
    const d = new Array(width * height).fill(Infinity);
    d[g] = 0;
    const q = [g];
    let head = 0;
    while (head < q.length) {
      const c = q[head++];
      const x = c % width;
      const y = (c / width) | 0;
      for (const dir of DIRS) {
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const n = ny * width + nx;
        if (!floor[n] || d[n] !== Infinity) continue;
        d[n] = d[c] + 1;
        q.push(n);
      }
    }
    dists.push(d);
  }
  // Dead squares: reverse-push reachability from the goals, ignoring
  // other boxes (over-approximates aliveness, so pruning is safe).
  const alive = new Array(width * height).fill(false);
  const q: Cell[] = [];
  for (const g of goalSet) {
    alive[g] = true;
    q.push(g);
  }
  let head = 0;
  while (head < q.length) {
    const box = q[head++];
    const x = box % width;
    const y = (box / width) | 0;
    for (const dir of DIRS) {
      // Reverse a push in direction `dir`: the box came from
      // box-dir, pushed by a player standing at box-2*dir. Both
      // of those cells must be floor. (Player reachability to the
      // pushing cell is ignored here, which only over-approximates
      // aliveness, so pruning stays safe.)
      const bx = x - dir.dx;
      const by = y - dir.dy;
      const px = x - 2 * dir.dx;
      const py = y - 2 * dir.dy;
      if (px < 0 || py < 0 || px >= width || py >= height) continue;
      if (bx < 0 || by < 0 || bx >= width || by >= height) continue;
      const pCell = py * width + px;
      const bCell = by * width + bx;
      if (!floor[pCell] || !floor[bCell] || alive[bCell]) continue;
      alive[bCell] = true;
      q.push(bCell);
    }
  }
  const dead = alive.map((a, i) => floor[i] && !a);
  return { width, height, floor, goals, goalSet, dead, dists };
}

function reachableCells(
  board: Board,
  start: Cell,
  boxSet: Set<Cell>,
): { set: Set<Cell>; min: Cell } {
  const { width, height, floor } = board;
  const set = new Set<Cell>([start]);
  const q = [start];
  let head = 0;
  let min = start;
  while (head < q.length) {
    const c = q[head++];
    if (c < min) min = c;
    const x = c % width;
    const y = (c / width) | 0;
    for (const dir of DIRS) {
      const nx = x + dir.dx;
      const ny = y + dir.dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const n = ny * width + nx;
      if (!floor[n] || boxSet.has(n) || set.has(n)) continue;
      set.add(n);
      q.push(n);
    }
  }
  return { set, min };
}

function playerPath(
  board: Board,
  start: Cell,
  target: Cell,
  boxSet: Set<Cell>,
): string | null {
  if (start === target) return '';
  const { width, height, floor } = board;
  const prev = new Map<Cell, { from: Cell; ch: string }>();
  const q = [start];
  prev.set(start, { from: -1, ch: '' });
  let head = 0;
  while (head < q.length) {
    const c = q[head++];
    const x = c % width;
    const y = (c / width) | 0;
    for (const dir of DIRS) {
      const nx = x + dir.dx;
      const ny = y + dir.dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const n = ny * width + nx;
      if (!floor[n] || boxSet.has(n) || prev.has(n)) continue;
      prev.set(n, { from: c, ch: dir.walk });
      if (n === target) {
        let cur = n;
        let s = '';
        while (cur !== start) {
          const p = prev.get(cur)!;
          s = p.ch + s;
          cur = p.from;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Minimum-cost assignment, boxes x goals (square, costs may be
// Infinity). Standard O(n^3) Hungarian with potentials; returns
// Infinity when no finite assignment exists.
function hungarian(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0);
  const way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const c = cost[i0 - 1][j - 1];
        const cur = (c === Infinity ? 1e9 : c) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (p[j] === 0) return Infinity;
    const c = cost[p[j] - 1][j - 1];
    if (c === Infinity) return Infinity;
    total += c;
  }
  return total;
}

interface HeapItem {
  f: number;
  g: number;
  key: string;
  boxes: Cell[];
  playerActual: Cell;
  parentKey: string | null;
  pushDir: number;
  pushedBoxFrom: Cell;
}

class MinHeap {
  private a: HeapItem[] = [];
  get size(): number {
    return this.a.length;
  }
  // Deterministic order: lower f first, then deeper g, then the
  // smaller state key, so two runs expand states in the same order.
  private static less(x: HeapItem, y: HeapItem): boolean {
    if (x.f !== y.f) return x.f < y.f;
    if (x.g !== y.g) return x.g > y.g;
    return x.key < y.key;
  }
  push(item: HeapItem): void {
    const a = this.a;
    a.push(item);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!MinHeap.less(a[i], a[p])) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop(): HeapItem | undefined {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0];
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let s = i;
        if (l < a.length && MinHeap.less(a[l], a[s])) s = l;
        if (r < a.length && MinHeap.less(a[r], a[s])) s = r;
        if (s === i) break;
        [a[s], a[i]] = [a[i], a[s]];
        i = s;
      }
    }
    return top;
  }
}

export interface SearchResult {
  moves: string;
  pushes: number;
  nodes: number;
  finalPlayer: Cell;
}

// Push A* on one board. `exitCell` (optional) must be reachable by
// the player in the solved state, so a solved room can be left again.
// `weight` 1 is optimal A*; a weight above 1 trades optimality for
// speed and is only used as a fallback. `maxExpansions` is a fixed
// work budget, never the wall clock: the grader requires two runs
// to write identical bytes, so the search must not be cut off by
// time. A search that runs out of budget returns null.
function pushAStar(
  board: Board,
  startBoxes: Cell[],
  startPlayer: Cell,
  opts: { exitCell?: Cell; maxExpansions?: number; weight?: number } = {},
): SearchResult | null {
  const { width, goalSet, goals, dead, dists } = board;
  const useHungarian = startBoxes.length <= 6 && startBoxes.length === goals.length;
  const heuristic = (boxes: Cell[]): number => {
    if (useHungarian) {
      const cost = boxes.map((b) => dists.map((d) => d[b]));
      return hungarian(cost);
    }
    let sum = 0;
    for (const b of boxes) {
      if (goalSet.has(b)) continue;
      let best = Infinity;
      for (let gi = 0; gi < goals.length; gi++) {
        const d = dists[gi][b];
        if (d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  };

  const hasFrozen2x2 = (boxes: Cell[], boxSet: Set<Cell>): boolean => {
    const { width: w, height: h, floor } = board;
    for (const b of boxes) {
      if (goalSet.has(b)) continue;
      const x = b % w;
      const y = (b / w) | 0;
      for (let dy = -1; dy <= 0; dy++) {
        for (let dx = -1; dx <= 0; dx++) {
          const x0 = x + dx;
          const y0 = y + dy;
          if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
          let allBlocked = true;
          let hasGoal = false;
          for (let yy = y0; yy <= y0 + 1; yy++) {
            for (let xx = x0; xx <= x0 + 1; xx++) {
              const c = yy * w + xx;
              if (goalSet.has(c)) hasGoal = true;
              if (floor[c] && !boxSet.has(c)) allBlocked = false;
            }
          }
          if (allBlocked && !hasGoal) return true;
        }
      }
    }
    return false;
  };

  const boxes0 = [...startBoxes].sort((a, b) => a - b);
  if (boxes0.some((b) => dead[b])) return null;
  const startBoxSet = new Set(boxes0);
  const startReach = reachableCells(board, startPlayer, startBoxSet);
  const keyOf = (boxes: Cell[], canonPlayer: Cell): string =>
    boxes.join(',') + '|' + canonPlayer;
  const startKey = keyOf(boxes0, startReach.min);
  const h0 = heuristic(boxes0);
  if (h0 === Infinity) return null;

  const heap = new MinHeap();
  const weight = opts.weight ?? 1;
  const root: HeapItem = {
    f: h0 * weight,
    g: 0,
    key: startKey,
    boxes: boxes0,
    playerActual: startPlayer,
    parentKey: null,
    pushDir: -1,
    pushedBoxFrom: -1,
  };
  heap.push(root);
  const bestG = new Map<string, number>([[startKey, 0]]);
  const cameFrom = new Map<string, HeapItem>([[startKey, root]]);
  const maxExpansions = opts.maxExpansions ?? 60000;
  let expanded = 0;
  let goalKey: string | null = null;
  let goalItem: HeapItem | null = null;

  while (heap.size > 0) {
    const cur = heap.pop()!;
    if (bestG.get(cur.key) !== cur.g) continue;
    expanded++;
    if (expanded > maxExpansions) return null;
    const solved = cur.boxes.every((b) => goalSet.has(b));
    if (solved) {
      if (opts.exitCell !== undefined) {
        const reach = reachableCells(board, cur.playerActual, new Set(cur.boxes));
        if (!reach.set.has(opts.exitCell)) continue;
      }
      goalKey = cur.key;
      goalItem = cur;
      break;
    }
    const boxSet = new Set(cur.boxes);
    const reach = reachableCells(board, cur.playerActual, boxSet);
    for (let bi = 0; bi < cur.boxes.length; bi++) {
      const box = cur.boxes[bi];
      const bx = box % width;
      const by = (box / width) | 0;
      for (let di = 0; di < DIRS.length; di++) {
        const dir = DIRS[di];
        const behindX = bx - dir.dx;
        const behindY = by - dir.dy;
        const destX = bx + dir.dx;
        const destY = by + dir.dy;
        if (behindX < 0 || behindY < 0 || destX < 0 || destY < 0) continue;
        const behind = behindY * width + behindX;
        const dest = destY * width + destX;
        if (!board.floor[dest] || boxSet.has(dest)) continue;
        if (!reach.set.has(behind)) continue;
        if (dead[dest]) continue;
        const nextBoxes = [...cur.boxes];
        nextBoxes[bi] = dest;
        nextBoxes.sort((a, b) => a - b);
        const nextBoxSet = new Set(nextBoxes);
        if (hasFrozen2x2(nextBoxes, nextBoxSet)) continue;
        const nextReach = reachableCells(board, box, nextBoxSet);
        const nextKey = keyOf(nextBoxes, nextReach.min);
        const ng = cur.g + 1;
        if (bestG.has(nextKey) && bestG.get(nextKey)! <= ng) continue;
        const h = heuristic(nextBoxes);
        if (h === Infinity) continue;
        bestG.set(nextKey, ng);
        const item: HeapItem = {
          f: ng + weight * h,
          g: ng,
          key: nextKey,
          boxes: nextBoxes,
          playerActual: box,
          parentKey: cur.key,
          pushDir: di,
          pushedBoxFrom: box,
        };
        cameFrom.set(nextKey, item);
        heap.push(item);
      }
    }
  }

  if (goalKey === null || goalItem === null) return null;

  const chain: HeapItem[] = [];
  let k: string | null = goalKey;
  while (k !== null) {
    const item = cameFrom.get(k)!;
    chain.push(item);
    k = item.parentKey;
  }
  chain.reverse();
  let moves = '';
  let playerPos = startPlayer;
  const liveBoxSet = new Set(startBoxes);
  for (let i = 1; i < chain.length; i++) {
    const item = chain[i];
    const dir = DIRS[item.pushDir];
    const boxFrom = item.pushedBoxFrom;
    const behind = boxFrom - dir.dy * width - dir.dx;
    const walk = playerPath(board, playerPos, behind, liveBoxSet);
    if (walk === null) return null;
    moves += walk + dir.push;
    liveBoxSet.delete(boxFrom);
    liveBoxSet.add(boxFrom + dir.dy * width + dir.dx);
    playerPos = boxFrom;
  }
  return { moves, pushes: chain.length - 1, nodes: expanded, finalPlayer: goalItem.playerActual };
}

// ---- room decomposition ---------------------------------------------

interface Room {
  cells: Set<Cell>;
  door: Cell;
  boxes: Cell[];
  goals: Cell[];
  entries: Cell[];
}

function floorComponents(
  level: ParsedLevel,
  removed: Cell,
): Set<Cell>[] {
  const { width, height, floor } = level;
  const seen = new Set<Cell>();
  const comps: Set<Cell>[] = [];
  for (let i = 0; i < width * height; i++) {
    if (!floor[i] || i === removed || seen.has(i)) continue;
    const comp = new Set<Cell>([i]);
    seen.add(i);
    const q = [i];
    let head = 0;
    while (head < q.length) {
      const c = q[head++];
      const x = c % width;
      const y = (c / width) | 0;
      for (const dir of DIRS) {
        const nx = x + dir.dx;
        const ny = y + dir.dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const n = ny * width + nx;
        if (!floor[n] || n === removed || seen.has(n)) continue;
        seen.add(n);
        comp.add(n);
        q.push(n);
      }
    }
    comps.push(comp);
  }
  return comps;
}

function findRooms(level: ParsedLevel): Room[] | null {
  const { width, height, floor, boxes, goals, player } = level;
  const boxSet = new Set(boxes);
  const goalSet = new Set(goals);
  const candidates: Room[] = [];
  for (let c = 0; c < width * height; c++) {
    // The door cell itself may start out holding a box (a pull can
    // leave one in the doorway), so a door candidate is any floor
    // cell except the player's start; its box/goal, if any, belongs
    // to the room cut off on one side of it.
    if (!floor[c] || c === player) continue;
    const comps = floorComponents(level, c);
    if (comps.length < 2) continue;
    for (const comp of comps) {
      if (comp.has(player)) continue;
      if (comp.size > 36) continue;
      const compBoxes = boxes.filter((b) => comp.has(b) || b === c);
      const compGoals = goals.filter((g) => comp.has(g) || g === c);
      if (compBoxes.length < 2 || compBoxes.length > 4) continue;
      if (compBoxes.length !== compGoals.length) continue;
      const roomCells = new Set(comp);
      roomCells.add(c);
      // Entry cells: floor neighbours of the door cell, outside the room.
      const outsideEntries: Cell[] = [];
      const dx0 = c % width;
      const dy0 = (c / width) | 0;
      for (const dir of DIRS) {
        const nx = dx0 + dir.dx;
        const ny = dy0 + dir.dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const n = ny * width + nx;
        if (floor[n] && !comp.has(n) && n !== c) outsideEntries.push(n);
      }
      if (outsideEntries.length === 0) continue;
      candidates.push({ cells: roomCells, door: c, boxes: compBoxes, goals: compGoals, entries: outsideEntries });
    }
  }
  // Greedy cover, smallest room first: take disjoint candidates
  // in size order until every box and goal is covered. This is
  // the picking that worked on every practice level; a
  // depth-first exact-cover search over all candidates was tried
  // and drowned in spurious subsets, so it is not used.
  const seenCand = new Set<string>();
  const uniq: Room[] = [];
  for (const cand of candidates) {
    const key = cand.door + ':' + [...cand.cells].sort((a, b) => a - b).join(',');
    if (seenCand.has(key)) continue;
    seenCand.add(key);
    uniq.push(cand);
  }
  uniq.sort((a, b) => a.cells.size - b.cells.size);
  const picked: Room[] = [];
  const usedCells = new Set<Cell>();
  const coveredBoxes = new Set<Cell>();
  for (const cand of uniq) {
    let overlap = false;
    for (const cell of cand.cells) if (usedCells.has(cell)) { overlap = true; break; }
    if (overlap) continue;
    for (const cell of cand.cells) usedCells.add(cell);
    for (const b of cand.boxes) coveredBoxes.add(b);
    picked.push(cand);
  }
  if (coveredBoxes.size !== boxes.length) return null;
  const coveredGoals = new Set<Cell>();
  for (const r of picked) for (const g of r.goals) coveredGoals.add(g);
  if (coveredGoals.size !== goals.length) return null;
  return picked;
}

export interface SolveResult {
  moves: string;
  pushes: number;
  nodes: number;
}

const ROOM_BUDGET = 60000;
const WHOLE_BUDGET = 60000;

interface BudgetState {
  remaining: number;
}

// One room search at the given weights, in order. Results are
// cached per (door, entry, exit-required, weight) inside a level,
// so trying room orders never re-searches a room.
function solveRoom(
  board: Board,
  room: Room,
  entry: Cell,
  exitRequired: boolean,
  cache: Map<string, SearchResult | null>,
  budget: BudgetState,
  weights: number[],
): SearchResult | null {
  const baseKey = room.door + '|' + entry + '|' + (exitRequired ? '1' : '0');
  for (const weight of weights) {
    const key = baseKey + '|' + weight;
    if (cache.has(key)) {
      const hit = cache.get(key)!;
      if (hit) return hit;
      continue;
    }
    const allowance = Math.min(ROOM_BUDGET, budget.remaining);
    if (allowance <= 0) {
      cache.set(key, null);
      return null;
    }
    const res = pushAStar(board, room.boxes, entry, {
      exitCell: exitRequired ? room.door : undefined,
      maxExpansions: allowance,
      weight,
    });
    budget.remaining -= res ? res.nodes : allowance;
    cache.set(key, res);
    if (res) return res;
  }
  return null;
}

function solveByRoomsAttempt(
  level: ParsedLevel,
  fullBoard: Board,
  rooms: Room[],
  lastNoExit: boolean,
  cache: Map<string, SearchResult | null>,
  budget: BudgetState,
  weights: number[],
): SolveResult | null {
  const { width, height } = level;
  let moves = '';
  let pushes = 0;
  let nodes = 0;
  let playerPos = level.player;
  const globalBoxes = new Set(level.boxes);
  const boards = new Map<number, Board>();
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    let board = boards.get(room.door);
    if (!board) {
      const floorSub = new Array(width * height).fill(false);
      for (const cell of room.cells) floorSub[cell] = true;
      floorSub[room.door] = true;
      for (const e of room.entries) floorSub[e] = true;
      board = buildBoard(width, height, floorSub, room.goals);
      for (const e of room.entries) board.dead[e] = true;
      boards.set(room.door, board);
    }
    let entry: Cell | null = null;
    let walkToEntry = '';
    for (const e of room.entries) {
      const walk = playerPath(fullBoard, playerPos, e, globalBoxes);
      if (walk !== null) {
        entry = e;
        walkToEntry = walk;
        break;
      }
    }
    if (entry === null) return null;
    const exitRequired = !(lastNoExit && ri === rooms.length - 1);
    const res = solveRoom(board, room, entry, exitRequired, cache, budget, weights);
    if (!res) return null;
    moves += walkToEntry + res.moves;
    pushes += res.pushes;
    nodes += res.nodes;
    playerPos = res.finalPlayer;
    for (const b of room.boxes) globalBoxes.delete(b);
    for (const g of room.goals) globalBoxes.add(g);
  }
  return { moves, pushes, nodes };
}

function permutations<T>(items: T[]): T[][] {
  if (items.length <= 1) return [items];
  const out: T[][] = [];
  for (let i = 0; i < items.length; i++) {
    const rest = [...items.slice(0, i), ...items.slice(i + 1)];
    for (const p of permutations(rest)) out.push([items[i], ...p]);
  }
  return out;
}

export function solveLevel(level: ParsedLevel): SolveResult | null {
  const fullBoard = buildBoard(level.width, level.height, level.floor, level.goals);
  const budget: BudgetState = { remaining: 300000 };
  const rooms = findRooms(level);
  if (rooms) {
    const cache = new Map<string, SearchResult | null>();
    const orders = permutations(rooms);
    // Optimal only first, in every order (cached, so re-tries
    // are cheap): every room exit-checked, then with the last
    // room free to end trapped. Weighted fallbacks run only
    // after every optimal order has failed, in the picked order
    // alone, so a hard room cannot burn the budget three times
    // over in every permutation.
    for (const order of orders) {
      const solved = solveByRoomsAttempt(level, fullBoard, order, false, cache, budget, [1]);
      if (solved) return solved;
      if (budget.remaining <= 0) break;
    }
    for (const order of orders) {
      const solved = solveByRoomsAttempt(level, fullBoard, order, true, cache, budget, [1]);
      if (solved) return solved;
      if (budget.remaining <= 0) break;
    }
    const weighted = solveByRoomsAttempt(level, fullBoard, rooms, false, cache, budget, [1.5, 3]);
    if (weighted) return weighted;
  }
  // Fallback: the whole level at once, optimal first, then weighted.
  for (const weight of [1, 1.5, 3]) {
    const allowance = Math.min(WHOLE_BUDGET, budget.remaining);
    if (allowance <= 0) return null;
    const res = pushAStar(fullBoard, level.boxes, level.player, {
      maxExpansions: allowance,
      weight,
    });
    budget.remaining -= res ? res.nodes : allowance;
    if (res) return { moves: res.moves, pushes: res.pushes, nodes: res.nodes };
  }
  return null;
}

function main(): void {
  // One run answers every instance in the input directory,
  // writing /output/<id>/solution.txt for each. An unsolved
  // level still gets an (empty) answer file, in the stated
  // format, so the entry stays fully answered. Local runs use
  // the same directories through the environment, as the
  // kit's starter does.
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  const ids = readdirSync(inputDir).sort();
  for (const id of ids) {
    let moves = '';
    try {
      const level = parseLevel(readFileSync(join(inputDir, id, 'level.txt'), 'utf8'));
      const res = solveLevel(level);
      if (res) moves = res.moves;
    } catch {
      moves = '';
    }
    mkdirSync(join(outputDir, id), { recursive: true });
    writeFileSync(join(outputDir, id, 'solution.txt'), moves + '\n');
  }
}

main();
