# -*- coding: utf-8 -*-
""" The 8-Puzzle

The 8-Puzzle game board consists of eight sliding tiles bearing the numbers
1-8 that move on a 3 x 3 grid. When the puzzle is in the solved state state,
there is a blank space in the top left-hand corner, followed by the tiles in
increasing numerical order from left to right, top to bottom. 

Circumstances under n = 4 is coded, but takes very long time to run even
with the best heuristics.

Example:
	```bash
	$ python3 the8puzzle.py
	```
	```python3
	import the8puzzle.py

	n = 3
	state = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
	steps, frontierSize, err = solvePuzzle(n, state, heuristics[2], True)
	```

Attributes:
	tot_dirs (int), directions (list), dirmatch (list): record the movement
		direction of the empty tile.
	patternDB_3 (list), patternDB_4 (list): memorizes the pattern database
		when n <= 4. Conditions in which n >= 5 are not supported.

"""

tot_dirs = 4
directions = [
	(0, 1),
	(0, -1),
	(1, 0),
	(-1, 0)
]
dirmatch = [1, 0, 3, 2]

patternDB_3 = None
patternDB_4 = None

class Memoize:
	"""Memoize class for Memoization

	Recording the caculated value for heuristic_Manhattan and
	heuristic_MisplacedTiles. heuristic_DisjointPatternDB has its own
	library.

	Implementing Memoize will slow the program down a bit when running a
	single instance of search as it brings extra cost within hashing.
	If multiple states is calculated, it would roughly run at the same 
	speed compared to the version without Memoization.

	(The hashing process roughly cost the same amount of time as
	heuristics themselves. It might be faster if I use the hash to
	represent the puzzle state directly.)
	"""

	def __init__(self, fn):
		self.fn = fn
		self.memo = {}

	def __call__(self, *args):
		hashable_args = PuzzleDict.state_to_hash(args[0])
		if hashable_args not in self.memo:
			self.memo[hashable_args] = self.fn(*args)
		return self.memo[hashable_args]

@Memoize
def heuristic_Manhattan(state):
	"""heuristic for Puzzle based on Manhattan distance. """
	ans = 0
	for i in range(len(state)):
		for j in range(len(state)):
			ans += abs(state[i][j] % len(state) - i) + \
				   abs(state[i][j] // len(state) - j)
	return ans

@Memoize
def heuristic_MisplacedTiles(state):
	"""heuristic for Puzzle based on Misplaced Tiles. """
	ans = 0 
	for i in range(len(state)):
		for j in range(len(state)):
			if state[i][j] != i * len(state) + j:
				ans += 1
	return ans

class DisjointPattern(object):
	"""Library object to store a part of the pattern database.

	Attributes:
		lib (dict): stores the database in the form of
			state_hash_value:steps.	
		pattern_range = stores the related numbers appeared in the 
			database.
		self.Empty_mark = stores the number used to represent 
			unrelated tiles, for hash simplicity.
	"""

	def BreadthFirstSearch(self, init_state):
		"""BFS method to search through the pattern state space.

		Attribute lib is constructed here. Called within __init__(). 

		Args:
			init_state: the initial state.
		"""
		from collections import deque
		"""Using deque() to enable popleft() method for queue."""

		n = len(init_state)
		self.lib = {}

		"""Defined only for the n power table - maybe there is a better method"""
		searchDictBFS = PuzzleDict(n)

		statehash = PuzzleDict.state_to_hash(init_state)
		self.lib[statehash] = 0

		queue = deque([])
		queue.append((init_state, 0, statehash))	

		count = 0

		"""Typical BFS Search."""
		while len(queue) > 0:
			curstate, cursteps, curstatehash = queue.popleft()

			for i in range(n):
				for j in range(n):
					if curstate[i][j] == self.Empty_mark:
						for directionid in range(tot_dirs):
							direction = directions[directionid]
							new_pos = (i + direction[0], 
									   j + direction[1])
							if new_pos[0] in range(0, n) and \
								new_pos[1] in range(0, n):
								if curstate[new_pos[0]][new_pos[1]] != self.Empty_mark:
									"""
									Applied state_to_hash_with_prev method to optimize runnning speed.
									Details will be mentioned later.
									"""
									statehash = searchDictBFS.state_to_hash_with_prev(
										curstatehash, 
										(i, j),
										new_pos,
										self.Empty_mark, 
										curstate[new_pos[0]][new_pos[1]]
									)
									if not statehash in self.lib:
										self.lib[statehash] = cursteps + 1
										newstate = [[curstate[i][j] for j in range(n)] for i in range(n)]

										newstate[i][j], newstate[new_pos[0]][new_pos[1]] = \
														newstate[new_pos[0]][new_pos[1]], newstate[i][j]
										queue.append((newstate, cursteps + 1, statehash))
										if n == 4: print(queue[-1])

	def __init__(self, graph, pattern_range):
		n = len(graph)
		self.pattern_range = pattern_range

		"""Define Empty_mark as the first unused number between 0 and n * n - 1"""
		self.Empty_mark = 1
		while self.Empty_mark in self.pattern_range and \
			  self.Empty_mark < n * n:
			self.Empty_mark += 1
		if self.Empty_mark == n * n: raise Exception('The graph is full without a Disjoint Pattern')

		for i in range(n):
			for j in range(n):
				if graph[i][j] == -1:
					graph[i][j] = self.Empty_mark

		self.BreadthFirstSearch(graph)

	def match(self, state):
		"""Match state with the heuristic function value within pattern DB.

		Args:
			state: the current state.

		Return:
			heuristic value if found, Error message otherwise.
		"""
		statehash = PuzzleDict.ranged_state_to_hash(state, self.pattern_range, self.Empty_mark)
		if statehash in self.lib:
			return self.lib[statehash]
		else:
			raise Exception('Current State Unmatched: ', str(state))

def heuristic_DisjointPatternDB(state):
	"""heuristic for Puzzle based on Disjoint Pattern Database. """
	if len(state) < 3:
		return 0
	elif len(state) == 3:
		"""heuristic generation when n = 3. """
		global patternDB_3
		if patternDB_3 == None:
			g_0 = [
				[ 0,  1,  2],
				[ 3,  4, -1],
				[-1, -1, -1]
			]
			g_1 = [
				[ 0, -1, -1],
				[-1, -1,  5],
				[ 6,  7,  8]
			]
			patternDB_3 = [
				DisjointPattern(g_0, {0, 1, 2, 3, 4}), 
				DisjointPattern(g_1, {0, 5, 6, 7, 8})
			]
		patternlist = patternDB_3
	elif len(state) == 4:
		"""
		heuristic generation when n = 4. Well, it is called '8-puzzle'...
		Nevermind.
		"""
		global patternDB_4
		if patternDB_4 == None:
			g_0 = [
				[ 0,  1,  2, -1],
				[ 4,  5,  6, -1],
				[-1, -1, -1, -1],
				[-1, -1, -1, -1]
			]
			g_1 = [
				[ 0, -1, -1, -1],
				[-1, -1, -1, -1],
				[ 8,  9, 10, -1],
				[12, 13, -1, -1]
			]
			g_2 = [
				[ 0, -1, -1,  3],
				[-1, -1, -1,  7],
				[-1, -1, -1, 11],
				[-1, -1, 14, 15]
			]
			patternDB_4 = [
				DisjointPattern(g_0, {0, 1, 2, 4, 5, 6}), 
				DisjointPattern(g_1, {0, 8, 9, 10, 12, 13}),
				DisjointPattern(g_2, {0, 3, 7, 11, 14, 15})
			]
			print('pattern generated')
		patternlist = patternDB_4
	else:
		raise ValueError('n value of larger than 4 is not supported')

	"""
	We add the found heuristic values from multiple database together, since
	In easch database only the relationship within the pattern considered.
	Also, all tiles that are unrelated are allowed to move. The interaction
	between each pattern should further boost the estimated number of moves.
	"""

	ansh = 0
	for pattern in patternlist:
		curh = pattern.match(state)
		ansh += curh

	return ansh

def checkValidity(n, state):
	"""Validation Check.

	Input:
		n: side length of puzzle.
		state: n * n square.

	Output:
		True if valid, False if otherwise.
	"""
	import numpy
	numpy_state = numpy.array(state)
	if numpy_state.shape != (n, n):
		return False

	tile_range = range(n * n)
	number_appeared = [False for _ in tile_range]
	for i in range(n):
		for j in range(n):
			if not state[i][j] in tile_range:
				return False
			if number_appeared[state[i][j]]:
				return False
			number_appeared[state[i][j]] = True
	return True

def checkSolvability(n, state):
	"""Solvability Check.
	Methodology: Check the number of reversed tiles within the puzzle. Each
	move of the empty tile either doesn't change the reversed tile count, or
	shift it upward or downward by an even number.
	If odd numbers of reversed tiles are found, then the puzzle is not
	solvable.

	Input:
		n: side length of puzzle.
		state: n * n puzzle.

	Output:
		True if Solvable, False if otherwise.
	"""
	invCounter = 0

	for tile1 in range(n * n):
		for tile2 in range(tile1 + 1, n * n):
			i = tile1 // n
			j = tile1 % n
			i_ = tile2 // n
			j_ = tile2 % n

			if state[i][j] and state[i_][j_] and state[i][j] > state[i_][j_]:
				invCounter += 1

	return invCounter & 1 == 0

class PuzzleDict(object):
	"""Puzzle Dictionary (hash table) for repeated state checking.

	Args:
		n: side length of puzzle.
	"""
	def __init__(self, n):
		self.dict = {}
		self.n = n
		self.n2 = n * n

		self.mul = [1]
		for i in range(1, self.n2):
			self.mul.append(self.mul[-1] * self.n2)

	"""
	Method that transfers state to its hash. Decorated as staticmethod
	since it is often used without defining the class.

	Hashing mechanic:
	h = \sigma_{i = 0}^{n - 1}{n^i * p_i}

	I didn't use contor expansion since it costs way more time, and to
	fulfill requirement of hashing different patterns. 

	Args:
		state: n * n puzzle.
	Output:
		the hash value of the state.
	"""
	@staticmethod
	def state_to_hash(state):
		hashres = 0
		iterfact = 1
		n2 = len(state) * len(state)
		for i in range(len(state)):
			for j in range(len(state)):
				hashres += state[i][j] * iterfact
				iterfact = iterfact * n2
		return hashres

	"""
	Similar to method above, but added a range limitation to check the
	hash value of a state regarding one of the databases.
	"""
	@staticmethod
	def ranged_state_to_hash(state, pattern_range, Empty_mark):
		hashres = 0
		iterfact = 1
		n2 = len(state) * len(state)
		for i in range(len(state)):
			for j in range(len(state)):
				if state[i][j] in pattern_range:
					hashres += state[i][j] * iterfact
				else:
					hashres += Empty_mark * iterfact
				iterfact = iterfact * n2
		return hashres

	"""
	Similar to state_to_hash method, however given the previous hash
	value, the time complexity is lowered from O(n^2) to O(1).
	(That is roughly 80 steps to 4 steps)
	Hasten running time by approximately 0.5 - 1 sec in the sample case.
	"""
	def state_to_hash_with_prev(self, prevhashres, pos1, pos2, num1, num2):
		pos1_1d = pos1[0] * self.n + pos1[1]
		pos2_1d = pos2[0] * self.n + pos2[1]

		hashres = \
			prevhashres + \
			self.mul[pos1_1d] * (num2 - num1) + \
			self.mul[pos2_1d] * (num1 - num2)

		return hashres

	"""
	Check if the state exists in the current hash table/dictionary.
	If it is in, check if the steps required for the old node is better
	than the new node. 

	Args:
		steps: the number of existing steps, g(n);
		state: n * n puzzle.
		statehash: hashvalue of n * n puzzle. Either one could be
			provided. statehash takes priority.
	Output:
		True if does not exist or refreshes the state; False otherwise.
	"""
	def checkState(self, steps, state = None, statehash = None):
		if state == None and statehash == None:
			raise Exception('Either state or statehash must be given')
		if statehash == None:
			statehash = self.state_to_hash(state)

		if statehash in self.dict and self.dict[statehash] <= steps:
			return False
		else:
			return True

	"""
	Add the state into the hash table/dictionary.

	Args:
		steps: the number of existing steps, g(n);
		state: n * n puzzle.
		statehash: hashvalue of n * n puzzle. Either one could be
			provided. statehash takes priority.
	Output:
		statehash, in case that it is not calculated beforehand.
	"""
	def addState(self, steps, state = None, statehash = None):
		if state == None and statehash == None:
			raise Exception('Either state or statehash must be given')
		if statehash == None:
			statehash = self.state_to_hash(state)

		self.dict[statehash] = steps;
		return statehash

class PuzzleNode(object):
	"""Puzzle Node for A* search.

	Args:
		state (2dlist): n * n puzzle.
		parent (PuzzleNode): the previous node.
		steps (int): the number of steps, cost function.
		zpos (tuple): the position of empty tile, optional.  
		statehash (int): hash value of the current puzzle, optional. 
	"""

	def __init__(self, state, parent, steps, zpos = None, statehash = None):
		self.state = state
		self.parent = parent
		self.steps = steps

		if zpos == None:
			for i in range(len(self.state)):
				for j in range(len(self.state)):
					if state[i][j] == 0: self.zpos = (i, j)
		else:
			self.zpos = zpos

		if statehash == None:
			self.statehash = PuzzleDict.state_to_hash(state)
		else:
			self.statehash = statehash

	"""
	For comparison in the priority queue.
	Anyone knows how to avoid this part?
	"""
	def __lt__(self, other):
		return self.steps < other.steps

	"""String Output"""
	def __tilelineoutput(self, linenum):
		ans = '| '
		for j in range(len(self.state) - 1):
			ans += str(self.state[linenum][j]) + ' | '
		ans += str(self.state[linenum][n - 1]) + ' |'
		return ans

	def __str__(self):
		ans = '-' * 13
		for _ in range(len(self.state) - 1):
			ans += '\n' + self.__tilelineoutput(_) + \
				   '\n' + '-' * 13
		ans += '\n' + self.__tilelineoutput(len(self.state) - 1) + \
			   '\n' + '-' * 13
		return ans

	"""An attempt to move the empty tile to another place.

	Args:
		direction (tuple): one of four viable directions.
		searchDict (PuzzleDict): contains the hash table.
	Returns:
		(True, newNode) if successful, (False, None) if otherwise.
	"""
	def move(self, direction, searchDict):
		new_zpos = (self.zpos[0] + direction[0], self.zpos[1] + direction[1])
		if new_zpos[0] in range(0, len(self.state)) and \
		   new_zpos[1] in range(0, len(self.state)):
			statehash = searchDict.state_to_hash_with_prev(
				self.statehash,
				self.zpos, new_zpos,
				0,
				self.state[new_zpos[0]][new_zpos[1]]
			)
			valid = searchDict.checkState(self.steps + 1, statehash = statehash)
			if valid: 
				"""For some reason, copy.deepcogy is much slower than
				building another array."""
				new_state = [
					[self.state[i][j] for j in range(len(self.state))]
					for i in range(len(self.state))
				]

				new_state[self.zpos[0]][self.zpos[1]], \
					new_state[new_zpos[0]][new_zpos[1]] = \
					new_state[new_zpos[0]][new_zpos[1]], \
				new_state[self.zpos[0]][self.zpos[1]]

				newNode = PuzzleNode(
					new_state,
					self,
					self.steps + 1,
					zpos = new_zpos,
					statehash = statehash
				)
				return (True, newNode)
			else:
				return (False, None)
		else:
			return (False, None)

def solvePuzzle(n, init_state, heuristic, print_):
	"""Main Function for Puzzle Solving. A* Search.

	Args:
		n (int): side length of puzzle.
		init_state (2dlist): the initial state.
		heuristic (function): heuristic function.
		print_ (bool): indicating whether to print the middle steps.
			print is changed to print_ to avoid comflict with internal
			print function.
	Returns:
		steps: the number of steps required to reach the goal state
			from the initial state.
		frontierSize: the maximum size of the frontier during the search
		err: an error code:
			0 if succeeded;
			-1 if the state is not of the correct size and format;
			-2 if the state cannot be achieved.
	"""

	import time
	start_time = time.time()
	"""Initialization"""

	import heapq
	if not checkValidity(n, init_state):
		return (0, 0, -1)
	if not checkSolvability(n, init_state):
		return (0, 0, -2)

	queue = []
	init_Node = PuzzleNode(init_state, None, 0)
	heapq.heappush(queue, (0 + heuristic(init_state), PuzzleNode(init_state, None, 0), -1))

	searchDict = PuzzleDict(n)
	searchDict.addState(0, statehash = init_Node.statehash)

	goal_state = [[i * n + j for j in range(n)] for i in range(n)]
	goal_statehash = PuzzleDict.state_to_hash(goal_state)

	resultFlag = False
	resultNode = None

	maxfrontierSize = 0

	"""A* Search"""

	while len(queue) > 0 and not resultFlag:
		_, currentNode, pastdir = heapq.heappop(queue)

		for directionid in range(tot_dirs):
			if directionid == pastdir: continue

			direction = directions[directionid]
			valid, newNode = currentNode.move(direction, searchDict)
			if valid:
				searchDict.addState(currentNode.steps + 1, statehash = newNode.statehash)

				"""f(n) = g(n) + h(n)"""
				heapq.heappush(queue, (newNode.steps + heuristic(newNode.state), newNode, dirmatch[directionid]))
				if newNode.statehash == goal_statehash:
					resultFlag = True
					resultNode = newNode
					break

		if len(queue) > maxfrontierSize: maxfrontierSize = len(queue)

	"""Output"""
	if resultFlag:
		if print_:
			trace_node = resultNode
			state_path = []
			while trace_node != None:
				state_path.append(trace_node)
				trace_node = trace_node.parent
			for trace_node in reversed(state_path):
				print(trace_node.state)
			print('Number of moves to reach the goal:', resultNode.steps)
			print('Maximum frontier size:', maxfrontierSize)

		elapsed_time = time.time() - start_time
		print('Elapsed Time:', elapsed_time)

		return (resultNode.steps, maxfrontierSize, 0)
	else:
		return (-2, maxfrontierSize, 0) #Need Edit

heuristics = [heuristic_Manhattan, heuristic_MisplacedTiles, heuristic_DisjointPatternDB]

if __name__ == '__main__':
	n = 3
	state = [
		[7, 2, 4],
		[5, 0, 6],
		[8, 3, 1]
	]
	steps, frontierSize, err = solvePuzzle(n, state, heuristics[2], False)
	print(steps, frontierSize, err)

	StateLib = [[
		[7, 2, 4],
		[5, 0, 6],
		[8, 3, 1]
	], [
		[5, 7, 6],
		[2, 4, 3],
		[8, 1, 0]
	], [
		[7, 0, 8],
		[4, 6, 1],
		[5, 3, 2]
	],[
		[2, 3, 7],
		[1, 8, 0],
		[6 ,5 ,4]
	]]

	print('-' * 30)
	for state in StateLib:
		print(state)
		for heuristic in heuristics:
			steps, frontierSize, err = solvePuzzle(n, state, heuristic, False)
			print(heuristic, steps, frontierSize, err)
		print('-' * 30)


	print(heuristic_DisjointPatternDB(state))