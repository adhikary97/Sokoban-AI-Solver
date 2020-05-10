import heapq, collections, os, signal, datetime, random

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def start(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def goalp(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def expand(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# A* search algorithm
class AStarSearch(SearchAlgorithm):

    def __init__(self, heuristic, verbose=0):
        self.verbose = verbose
        self.heuristic = heuristic

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0
        self.finalCosts = collections.defaultdict(lambda:float('inf'))

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.start()
        self.finalCosts[startState] = 0
        frontier.update(startState, self.heuristic(startState))
        estCostNotified = 0

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, estimatedCost = frontier.removeMin()

            if state == None: break

            pastCost = self.finalCosts[state]

            if (self.verbose >= 1 and estimatedCost > estCostNotified) or (self.verbose >= 2 and random.randint(0,1000)==0):
                print('estimatedCost {} started, {} states expanded, sample state is {}'.format(estimatedCost, self.numStatesExplored, state.data))
                if self.verbose >= 2:
                    print('h value is {}'.format(self.heuristic(state)))
                    while state != startState:
                        action, prevState = backpointers[state]
                        print('   from action {}'.format(action))
                        state = prevState
                estCostNotified = estimatedCost

            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(("Exploring %s with pastCost %s and estimated cost %s" % (state, pastCost, estimatedCost)))

            # Check if we've reached an end state; if so, extract solution.
            if problem.goalp(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print(("numStatesExplored = %d" % self.numStatesExplored))
                    print(("totalCost = %s" % self.totalCost))
                    print(("actions = %s" % self.actions))
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.expand(state):
                if self.verbose >= 3:
                    print(("  Action %s => %s with cost %s + %s + %s" % (action, newState, pastCost, cost, estimatedCost)))
                newPastCost = pastCost + cost
                self.finalCosts[newState] = min(newPastCost,self.finalCosts[newState])

                if frontier.update(newState, newPastCost + self.heuristic(newState)):
                # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print("No path found")

############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).
class UniformCostSearch(AStarSearch):
    def __init__(self, verbose = 0):
        self.heuristic = lambda _ : 0
        self.verbose = verbose

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...

# Run a function, timing out after maxSeconds.
class TimeoutFunctionException(Exception):
    pass
class TimeoutFunction:
    def __init__(self, function, maxSeconds):
        self.maxSeconds = maxSeconds
        self.function = function

    def handle_maxSeconds(self, signum, frame):
        print('TIMEOUT!')
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if os.name == 'nt':
            # Windows does not have signal.SIGALRM
            # Will not stop after maxSeconds second but can still throw an exception
            timeStart = datetime.datetime.now()
            result = self.function(*args)
            timeEnd = datetime.datetime.now()
            if timeEnd - timeStart > datetime.timedelta(seconds=self.maxSeconds + 1):
                raise TimeoutFunctionException()
            return result
            # End modification for Windows here
        old = signal.signal(signal.SIGALRM, self.handle_maxSeconds)
        signal.alarm(self.maxSeconds + 1)
        result = self.function(*args)
        signal.alarm(0)
        return result
