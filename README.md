# Game-Playing AI Agent: Optimized Decision-Making

This is a team-based higher-level coursework project. This is a mirror repository of the original (see COMP424_Project).

This project is an exploratory, competitive challenge to create the most effective game-playing agent for the game Attax. Student agents were matched randomly against one another in Attaxx combat, testing the efficiency of decision-making algorithms and their implementations. 

**Attaxx Briefly:**

_Attaxx is a game between two opponents (agents), played with a random game board of size N × N and two different colored sets of game pieces per player (in this case Blue and Brown)._ 

- The goal for each player is to place as many of its own pieces on the game board while minimizing the presence of the opponent's pieces.
- The game is over when one of the player's has no more pieces left to place on the board.
- The winner is the player that outnumbers the other by the count of pieces present on the board when the game is over.
- 
_Agents can choose between two types of moves - duplication or jumping._

- Duplication allows for the player to duplicate any piece, placing the new piece adjacent to the original piece (horizontally, vertically or diagonally adjacent square), while the original piece remains in its initial square.
- Jumping allows for the player to move a piece 2 squares away from its original location, in the horizontal, vertical, or diagonal direction.
- After the move has been taken, if there are any opponent pieces adjacent to the newly placed or relocated piece, they will be converted to the current player's color.

Fundamentally, the Student agent is designed to (minimally) outperform a Random agent (every decision is random) and the Greedy Corners Agent - implements a greedy algorithm that focuses primarily on covering the corners of the game board with its pieces. Per requirements of inter-agent combat, the turn out time per decision step for the Student agent must also be under 2 seconds - this was a key design consideration during development.

**Agent Versions Developed:** 

- Minimax with α − β Pruning
- Hybrid Minimax MCTS Step Agent
- Early Game Minimax Late Game MCTS Agent
- Minimax Transposition Table and Zobrist

_For detailed analysis of each agent's behavior and performance, refer to the "Game Playing AI Agent: Optimized Decision-Making Techniques" paper._

**General Overview of Agents**

- All agent versions developed by the team consistently outperformed both random agent and Greedy Corners agent (default agents).
- All versions used heuristics which included the heuristics of the Greedy Corners agent, with additional specifications for improved performance.
- The primary challenge during development was not ensuring that the Student agent outperforms default agents, but rather on improving the efficiency of execution - minimizing turn out time per decision (keeping it consistently under 2 seconds).

The best performing agent implemented a hybrid approach, combining Minimax optimized with alpha-beta pruning, iterative deepening, and a transposition table using Zobrist hashing.

- **Alpha-beta pruning** reduced the search space by preemptively eliminating branches that do not change the outcome. This effectively reduced noise, allowing the agent to focus on proactive decision-making.
- **Iterative deepening** is a search algorithm incrementally increases search depth, thus putting a reasonable restriction on search time - ensuring that decision-making is within the 2 second limit.
- The **transposition table with Zobrist hashing** is an optimization which prevents the agent from repeating previously explored board positions, thus increasing the efficiency of decision-making and reducing suboptimal behavior.

## Setup

_The project is Python-based, set up inside a virtual environment._

```bash
python3 -m venv venv
```
and then:
```
source venv/bin/activate
```

_Game setup requires cloning this repository and installing the dependencies inside a virtual environment._

```bash
pip install -r requirements.txt
```

## Playing a game

A game between two agents, when initialized, can be traced through the console move by move, in the order that the players take turns making moves.

_For example:_

```bash
python simulator.py --player_1 random_agent --player_2 random_agent
```

## Visualizing a game

A game between two agents can be visualized using the `--display` flag, slowing the game for better analysis with `--display_delay`.

_For example:_

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --display
```

## Play on your own!

It is also possible to try manually playing (as a human agent):

_For example:_

```bash
python simulator.py --player_1 human_agent --player_2 random_agent --display
```

## Autoplaying multiple games

_For example:_

```bash
python simulator.py --player_1 random_agent --player_2 random_agent --autoplay
```

## Full API

```bash
python simulator.py -h       
usage: simulator.py [-h] [--player_1 PLAYER_1] [--player_2 PLAYER_2]
                    [--board_path BOARD_PATH] [--display]
                    [--display_delay DISPLAY_DELAY]

optional arguments:
  -h, --help            show this help message and exit
  --player_1 PLAYER_1
  --player_2 PLAYER_2
  --board_path BOARD_PATH
  --board_roster_dir BOARD_ROSTER_DIR
  --display
  --display_delay DISPLAY_DELAY
  --autoplay
  --autoplay_runs AUTOPLAY_RUNS
```

## Maximum Memory Measurements
_Measured using:_
```
/usr/bin/time -v python simulator.py --player_1 random_agent --player_2 <your-agent> --autoplay
```

_Example output in the console:_
```
Command being timed: "python simulator.py --player_1 random_agent --player_2 random_agent --autoplay"
        User time (seconds): 8.77
        System time (seconds): 0.28
        Percent of CPU this job got: 35%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:25.60
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 61056
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 188
        Minor (reclaiming a frame) page faults: 23247
        Voluntary context switches: 1682
        Involuntary context switches: 687
        Swaps: 0
        File system inputs: 48512
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```
where the max resident set size is the most RAM that has been used (in kilobytes).

## About

This is a class project for COMP 424, McGill University, Fall 2025 (it was originally forked with the permission of Jackie Cheung and David Meger).

## License

[MIT](LICENSE)
