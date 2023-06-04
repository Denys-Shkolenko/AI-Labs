import random


def q_learning_step(q_matrix, r_matrix, state, gamma, epsilon):
    action = choose_action(q_matrix, r_matrix, state, epsilon)
    reward = r_matrix[state][action]
    next_state = action
    q_matrix[state][action] = reward + gamma * max(q_matrix[next_state])
    return next_state, q_matrix


def shortest_way(q_matrix, r_matrix, state, num_states):
    way = []
    while state != num_states - 1:
        state = choose_action(q_matrix, r_matrix, state, -1)
        way.append(state)
    return way


def choose_action(Q, R, state, epsilon):
    possible_actions = [action for action, reward in enumerate(R[state]) if reward != -1]
    if random.random() < epsilon:
        action = random.choice(possible_actions)
    else:
        action = max(possible_actions, key=lambda x: Q[state][x])
    return action


from tabulate import tabulate

def print_matrix(matrix):
    table = [[i] + row for i, row in enumerate(matrix)]
    print(tabulate(table, headers=list(range(len(matrix[0]))), tablefmt="grid"))


# def print_matrix(matrix):
#     num_rows = len(matrix)
#     num_cols = len(matrix[0])
#
#     header = "     "
#     for j in range(num_cols):
#         header += f"{j:7d}"
#     print(header)
#
#     for i in range(num_rows):
#         row = f"{i:3d} |"
#         for j in range(num_cols):
#             row += f"{matrix[i][j]:7.2f}"
#         print(row)


if __name__ == "__main__":
    R = [
        [-1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1],
        [0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, 0, 0, 100],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 100]
    ]

    num_states = len(R)
    num_actions = len(R[0])
    gamma = 0.8
    epsilon = 0.2
    num_episodes = 100_000

    Q = [[0] * num_actions for _ in range(num_states)]

    for episode in range(num_episodes):
        state = random.randint(0, num_states - 1)
        while state != num_states - 1:
            state, Q = q_learning_step(Q, R, state, gamma, epsilon)

    print("Q-Matrix:")
    print_matrix(Q)

    way = shortest_way(Q, R, 0, num_states)
    print("\n\nResult:", ' -> '.join(str(state) for state in way))
