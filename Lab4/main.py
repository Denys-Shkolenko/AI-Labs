import random


# Ініціалізувати матрицю Q нулями
def initialize_q_matrix(num_states, num_actions):
    return [[0] * num_actions for _ in range(num_states)]


# Виконати крок алгоритму Q-навчання
def q_learning_step(q_matrix, r_matrix, state, gamma, epsilon):
    # num_actions = len(q_matrix[0])
    action = choose_action(q_matrix, r_matrix, state, epsilon)

    # Отримати винагороду за вибрану дію
    reward = get_reward(state, action)

    # Отримати наступний стан після виконання дії
    next_state = action

    # Оновити матрицю Q
    q_matrix[state][action] = reward + gamma * max(q_matrix[next_state])

    return next_state, q_matrix


def shortest_way(q_matrix, r_matrix, state, num_states):
    way = []
    while state != num_states - 1:
        state = choose_action(q_matrix, r_matrix, state, -1)
        way.append(state)
    return way


# Вибрати дію з використанням епсилон-жадібного підходу
def choose_action(Q, R, state, epsilon):
    possible_actions = [action for action, reward in enumerate(R[state]) if reward != -1]
    if random.random() < epsilon:
        action = random.choice(possible_actions)
    else:
        action = max(possible_actions, key=lambda x: Q[state][x])
    return action


# Отримати винагороду за виконану дію в стані
def get_reward(state, action):
    return R[state][action]


def print_matrix(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Виведення нумерації стовпців
    header = "     "
    for j in range(num_cols):
        header += f"{j:8d}"
    print(header)

    # Виведення рядків матриці з нумерацією
    for i in range(num_rows):
        row = f"{i:3d} |"
        for j in range(num_cols):
            row += f"{matrix[i][j]:8.2f}"
        print(row)


if __name__ == "__main__":

    # Матриця R
    R = [
        [-1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, 0, 0, -1, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, 0, 0, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 100],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100]
    ]

    # Параметри алгоритму Q-навчання
    num_states = len(R)
    num_actions = len(R[0])
    gamma = 0.8
    epsilon = 0.2
    num_episodes = 100_000

    # Ініціалізувати матрицю Q
    Q = initialize_q_matrix(num_states, num_actions)

    # Виконати задану кількість епізодів
    for episode in range(num_episodes):
        # Початковий стан агента
        state = random.randint(0, num_states - 1)

        # Виконати епізод Q-навчання
        while state != num_states - 1:  # Поки агент не досягне цільового стану
            state, Q = q_learning_step(Q, R, state, gamma, epsilon)

    # Вивести отриману матрицю Q
    print("Q-Matrix:")
    print_matrix(Q)

    way = shortest_way(Q, R, 0, num_states)
    print("\n\nResult:", ' -> '.join(str(state) for state in way))
