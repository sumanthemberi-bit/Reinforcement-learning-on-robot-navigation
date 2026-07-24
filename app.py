"""
Robot Navigation with Reinforcement Learning - Streamlit Web App
Day 6: converts the Jupyter prototype (Days 1-5) into an interactive web app.

Run with:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import time
import streamlit as st

# ----------------------------------------------------------------------
# 1. PAGE CONFIG + CONSTANTS
# ----------------------------------------------------------------------
st.set_page_config(page_title="RL Robot Navigation", page_icon="🤖", layout="wide")

GRID_SIZE = 10
ACTIONS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}
ACTION_NAMES = list(ACTIONS.keys())
ARROW_VECTORS = {
    "up":    (0, -0.3),
    "down":  (0, 0.3),
    "left":  (-0.3, 0),
    "right": (0.3, 0),
}
cmap = ListedColormap(["white", "black", "limegreen", "royalblue"])


# ----------------------------------------------------------------------
# 2. ENVIRONMENT + Q-LEARNING FUNCTIONS (same logic as the notebooks)
# ----------------------------------------------------------------------
def step(robot_pos, action, goal_pos, obstacles, size=GRID_SIZE):
    dr, dc = ACTIONS[action]
    new_pos = (robot_pos[0] + dr, robot_pos[1] + dc)
    out_of_bounds = not (0 <= new_pos[0] < size and 0 <= new_pos[1] < size)
    if out_of_bounds:
        return robot_pos, -5, False
    if new_pos in obstacles:
        return robot_pos, -5, False
    if new_pos == goal_pos:
        return new_pos, 10, True
    return new_pos, -1, False


def choose_action(state, q_table, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTION_NAMES)
    row, col = state
    q_values = q_table[row, col]
    max_q = np.max(q_values)
    best_indices = np.flatnonzero(q_values == max_q)
    best_index = random.choice(best_indices)
    return ACTION_NAMES[best_index]


def update_q(q_table, state, action, reward, next_state, alpha, gamma):
    row, col = state
    action_index = ACTION_NAMES.index(action)
    next_row, next_col = next_state
    best_next_q = np.max(q_table[next_row, next_col])
    current_q = q_table[row, col, action_index]
    target = reward + gamma * best_next_q
    q_table[row, col, action_index] = current_q + alpha * (target - current_q)
    return q_table


def train(goal_pos, obstacles, episodes, max_steps, alpha, gamma,
          epsilon_start, epsilon_decay, epsilon_min):
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    epsilon = epsilon_start
    episode_rewards = []

    for _ in range(episodes):
        robot_pos = (0, 0)
        total_reward = 0
        for _ in range(max_steps):
            action = choose_action(robot_pos, q_table, epsilon)
            new_pos, reward, done = step(robot_pos, action, goal_pos, obstacles)
            q_table = update_q(q_table, robot_pos, action, reward, new_pos, alpha, gamma)
            robot_pos = new_pos
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_rewards


def get_trained_path(q_table, goal_pos, obstacles, max_steps):
    robot_pos = (0, 0)
    path = [robot_pos]
    for _ in range(max_steps):
        action = choose_action(robot_pos, q_table, epsilon=0.0)
        robot_pos, _, done = step(robot_pos, action, goal_pos, obstacles)
        path.append(robot_pos)
        if done:
            break
    return path


def is_goal_reachable(start, goal, obstacles, size=GRID_SIZE):
    """BFS flood-fill: returns True if goal can be reached from start avoiding obstacles."""
    from collections import deque
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for dr, dc in ACTIONS.values():
            neighbor = (current[0] + dr, current[1] + dc)
            in_bounds = 0 <= neighbor[0] < size and 0 <= neighbor[1] < size
            if in_bounds and neighbor not in obstacles and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


# ----------------------------------------------------------------------
# 3. SESSION STATE (Streamlit's memory across reruns/interactions)
# ----------------------------------------------------------------------
if "obstacles" not in st.session_state:
    st.session_state.obstacles = {
        (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6), (5, 7),
        (7, 1), (7, 2), (7, 3),
    }
if "goal_pos" not in st.session_state:
    st.session_state.goal_pos = (9, 9)
if "q_table" not in st.session_state:
    st.session_state.q_table = None
if "episode_rewards" not in st.session_state:
    st.session_state.episode_rewards = None
if "path" not in st.session_state:
    st.session_state.path = None
if "path_step" not in st.session_state:
    st.session_state.path_step = 0


# ----------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ----------------------------------------------------------------------
st.sidebar.header("Training settings")
episodes = st.sidebar.slider("Episodes", 100, 5000, 2000, step=100)
alpha = st.sidebar.slider("Learning rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount factor (gamma)", 0.0, 0.99, 0.9)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.90, 0.999, 0.995, format="%.3f")

st.sidebar.header("Grid editing")
st.sidebar.write("Click cells below to toggle obstacles on/off.")
if st.sidebar.button("Clear all obstacles"):
    st.session_state.obstacles = set()
if st.sidebar.button("Reset to default layout"):
    st.session_state.obstacles = {
        (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6), (5, 7),
        (7, 1), (7, 2), (7, 3),
    }

train_clicked = st.sidebar.button("Train robot", type="primary")


# ----------------------------------------------------------------------
# 5. MAIN LAYOUT: clickable grid editor
# ----------------------------------------------------------------------
st.title("Robot Navigation with Reinforcement Learning")
st.caption(
    "A robot learns to navigate a grid world using tabular Q-learning. "
    "Edit the grid, tune the hyperparameters, and watch it learn."
)
st.write(
    "Click cells to place obstacles, tune settings in the sidebar, "
    "then hit **Train robot**."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Grid editor")
    for row in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for col in range(GRID_SIZE):
            pos = (row, col)
            if pos == st.session_state.goal_pos:
                label = "G"
            elif pos == (0, 0):
                label = "S"
            elif pos in st.session_state.obstacles:
                label = "X"
            else:
                label = " "
            if cols[col].button(label, key=f"cell_{row}_{col}"):
                if pos not in ((0, 0), st.session_state.goal_pos):
                    if pos in st.session_state.obstacles:
                        st.session_state.obstacles.discard(pos)
                        st.rerun()
                    else:
                        candidate_obstacles = st.session_state.obstacles | {pos}
                        if is_goal_reachable((0, 0), st.session_state.goal_pos, candidate_obstacles):
                            st.session_state.obstacles.add(pos)
                            st.rerun()
                        else:
                            st.warning(
                                f"Can't place an obstacle at {pos} -- it would completely "
                                "block the path to the goal."
                            )

with right:
    st.subheader("Live preview")
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for obs in st.session_state.obstacles:
        grid[obs] = 1
    grid[st.session_state.goal_pos] = 2
    grid[0, 0] = 3
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    st.pyplot(fig)


# ----------------------------------------------------------------------
# 6. TRAINING TRIGGER
# ----------------------------------------------------------------------
if train_clicked:
    with st.spinner("Training..."):
        q_table, episode_rewards = train(
            goal_pos=st.session_state.goal_pos,
            obstacles=st.session_state.obstacles,
            episodes=episodes,
            max_steps=100,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.05,
        )
        st.session_state.q_table = q_table
        st.session_state.episode_rewards = episode_rewards
        st.session_state.path = get_trained_path(
            q_table, st.session_state.goal_pos, st.session_state.obstacles, max_steps=100
        )
        st.session_state.path_step = 0
    st.success("Training complete!")


# ----------------------------------------------------------------------
# 7. RESULTS: reward curve, heatmap + policy, path playback
# ----------------------------------------------------------------------
if st.session_state.q_table is not None:
    st.divider()
    st.subheader("Results")

    tab1, tab2, tab3 = st.tabs(["Learning curve", "Q-value heatmap", "Watch the robot"])

    with tab1:
        rewards = st.session_state.episode_rewards
        window = 20
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rewards, alpha=0.3, label="Raw reward per episode")
        if len(rewards) >= window:
            rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), rolling_avg, color="red",
                     label=f"{window}-episode rolling average")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        value_map = np.max(st.session_state.q_table, axis=2)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(value_map, cmap="viridis")
        fig.colorbar(im, ax=ax, label="Best learned Q-value")
        for obs in st.session_state.obstacles:
            ax.scatter(obs[1], obs[0], marker="x", color="red", s=100)
        ax.scatter(st.session_state.goal_pos[1], st.session_state.goal_pos[0],
                    marker="*", color="gold", s=300, edgecolors="black")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row, col) in st.session_state.obstacles or (row, col) == st.session_state.goal_pos:
                    continue
                best_index = np.argmax(st.session_state.q_table[row, col])
                best_action = ACTION_NAMES[best_index]
                dx, dy = ARROW_VECTORS[best_action]
                ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.15, color="white")
        st.pyplot(fig)

    with tab3:
        path = st.session_state.path
        st.write(f"Trained path length: {len(path) - 1} steps")

        speed = st.slider("Animation speed (seconds per step)", 0.05, 1.0, 0.3, key="anim_speed")
        play_clicked = st.button("▶ Play animation", key="play_button")

        frame_placeholder = st.empty()

        def draw_frame(pos, step_i, total_steps):
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for obs in st.session_state.obstacles:
                grid[obs] = 1
            grid[st.session_state.goal_pos] = 2
            grid[pos] = 3
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
            ax.set_title(f"Step {step_i}/{total_steps}")
            ax.set_xticks(range(GRID_SIZE))
            ax.set_yticks(range(GRID_SIZE))
            frame_placeholder.pyplot(fig)
            plt.close(fig)  # free memory -- without this, matplotlib keeps every figure in memory

        if play_clicked:
            for step_i, pos in enumerate(path):
                draw_frame(pos, step_i, len(path) - 1)
                time.sleep(speed)
        else:
            draw_frame(path[0], 0, len(path) - 1)
else:
    st.info("Set up your grid and click **Train robot** in the sidebar to get started.")
