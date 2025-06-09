import numpy as np
# Define the criteria
num_criteria = 6
criteria = ['Navigation Efficiency', 'Payload Capacity', 'Safety', 'Communication Effectiveness', 'Adaptability', 'Energy Efficiency']

# Define the task types
task_types = ['Scan and Rescue', 'Supply Necessities', 'Lifting Remains']

# Define the robot types
robot_types = ['Drone', 'Ground Robot', 'Creeper Robot']

# AHP for weight calculation of criteria
criteria_comparison_matrix = np.zeros((num_criteria, num_criteria))
for i in range(num_criteria):
    for j in range(i+1, num_criteria):
        while True:
            comparison = input(f"Enter pairwise comparison between {criteria[i]} and {criteria[j]} (e.g., 3 or 1/3): ")
            try:
                if '/' in comparison:
                    numerator, denominator = map(float, comparison.split('/'))
                    comparison_value = numerator / denominator
                else:
                    comparison_value = float(comparison)
                criteria_comparison_matrix[i, j] = np.round(comparison_value, 4)
                criteria_comparison_matrix[j, i] = np.round(1 / comparison_value, 4)
                break
            except ValueError:
                print("Invalid input. Please enter a number or a fraction (e.g., 3 or 1/3).")
for i in range(num_criteria):
    criteria_comparison_matrix[i, i] = 1

# Calculate the weights of the criteria using AHP
eigenvalues, eigenvectors = np.linalg.eig(criteria_comparison_matrix)
max_eigenvalue = np.max(eigenvalues)
criteria_weights = eigenvectors[:, np.argmax(eigenvalues)]
criteria_weights = np.round(criteria_weights / np.sum(criteria_weights), 4)
print("Criteria Weights:")
print(criteria_weights)

# Define the robots
num_robots = int(input("Enter the number of robots: "))
robots = []
drone_id = 1
ground_robot_id = 1
creeper_robot_id = 1
for i in range(num_robots):
    robot_type = input(f"Enter the type of robot {i+1} (Drone, Ground Robot, Creeper Robot): ")
    while robot_type not in robot_types:
        print("Invalid robot type. Please enter one of the following: Drone, Ground Robot, Creeper Robot")
        robot_type = input(f"Enter the type of robot {i+1} (Drone, Ground Robot, Creeper Robot): ")
    values = input(f"Enter the values for {robot_type} {i+1} (6 values separated by spaces): ")
    values = [np.round(float(x), 4) for x in values.split()]
    if robot_type == 'Drone':
        robot_id = f"D{drone_id}"
        drone_id += 1
    elif robot_type == 'Ground Robot':
        robot_id = f"G{ground_robot_id}"
        ground_robot_id += 1
    else:
        robot_id = f"C{creeper_robot_id}"
        creeper_robot_id += 1
    robots.append({'ID': robot_id, 'Type': robot_type, 'Values': values})

# Define the tasks
num_tasks = int(input("Enter the number of tasks: "))
tasks = []
for i in range(num_tasks):
    task_type = input(f"Enter the type of task {i+1} (Scan and Rescue, Supply Necessities, Lifting Remains): ")
    while task_type not in task_types:
        print("Invalid task type. Please enter one of the following: Scan and Rescue, Supply Necessities, Lifting Remains")
        task_type = input(f"Enter the type of task {i+1} (Scan and Rescue, Supply Necessities, Lifting Remains): ")
    scores = input(f"Enter scores for Task {i+1} (6 values separated by spaces on a scale of 1-9): ")
    scores = [np.round(float(x), 4) for x in scores.split()]
    tasks.append({'Type': task_type, 'Scores': scores})

# Calculate task priorities
task_priorities = {}
for task in tasks:
    priority = np.round(sum(task['Scores'][i] * criteria_weights[i] for i in range(num_criteria)), 4)
    task_priorities[task['Type']] = priority

# Sort tasks by priority
sorted_tasks = sorted(task_priorities.items(), key=lambda x: x[1], reverse=True)
print("\nTask Priorities:")
for task, priority in sorted_tasks:
    print(f"{task}: {priority}")

# Calculate decision matrix
decision_matrix = np.array([robot['Values'] for robot in robots])
# Calculate normalized decision matrix
normalized_decision_matrix = decision_matrix / np.max(decision_matrix, axis=0)

# Calculate weighted normalized decision matrix
weighted_normalized_decision_matrix = normalized_decision_matrix * criteria_weights

# Calculate ideal best and ideal worst solutions
ideal_best = np.max(weighted_normalized_decision_matrix, axis=0)
ideal_worst = np.min(weighted_normalized_decision_matrix, axis=0)

# Calculate distance from ideal best and ideal worst solutions
distance_best = np.sqrt(np.sum((weighted_normalized_decision_matrix - ideal_best) ** 2, axis=1))
distance_worst = np.sqrt(np.sum((weighted_normalized_decision_matrix - ideal_worst) ** 2, axis=1))

# VIKOR index calculations
S = distance_best
R = np.max(np.abs(weighted_normalized_decision_matrix - ideal_best), axis=1)

S_best = np.min(S)
S_worst = np.max(S)
R_best = np.min(R)
R_worst = np.max(R)

v = 0.5  # weight of the strategy of the majority of criteria
Q = v * (S - S_best) / (S_worst - S_best) + (1-v) * (R - R_best) / (R_worst - R_best)

# Robot rankings
robot_rankings = np.argsort(Q)
print("\nRobot Rankings:")
for i, ranking in enumerate(robot_rankings):
    print(f"Rank {i+1}: {robots[ranking]['ID']} ({robots[ranking]['Type']})")

# Assign tasks to robots
available_robots = robots[:]
for task_type, task_priority in sorted_tasks:
    print(f"\n{task_type} Priority: {task_priority}")
    if task_type == 'Scan and Rescue':
        required_robots = [robot for robot in available_robots if robot['Type'] in ['Drone', 'Ground Robot']]
        if len(required_robots) < 2:
            print("Not enough robots available for Scan and Rescue task.")
            continue
        required_robots.sort(key=lambda x: np.round(sum(x['Values'][i] * criteria_weights[i] for i in range(num_criteria)), 4), reverse=True)
        print(f"Team for {task_type}:")
        team = [required_robots[0], required_robots[1]]
        for robot in team:
            print(f"{robot['ID']} ({robot['Type']})")
            available_robots.remove(robot)
    elif task_type == 'Supply Necessities':
        required_robots = [robot for robot in available_robots if robot['Type'] in ['Ground Robot', 'Creeper Robot', 'Drone']]
        if not required_robots:
            print("No robots available for Supply Necessities task.")
            continue
        required_robots.sort(key=lambda x: np.round(sum(x['Values'][i] * criteria_weights[i] for i in range(num_criteria)), 4), reverse=True)
        print(f"Team for {task_type}:")
        team = [required_robots[0]]
        for robot in team:
            print(f"{robot['ID']} ({robot['Type']})")
            available_robots.remove(robot)
    elif task_type == 'Lifting Remains':
        required_robots = [robot for robot in available_robots if robot['Type'] == 'Ground Robot']
        if len(required_robots) < 2:
            print("Not enough Ground Robots available for Lifting Remains task.")
            continue
        required_robots.sort(key=lambda x: np.round(sum(x['Values'][i] * criteria_weights[i] for i in range(num_criteria)), 4), reverse=True)
        print(f"Team for {task_type}:")
        team = [required_robots[0], required_robots[1]]
        for robot in team:
            print(f"{robot['ID']} ({robot['Type']})")
            available_robots.remove(robot)
print("\nCriteria Comparison Matrix:")
print(np.round(criteria_comparison_matrix, 4))
print("\nCriteria Weights:")
print(np.round(criteria_weights,4))
print("\nDecision Matrix:")
print(np.round(decision_matrix, 4))
print("\nNormalized Decision Matrix:")
print(np.round(normalized_decision_matrix, 4))
print("\nWeighted Normalized Decision Matrix:")
print(np.round(weighted_normalized_decision_matrix, 4))

# Display ideal best and worst solutions
print("\nIdeal Best Solution:")
print(np.round(ideal_best, 4))
print("\nIdeal Worst Solution:")
print(np.round(ideal_worst, 4))

# Display distance calculations
print("\nDistance from Ideal Best Solution:")
print(np.round(distance_best, 4))
print("\nDistance from Ideal Worst Solution:")
print(np.round(distance_worst, 4))

print("\nVIKOR Index (Q):")
print(np.round(Q, 4))

# Display Robot rankings
print("\nRobot Rankings:")
for i, ranking in enumerate(robot_rankings):
    print(f"Rank {i+1}: {robots[ranking]['ID']} ({robots[ranking]['Type']})")

# Display task priorities
print("\nTask Priorities:")
for task, priority in task_priorities.items():
    print(f"{task}: {np.round(priority, 4)}")
