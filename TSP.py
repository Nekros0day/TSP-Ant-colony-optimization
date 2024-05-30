import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from PIL import Image

def compute_delta_pheromone_levels(path_collection, path_length_collection):
    number_of_cities = path_collection.shape[1]
    delta_pheromone_level = np.zeros((number_of_cities, number_of_cities))

    for k in range(len(path_collection)):
        ant_path = path_collection[k]
        path_length = path_length_collection[k]
        update_delta_pheromone(ant_path, path_length, delta_pheromone_level)
    
    return delta_pheromone_level

def update_delta_pheromone(ant_path, path_length, delta_pheromone_level):
    for i in range(1, len(ant_path)):
        delta_pheromone_level[ant_path[i], ant_path[i-1]] += 1 / path_length
    delta_pheromone_level[ant_path[0], ant_path[-1]] += 1 / path_length

def generate_path(pheromone_level, visibility, alpha, beta):
    number_of_cities = pheromone_level.shape[0]
    cities = np.arange(number_of_cities)
    tabu_list = np.zeros(number_of_cities, dtype=int)
    current_city = np.random.randint(number_of_cities)
    accumulated_prob = 0

    for path_index in range(number_of_cities):
        tabu_list[path_index] = current_city
        available_cities = np.setdiff1d(cities, tabu_list[:path_index + 1])
        
        if len(available_cities) == 0:
            break
        
        normalization_constant = np.sum((pheromone_level[current_city, available_cities] ** alpha) * 
                                        (visibility[current_city, available_cities] ** beta))
        
        random_threshold = np.random.rand()

        for next_city in available_cities:
            accumulated_prob += (pheromone_level[current_city, next_city] ** alpha * 
                                 visibility[current_city, next_city] ** beta) / normalization_constant
            
            if accumulated_prob >= random_threshold:
                current_city = next_city
                accumulated_prob = 0
                break
    
    return tabu_list

def get_path_length(path, city_location):
    total_cities = len(path)
    path_length = 0
    
    for index in range(total_cities - 1):
        path_length += np.linalg.norm(city_location[path[index + 1]] - city_location[path[index]])
    path_length += np.linalg.norm(city_location[path[0]] - city_location[path[-1]])
    
    return path_length

def get_visibility(city_location):
    total_cities = city_location.shape[0]
    visibility_matrix = np.zeros((total_cities, total_cities))

    for city1 in range(total_cities):
        for city2 in range(total_cities):
            if city1 != city2:
                visibility_matrix[city1, city2] = 1 / np.linalg.norm(city_location[city1] - city_location[city2])
    
    return visibility_matrix

def initialize_pheromone_levels(number_of_cities, tau0):
    return np.ones((number_of_cities, number_of_cities)) * tau0

def update_pheromone_levels(pheromone_level, delta_pheromone_level, rho):
    pheromone_level = pheromone_level * (1 - rho) + delta_pheromone_level
    return np.maximum(pheromone_level, 1e-15)

def initialize_connections(city_location, ax):
    connections = []
    for _ in range(len(city_location)):
        line, = ax.plot([0, 0], [0, 0], 'k-')
        connections.append(line)
    return connections

def initialize_tsp_plot(city_location, plot_range):
    fig, ax = plt.subplots()
    ax.set_xlim(plot_range[0], plot_range[1])
    ax.set_ylim(plot_range[2], plot_range[3])
    ax.set_aspect('equal')
    ax.grid(True)
    
    for loc in city_location:
        circle = plt.Circle((loc[0], loc[1]), 0.25, color='blue', fill=True)
        ax.add_patch(circle)
    
    return fig, ax

def plot_path(connections, city_location, path):
    n_cities = len(city_location)
    for i in range(n_cities - 1):
        connections[i].set_data([city_location[path[i]][0], city_location[path[i + 1]][0]],
                                [city_location[path[i]][1], city_location[path[i + 1]][1]])
    connections[n_cities - 1].set_data([city_location[path[n_cities - 1]][0], city_location[path[0]][0]],
                                       [city_location[path[n_cities - 1]][1], city_location[path[0]][1]])
    
    plt.draw()
    plt.pause(0.01)

def save_plot_as_image(fig, iteration, ant, path_length, image_list):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_image = Image.fromarray(image)
    image_list.append(pil_image)

def load_city_locations():
    city_location = np.array([
        [16.303268138034628, 7.129182064109691],
        [12.316757719315326, 4.409627113373858],
        [13.072546875817137, 8.017714836151125],
        [15.604832582681968, 2.158974345696600],
        [17.401803390885384, 6.399746847414304],
        [13.222294712794524, 6.594405739845993],
        [9.651401439940274, 4.101198651555478],
        [7.266945531506188, 8.470221602515906],
        [14.367375359988275, 4.222132128343533],
        [14.962859684002455, 4.269858674316712],
        [11.938590948584864, 7.758537705489252],
        [13.001422207815457, 5.520082873106608],
        [15.180725163634072, 5.258032028577626],
        [11.332145442942258, 3.530621804929355],
        [13.873312875455952, 6.423880384388132],
        [5.948397479359290, 6.408929630273879],
        [12.356352045729189, 2.557651117381416],
        [17.524281068057377, 8.241349458996412],
        [9.534118260722675, 5.841985848136163],
        [9.493230302183955, 8.800041640977472],
        [4.243872865253579, 5.316023998406051],
        [11.445993868754178, 2.772317410483613],
        [13.465846625601467, 2.374006372866969],
        [12.549882381444197, 3.250727104012847],
        [15.022876660629374, 12.847002557639119],
        [17.258529041413983, 11.958641191296660],
        [3.118241180447490, 16.687427462471540],
        [7.547901301092515, 17.017342818783430],
        [9.893970539434953, 12.558717616714546],
        [7.242966601059464, 14.023109389126208],
        [17.049919159341542, 16.674004596148698],
        [13.306955569788013, 12.120028543265505],
        [5.543160453559379, 17.177785436628639],
        [7.390442331627048, 15.586007557543070],
        [13.111798155790638, 18.390143470927349],
        [5.433830846448841, 12.060228766444126],
        [4.965378753107088, 15.105833965236206],
        [6.139801873343221, 18.582953994853849],
        [2.287305867377466, 16.851273103617167],
        [2.668562789326542, 14.140342876975094],
        [12.757383935685304, 16.085879435855677],
        [6.995577348870533, 16.001123126527443],
        [16.047806952644937, 18.821217829297055],
        [1.717066140146145, 15.654990804361987],
        [4.050481441345405, 18.831727526250820],
        [8.873641846107201, 17.251578989266683],
        [8.410528045394368, 14.710165504625886],
        [6.303890542990022, 17.231646760185381],
        [5.735038907052351, 14.282577667127638],
        [12.060203772802080, 11.427613305038985]
    ])
    return city_location

# Main
city_location = load_city_locations()
number_of_cities = len(city_location)

# Parameters
number_of_ants = 50
alpha = 1.0
beta = 3.0
rho = 0.3
tau0 = 0.1
target_path_length = 99.9999999

# Initialization
plot_range = [0, 20, 0, 20]
fig, ax = initialize_tsp_plot(city_location, plot_range)
connections = initialize_connections(city_location, ax)
pheromone_level = initialize_pheromone_levels(number_of_cities, tau0)
visibility = get_visibility(city_location)

# Main loop
minimum_path_length = float('inf')
i_iteration = 0
path_collection = np.zeros((number_of_ants, number_of_cities), dtype=int)
path_length_collection = np.zeros(number_of_ants)
image_list = []

while minimum_path_length > target_path_length:
    i_iteration += 1

    # Generate paths
    for k in range(number_of_ants):
        path = generate_path(pheromone_level, visibility, alpha, beta)
        path_length = get_path_length(path, city_location)
        if path_length < minimum_path_length:
            minimum_path_length = path_length
            print(f"Iteration {i_iteration}, ant {k}: path length = {minimum_path_length:.5f}")
            plot_path(connections, city_location, path)
            save_plot_as_image(fig, i_iteration, k, path_length, image_list)
        path_collection[k] = path
        path_length_collection[k] = path_length

    # Update pheromone levels
    delta_pheromone_level = compute_delta_pheromone_levels(path_collection, path_length_collection)
    pheromone_level = update_pheromone_levels(pheromone_level, delta_pheromone_level, rho)

# Save images as GIF
image_list[0].save('tsp_solution.gif', save_all=True, append_images=image_list[1:], duration=300, loop=0)

plt.show()
