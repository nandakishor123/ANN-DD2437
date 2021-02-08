import numpy as np

def get_data_matrix():
    with open("data_lab2/animals.dat","r") as bf:
        data = bf.read()
    data2 = data.split(",")
    data_matrix = np.zeros((32,84))
    for i in range(data_matrix.shape[0] - 1):
        for j in range(data_matrix.shape[1] - 1):
            position = i * data_matrix.shape[1] + j
            data_matrix[i][j] = data2[position]
    return data_matrix #returning the dataset created

def get_animal_names():
    with open("data_lab2/animalnames.txt","r") as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    return names

def update_radius(radius, i ,time):
    d_r = radius * np.exp(-i/time)
    return d_r

def update_learning_rate(init_learningrate, epochs, i):
    d_learning_rate = init_learningrate * np.exp(-i/epochs)
    return d_learning_rate

def calculate_influence(distance, radius):
    influence = np.exp(-distance/(2* (radius**2)))
    return influence

def find_unit(t,net):
    #t represents the target features
    #net represents the weight matrix

    node_index = []
    min_dist = np.iinfo(np.int).max

    for x in range(net.shape[0]):
        w = net[x].reshape(1,84) #converting the reight matrix in column vector form to row vector form
        distance = np.sum((w - t)**2)

        if distance < min_dist:
            min_dist = distance
            node_index = x

    node = net[node_index].reshape(84,1)
    return (node,node_index)

def main():
    animals_data = get_data_matrix()
    epochs = 20
    learning_rate = 0.2
    radius = 50
    time = epochs/np.log(radius)
    net = np.random.random((100, 84)) #weight matrix of the prescribed size

    for i in range(epochs):
        for j in range(len(animals_data)):

            row = animals_data[j][:]
            node,index = find_unit(row,net) #passing the weight matrix and the features to find the best node

            r = update_radius(radius, i, time)
            lr = update_learning_rate(learning_rate, epochs, i)

            #after finding the best node and its, index, we go on to update the weights corresponding to that node only.

            for k in range(net.shape[0]):
                w = net[k].reshape(1,84)
                w_dist = np.sum((k - index) **2)

                if w_dist <= r**2:
                    influence = calculate_influence(w_dist,r)
                    new_w = w + (lr * influence * (row - w))
                    net[k] = new_w[0].reshape(1,84)
    
    #indexing of the target species
    names = get_animal_names()
    pos = []
    index_names = []

    for i in range(len(animals_data)):
        row = animals_data[i][:]
        node,index = find_unit(row,net)
        index_names.append(i)
        pos.append(index)
    
    sorted_pos = sorted(zip(pos,index_names))
    print(sorted_pos)
    names_sorted = []

    for i in range(len(names)):
        pos = sorted_pos[i][1]
        names_sorted.append(names[pos])
        print(names_sorted[i])

if __name__ == "__main__":
    main()