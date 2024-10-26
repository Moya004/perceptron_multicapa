import numpy as np
import utilities as ut
import os
import matplotlib.pyplot as plt

def normalize_patterns(patterns: list) -> list:
    return [((2 * (pattern[0] - 1) / 23) - 1, 2 * ((pattern[1] - 1) / 6) - 1, pattern[2]) for pattern in patterns]

def build_Network(dimensions: tuple, activation: ut.FunctionType = ut.tansig) -> list:
    network = []
    for i in range(1, len(dimensions)):
        layer = []
        for _ in range(dimensions[i]):
            neuron = ut.Neuron(activation=activation, bias=np.random.uniform(-0.05, 0.05))
            layer.append(neuron)
        network.append(layer)
    return network

def build_weights(dimensions: tuple) -> list:
    matrix = []
    for i in range(1, len(dimensions)):
        layer = []
        for _ in range(dimensions[i]):
            weights = np.random.uniform(-0.05, 0.05, dimensions[i - 1])
            layer.append(weights)
        matrix.append(np.array(layer))
        
    return matrix

def update_weights(neurons: list, weights: list, learning_rate: float):
    for i in range(len(neurons)):
        for j in range(len(weights[i])):
            weights[i][j] += learning_rate * neurons[i].error * neurons[i].entries[j] * ut.dev_tansig(neurons[i].output)


def main():
    raw_data = np.array([
        [1.23, 1.6049, 1.663, 1.7299, 1.7129, 1.713, 1.414],
        [1.0889, 1.4389, 1.4689, 1.5159, 1.4569, 1.2821, 1.325],
        [1.0289, 1.3361, 1.387, 1.4342, 1.182, 1.2749, 1.329],
        [0.9879, 1.3559, 1.3751, 1.3931, 1.288, 1.172, 1.2289],
        [0.9879, 1.3439, 1.3611, 1.3909, 1.2331, 1.061, 1.124],
        [1.1105, 1.389, 1.414, 1.431, 1.1911, 1.1059, 1.0191],
        [1.3729, 1.5609, 1.604, 1.614, 1.157, 1.171, 0.9989],
        [1.6649, 1.775, 1.8009, 1.817, 1.17, 1.2751, 0.9989],
        [1.79, 2.018, 2.0739, 2.0989, 1.339, 1.4121, 0.979],
        [2.1569, 2.0799, 2.1301, 2.226, 1.337, 1.545, 1.015],
        [2.323, 2.3359, 2.3649, 2.381, 1.4799, 1.5831, 1.1271],
        [2.3659, 2.366, 2.399, 2.3741, 1.574, 1.741, 1.2771],
        [2.3731, 2.3359, 2.358, 2.3101, 1.5951, 1.7129, 1.295],
        [2.2311, 2.156, 2.2, 2.2, 1.5771, 1.62, 1.1459],
        [2.156, 2.0799, 2.1731, 2.1731, 1.5091, 1.451, 1.2999],
        [2.208, 2.1651, 2.1749, 2.088, 1.5629, 1.5831, 1.26],
        [2.2999, 2.2551, 2.2009, 2.1651, 1.545, 1.6251, 1.2669],
        [2.3741, 2.3671, 2.3349, 2.238, 1.638, 1.6251, 1.3611],
        [2.5, 2.477, 2.464, 2.464, 1.731, 1.9, 1.238],
        [2.484, 2.481, 2.479, 2.4, 1.748, 1.904, 1.612],
        [2.536, 2.354, 2.441, 2.1954, 1.8194, 1.98, 1.296],
        [2.0, 2.21, 2.004, 1.995, 1.562, 1.9131, 1.26],
        [1.989, 1.7085, 1.8582, 1.842, 1.5882, 1.958, 1.615],
        [1.808, 1.7, 1.7071, 1.793, 1.748, 1.49, 1.5651],
    ])
    max_value = np.max(raw_data)
    min_value = np.min(raw_data)
    patterns = [(fila + 1, columna + 1, valor) for fila, arr in enumerate(raw_data) for columna, valor in enumerate(arr)]
    patterns = normalize_patterns(patterns)
    traning_patterns_indeces = np.random.choice(len(patterns), int(len(patterns) * 0.7), replace=False)
    traning_patterns = [patterns[i] for i in traning_patterns_indeces]
    testing_patterns = [pattern for pattern in patterns if pattern not in traning_patterns]

    raw_dimensions = input("Ingrese las dimensiones de la red: ")
    input_dimensions = [int(layer) for layer in raw_dimensions.split()]
    red_dimensions = tuple([2] + input_dimensions + [1])

    red = build_Network(red_dimensions)
    weights = build_weights(red_dimensions)

    n = float(input("Ingrese el parametro n: "))                            #coeficiente de aprendizaje
    tol = 10 ** -3                                                          #tolerancia
    max_iter = 100                                                        #iteraciones maximas
    epocas = 0                                                              #contador de epocas
    err_pattern = [1 for _ in range(len(traning_patterns))]                 #error inicial
    err_by_epocas = []

    while ut.min_squares(err_pattern) >= tol and epocas < max_iter:
        err_by_epocas.append(ut.min_squares(err_pattern))
        if epocas % 50 == 0:    
            print(f"Epoca: {epocas}")
            print(f"Error: {ut.min_squares(err_pattern)}")
            print(f'Pesos: [')
            for weight in weights:
                print(f'{weight}\n')
            print(']')

        epocas += 1
        err_pattern = []
        for pattern in traning_patterns:
            inputs = pattern[:2]
            output = pattern[2]

            for neurons, weight in zip(enumerate(red), enumerate(weights)):
                match weight[0]:
                    case 0:
                        for neuron, w in zip(neurons[1], weight[1]):
                            neuron.set_entries(inputs)
                            neuron.calculate_output(w)
                    case _:
                        for neuron, w in zip(neurons[1], weight[1]):
                            neuron.set_entries([prev_neuron.output for prev_neuron in red[weight[0] - 1]])
                            neuron.calculate_output(w)
            
            denormalized_output = ( (red[-1][0].output + 1) * (max_value - min_value) ) / 2 + min_value
            err = output - denormalized_output
            err_pattern.append(err)

            for neurons, weight in zip(enumerate(reversed(red)), enumerate(reversed(weights))):
                match weight[0]:
                    case 0:
                        for neuron in neurons[1]:
                            neuron.set_error(err)
                            neuron.update_bias(n)

                    case _:
                        for i in range(len(neurons[1])):
                            neurons[1][i].update_error(np.matrix(weights[-weight[0]]).T[i], [neuron.error for neuron in red[-weight[0]]])
                            neurons[1][i].update_bias(n)

            new_weights = weights.copy()
            for neurons, weight in zip(enumerate(red), enumerate(weights)):
                match weight[0]:
                    case 0:
                        for i in range(len(weight[1])):
                            for j in range(len(weight[1][i])):
                               new_weights[weight[0]][i][j] += n * neurons[1][i].error * inputs[j] * ut.dev_tansig(neurons[1][i].output)
                    case _:
                        update_weights(neurons[1], new_weights[weight[0]], n)


    err_by_epocas.append(ut.min_squares(err_pattern))  
    print(f"Epoca: {epocas}")
    print(f"Error: {ut.min_squares(err_pattern)}")
    print(f'Pesos: [')
    for weight in weights:
        print(f'{weight}\n')
    print(']')
    err_by_epocas.pop(0)
    plt.xlabel('Epocas')
    plt.ylabel('Error')
    plt.title('Error por epoca')
    plt.plot(err_by_epocas, label='Error')
    plt.text(0.5, 0.9, f'n: {n}', transform=plt.gca().transAxes)
    plt.text(0.5, 0.85, f'Tolerancia: {tol}', transform=plt.gca().transAxes)
    plt.text(0.5, 0.8, f'Dimensión de la Red: {red_dimensions}', transform=plt.gca().transAxes)
    textstr = '\n'.join((
        f'epoca: {weights}',
        f'error: {err_by_epocas}',
    ))
    
    
    # Create directory if it doesn't exist
    directory = f'tests/{raw_dimensions}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot
    plt.savefig(f'{directory}/error.png')
    plt.show()
    day = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
    for i in range(7):
        test = []
        expected = []
        denormalized_hours = []
        for pattern in filter(lambda x: round((( (x[1] + 1) *  6) / 2) + 1) == i+1, testing_patterns):
            inputs = pattern[:2]
            output = pattern[2]

            for neurons, weight in zip(enumerate(red), enumerate(weights)):
                match weight[0]:
                    case 0:
                        for neuron, w in zip(neurons[1], weight[1]):
                            neuron.set_entries(inputs)
                            neuron.calculate_output(w)
                    case _:
                        for neuron, w in zip(neurons[1], weight[1]):
                            neuron.set_entries([prev_neuron.output for prev_neuron in red[weight[0] - 1]])
                            neuron.calculate_output(w)

            denormalized_output = ((red[-1][0].output + 1) * (max_value - min_value)) / 2 + min_value
            denormalized_hour = (( (pattern[0] + 1) * 23) / 2) + 1
            expected.append(pattern[2])
            test.append(denormalized_output)
            denormalized_hours.append(denormalized_hour)
        plt.figure()
        plt.plot(denormalized_hours, test ,label=f'Prediccion {day[i]}', color='red')
        plt.plot(denormalized_hours, expected, label=f'Esperado {day[i]}')
        plt.xlabel('Horas del dia')
        plt.ylabel('consumo de energia')
        plt.title(f'Consumo de energia por hora del dia {day[i]}')
        plt.legend()
        
        # Create directory if it doesn't exist
        directory = f'tests/{raw_dimensions}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the plot
        plt.savefig(f'{directory}/{day[i]}.png')
        plt.show()

if __name__ == '__main__':
    main()