from math import exp

def summation(weights, inputs):
   sum = weights[-1]
   for i in range(len(weights)-1):
      sum += weights[i] * inputs[i]
   return sum  

def sigmoid(x):
   #return 1.0 / (1.0 + exp(-x))   
   if x < 0:
      return 1.0 - 1.0 / (1.0 + exp(x))  
   return 1.0 / (1.0 + exp(-x))    

def feed_forward(network, x_row):
   inputs = x_row
   for layer in network:
      new_inputs = []
      for neuron in layer:
         sum = summation(neuron['weights'], inputs)
         neuron['output'] = sigmoid(sum)
         new_inputs.append(neuron['output'])
      inputs = new_inputs
   return inputs

def normalize(data, mn, mx):
   n_rows = len(data)
   n_cols = len(data[0])
   data_normalized = data
   for j in range(n_cols):
      range_size = mx[j] - mn[j] + 1
      for i in range(n_rows):
         data_normalized[i][j] = (data[i][j] - mn[j]) / range_size
   return data_normalized    

def load():
    file = open("weights.txt", 'r')

    line = file.readline().split()
    n_inputs = int(line[0])
    n_hidden = int(line[1])
    n_outputs = int(line[2]) 

    network = list()

    hidden_layer = [{'weights':[]} for i in range(n_hidden)]
    for i in range(n_hidden):
        line = file.readline().split()
        line = [float(x) for x in line]
        hidden_layer[i]['weights'] = line
    network.append(hidden_layer)

    line = file.readline()

    output_layer = [{'weights':[]} for i in range(n_outputs)]
    for i in range(n_outputs):
        line = file.readline().split()
        line = [float(x) for x in line]
        output_layer[i]['weights'] = line
    network.append(output_layer)

    line = file.readline()

    line = file.readline().split()
    mn_input = [float(x) for x in line]
    
    line = file.readline().split()
    mx_input = [float(x) for x in line]

    line = file.readline().split()
    mn_output = [float(x) for x in line]

    line = file.readline().split()
    mx_output = [float(x) for x in line]

    file.close()

    return network, mn_input, mx_input, mn_output, mx_output

def predict(network, x_test, y_test):
    result = []
    sum_error = 0.0
    for i in range(len(x_test)):
        x_row = x_test[i]
        outputs = feed_forward(network, x_row)
        result.append(outputs)
        expected = y_test[i]
        sum_error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))]) / len(expected)
    return result, sum_error / len(x_test)
        


file =  open("test.txt", "r")

line = file.readline().split()
n_inputs = int(line[0])
n_outputs = int(line[1])

line = file.readline()
n_examples = int(line)

x_test = list()
y_test = list()
for i in range(n_examples):
    line = file.readline().split()
    line = [float(x) for x in line]
    x_test.append(line[:n_inputs])
    y_test.append(line[n_inputs:])

file.close()     

network, mn_input, mx_input, mn_output, mx_output = load()
x_test_normalized = normalize(x_test, mn_input, mx_input)   
y_test_normalized = normalize(y_test, mn_output, mx_output)

y_predicted_normalized, MSE = predict(network, x_test_normalized, y_test_normalized) 

print(MSE)