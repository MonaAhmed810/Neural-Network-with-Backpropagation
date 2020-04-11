from math import exp
from random import seed
from random import random

seed(1)

def rand():
    return random() * 20 - 10

def initialize_network(n_inputs, n_hidden, n_outputs):
   network = list()
   hidden_layer = [{'weights':[rand() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
   network.append(hidden_layer)
   output_layer = [{'weights':[rand() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
   network.append(output_layer)
   return network

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

def d_sigmoid(x):
    return x * (1.0 - x)

def back_propagate(network, expected):
    output_layer = network[1]
    for j in range(len(output_layer)):
      neuron = output_layer[j]
      neuron['error'] = (expected[j] - neuron['output']) * d_sigmoid(neuron['output'])     
    hidden_layer = network[0]
    for j in range(len(hidden_layer)):
      error = 0.0
      for neuron in output_layer:
         error += (neuron['weights'][j] * neuron['error'])
      neuron = hidden_layer[j]
      neuron['error'] = error * d_sigmoid(neuron['output'])

def update_weights(network, x_row, l_rate):
    hidden_layer = network[0]
    inputs = x_row
    for neuron in hidden_layer:
      for j in range(len(inputs)):
         neuron['weights'][j] += l_rate * neuron['error'] * inputs[j]
      neuron['weights'][-1] += l_rate * neuron['error']
    output_layer = network[1]    
    inputs = [neuron['output'] for neuron in hidden_layer]
    for neuron in output_layer:
      for j in range(len(inputs)):
         neuron['weights'][j] += l_rate * neuron['error'] * inputs[j]
      neuron['weights'][-1] += l_rate * neuron['error']

def train(network, x_train, y_train, l_rate, n_epoch, acceptable_MSE):
   for epoch in range(1, n_epoch + 1):
      sum_error = 0.0
      for i in range(len(x_train)):
          x_row = x_train[i]
          outputs = feed_forward(network, x_row)
          expected = y_train[i]
          sum_error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))]) / len(expected)
          back_propagate(network, expected)
          update_weights(network, x_row, l_rate)
      MSE = sum_error / len(x_train)
      print('>epoch=%d, error=%.3f' % (epoch, MSE))
      if MSE <= acceptable_MSE or epoch == n_epoch:
         print('>>Best MSE=%.3f' % (MSE))   
         break

def normalize(data):
   n_rows = len(data)
   n_cols = len(data[0])
   data_normalized = data
   mn_list = []
   mx_list = [] 
   for j in range(n_cols):
      mn = 1000
      mx = 0
      for i in range(n_rows):
         mn = min(mn, data[i][j])
         mx = max(mx, data[i][j])
      mn_list.append(mn)
      mx_list.append(mx) 				 
      range_size = mx - mn + 1
      for i in range(n_rows):
         data_normalized[i][j] = (data[i][j] - mn) / range_size
   return data_normalized, mn_list, mx_list      

def save(network, mn_input, mx_input, mn_output, mx_output):
   n_hidden = len(network[0])	
   n_inputs = len(network[0][0]['weights']) - 1
   n_outputs = len(network[1])

   file = open("weights.txt", "w")
   file.write(str(n_inputs) + " " + str(n_hidden) + " " + str(n_outputs) + "\n")	
   hidden_layer = network[0]
   for i in range(len(hidden_layer)):
      neuron = hidden_layer[i]
      for weight in neuron["weights"]:
         file.write(str(weight) + " ")
      file.write("\n")   
   file.write("-----------------------------------------------------------------------\n")
   output_layer = network[1]
   for i in range(len(output_layer)):
      neuron = output_layer[i]
      for weight in neuron["weights"]:
         file.write(str(weight) + " ")
      file.write("\n")  
   file.write("-----------------------------------------------------------------------\n")
   for mn in mn_input:
	   file.write(str(mn) + " ")
   file.write("\n")	   
   for mx in mx_input:
	   file.write(str(mx) + " ")
   file.write("\n")
   for mn in mn_output:
	   file.write(str(mn) + " ")
   file.write("\n")
   for mx in mx_output:
	   file.write(str(mx) + " ")
   file.write("\n")

   file.close()   

file =  open("train.txt", "r")

line = file.readline().split()
n_inputs = int(line[0])
n_hidden = int(line[1])
n_outputs = int(line[2])

line = file.readline()
n_examples = int(line)

x_train = list()
y_train = list()
for i in range(n_examples):
    line = file.readline().split()
    line = [float(x) for x in line]
    x_train.append(line[:n_inputs])
    y_train.append(line[n_inputs:])

file.close()    

x_train_normalized, mn_input, mx_input = normalize(x_train)    
y_train_normalized, mn_output, mx_output = normalize(y_train)    

l_rate = 0.01
n_epoch = 500
acceptable_MSE = 0.002
network = initialize_network(n_inputs, n_hidden, n_outputs)
train(network, x_train_normalized, y_train_normalized, l_rate, n_epoch, acceptable_MSE)
save(network, mn_input, mx_input, mn_output, mx_output)

print("Done")