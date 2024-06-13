
using MLDatasets: MNIST
using Statistics
using ProgressMeter, BenchmarkTools
using LinearAlgebra
using Random
include("../src/graphutils.jl")
include("../src/autodifflib.jl")

# Ładowanie danych MNIST
train_data = MNIST(:train)
test_data  = MNIST(:test)
learning_rate = 0.015
epochs = 5

conf = (;
    learning_rate = 0.015,
    epochs = 5,
    batchsize = 100,
    num_of_classes = 10,
    hidden_size = 64,
    rng = Random.default_rng()
)

# Funkcja do ładowania danych
function initRNN(learning_rate::Int, epochs::Int) #returns layers
    num_of_classes = 10; #0-9 cyfry
    input_size = length(vec(train_data[1].features)) #feature_dim
    
    hidden_size = 64 #liczba neuronów ukrytych

    rng = Random.default_rng()

    Wxh = xavier_init(rng, hidden_size, input_size) 
    Whh = xavier_init(rng, hidden_size, hidden_size)
    Why = xavier_init(rng, num_of_classes, hidden_size) 
    bx = xavier_init(rng, hidden_size,1) 
    bh = xavier_init(rng, hidden_size,1) 
    b = xavier_init(rng, num_of_classes,1)

    hiddens = Vector{Vector{Float32}}()
    outputs = Vector{Vector{Float32}}()
    inputs = Vector{Vector{Float32}}()

    Wxh_grad = zeros(Float32, hidden_size,  input_size)
    Whh_grad = zeros(Float32, hidden_size, hidden_size)
    Why_grad = zeros(Float32, num_of_classes, hidden_size)
    bx_grad = zeros(Float32, hidden_size,1) 
    bh_grad = zeros(Float32, hidden_size,1) 
    b_grad = zeros(Float32, num_of_classes,1)

    input_layer = InputLayer(x -> x,() -> 1, nothing, Wxh, Wxh_grad, bx, bx_grad, inputs)

    hidden_layer = HiddenLayer(tanhip, tanhip_derivative, nothing, Whh, Whh_grad, bh, bh_grad, hiddens)

    output_layer = OutputLayer(softmax, x -> x, nothing, Why, Why_grad, b, b_grad, outputs)

    return input_layer, hidden_layer, output_layer
end

function trainRNN(learning_rate::Float64, epochs::Int) 
    clamp = 5.0
    epoch_loss = 0
    batchsize = 100
    model = Model(initRNN(1,1)...,learning_rate,x->x,mse_grad,clamp) #TODO implement loss





    for i in 1:epochs

        data_lenght = length(train_data)
          
        println(string(i)*" epoch")
            
        (batch_x,batch_y) = batch_loader(train_data.features, train_data.targets, batchsize, data_lenght)
        

        for j in 1:(floor(Int, data_lenght/batchsize))

        
        forward_pass(model, batch_x[j], batchsize) 
     
        println(string(i) * " Accuracy: " * string(calculate_accuracy(model.out.outputs,batch_y[j])))
        
        backward_pass(model, batch_y[j], batchsize)
       
        resetRNN(model)
        end
        
    end
    return model
end

function resetRNN(model::Model)
    empty!(model.hid.hiddens)
    empty!(model.out.outputs)
    empty!(model.in.inputs)
    model.hid.weights_grad = zeros(size(model.hid.weights_grad))
    model.out.weights_grad = zeros(size( model.out.weights_grad))
    model.in.weights_grad = zeros(size(model.in.weights_grad))

    model.hid.bias_grad = zeros(size(model.hid.bias_grad))
    model.out.bias_grad = zeros(size(model.out.bias_grad))
    model.in.bias_grad = zeros(size(model.in.bias_grad))
end



function calculate_accuracy(predictions, targets)
    n_samples = length(targets)
    n_correct = 0
    #actuals = one_hot.(targets)
    #loss = mse.(actuals,model.out.outputs)
    #println(predictions,targets)
    #debug(predictions)
    #debug(targets)
    for i in 1:n_samples
        if argmax(predictions[i])[1]-1 == targets[i] #because of 0 
            #println("correct prediction! pred:",argmax(predictions[i]),"y:",predictions[i],"actual:",targets[i])
            n_correct+=1
        end
    end
    return n_correct/n_samples
end

function testRNN(model::Model)

    test_data_length = length(test_data)
    
    resetRNN(model)




   



        
    (test_x,test_y) = batch_loader(train_data.features, train_data.targets, test_data_length, test_data_length)
    
    

    forward_pass(model, test_x[1], test_data_length) 



    res = calculate_accuracy(model.out.outputs,test_y[1])

    println( " Accuracy: ", res)
    println(" Loss: ", mean(cross_entropy_loss.(test_y[1],model.out.outputs)))

    
    return res
end

m = trainRNN(learning_rate,epochs)

println("")


acc_res=[]
for a in 1:100
    println(string(a) * " test")
    push!(acc_res,testRNN(m) )
    println(" ")

end
println(" ")
println("Minimum accuracy score in 100 tests: " * string(minimum(acc_res)))
println("Mean accuracy score in 100 tests: " * string(mean(acc_res)))
println("END    END    END    END    END    END    END    END    END    ")