using Plots

xs = range(-10, 10, length=101)

sigmoid(x) = 1 / (1 + exp(-x))
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
relu(x) = max(0, x)

plot(xs, sigmoid.(xs), framestyle=:origin, legend=false, lw=2)
savefig("./intro/images/14_sigmoid.png")
plot(xs, tanh.(xs), framestyle=:origin, legend=false, lw=2)
savefig("./intro/images/14_tanh.png")
plot(xs, relu.(xs), framestyle=:origin, legend=false, lw=2)
savefig("./intro/images/14_relu.png")
