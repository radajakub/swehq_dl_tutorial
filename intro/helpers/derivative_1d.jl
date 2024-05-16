using Plots

function plot_derivative1d()
    # plot a function and its derivative in a point
    x = range(-2, 3, length=1000)
    f(x) = x * sin(x^2) + 1
    plot(x, f.(x), label="f(x)", dpi=300)

    x0 = -0.5
    y0 = f(x0)
    scatter!([x0], [y0], label="x0")

    df(x) = sin(x^2) + 2x^2 * cos(x^2)
    tangent(x) = df(x0) * (x - x0) + y0

    x = range(-1, 0, length=100)
    plot!(x, tangent.(x), label="tangent at x0", color=:red, arrow=:closed)
end


plot_derivative1d()
savefig("./intro/images/derivative_1d.png")
