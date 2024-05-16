using Plots

xs = range(-10, 10, length=100)
ys = range(-5, 5, length=100)

f(x, y) = -x^2 - 2y^2
df(x, y) = [-2x, -2y]


plot(xs, ys, f, st=[:surface], cbar=false, c=:viridis, xlabel="x", ylabel="y", zlabel="f(x, y)", dpi=300)
savefig("./intro/images/05_function_2d.png")

x0 = [-7.5, -3]
direction = df(x0[1], x0[2])
x1 = x0 + 0.2 * direction

contour(xs, ys, f, fill=true, c=:viridis, xlabel="x", ylabel="y", dpi=300)
plot!([x0[1], x1[1]], [x0[2], x0[2]], arrow=true, linewidth=1, color=:red, legend=false)
plot!([x1[1], x1[1]], [x0[2], x1[2]], linewidth=1, color=:red, linestyle=:dot, legend=false)
plot!([x0[1], x0[1]], [x0[2], x1[2]], arrow=true, linewidth=1, color=:red, legend=false)
plot!([x0[1], x1[1]], [x1[2], x1[2]], linewidth=1, color=:red, linestyle=:dot, legend=false)
plot!([x0[1], x1[1]], [x0[2], x1[2]], arrow=true, linewidth=2, color=:red, legend=false)
scatter!([x0[1]], [x0[2]], color=:red, legend=false)
savefig("./intro/images/05_contour_2d.png")
