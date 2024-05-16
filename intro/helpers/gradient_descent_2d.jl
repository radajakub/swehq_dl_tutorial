using Plots

xs = range(-2, 3, length=100)
ys = range(-2, 3, length=100)

f(x, y) = y * sin(x^2) + x * cos(y^2) + 1
df(x, y) = [2 * x * y * cos(x^2) + cos(y^2), sin(x^2) - 2 * x * y * sin(y^2)]

plot(xs, ys, f, st=[:surface], cbar=false, c=:viridis, xlabel="x", ylabel="y", zlabel="f(x, y)", dpi=300)
savefig("./intro/images/06_gd_function.png")

p1 = contour(xs, ys, f, fill=true, cbar=false, c=:viridis, xlabel="x", ylabel="y", dpi=300)
savefig("./intro/images/06_gd_contour.png")

function gd(x0, lr)
    p = contour(xs, ys, f, fill=true, cbar=false, c=:viridis, xlabel="x", ylabel="y", dpi=300)
    scatter!(p, [x0[1]], [x0[2]], color=:red, legend=false)
    plot!(1, legend=false, arrow=true, xlim=(-2, 3), ylim=(-2, 3), color=:red)

    anim = Animation()

    for _ in 1:100
        push!(p, 3, x0[1], x0[2])
        frame(anim)
        grad = df(x0[1], x0[2])
        if all(abs.(grad) .<= 1e-5)
            println("break")
            break
        end
        x0 -= lr * grad
    end
    gif(anim, "./intro/images/06_gd_2_badlr.gif", fps=10)
end

gd([0.5, 1], 0.1)
gd([2, 0.5], 0.01)
gd([2, 0.5], 0.1)
