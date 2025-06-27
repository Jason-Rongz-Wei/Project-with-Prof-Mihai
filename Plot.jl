using JuMP, Ipopt, Plots
using DelimitedFiles
using Random

gr()

U_opt = readdlm("U_optimized.csv", ',')


function plot_U(U_opt; mode="surface", levels=20)
    x = range(0, stop=1, length=size(U_opt, 2))
    t = range(0, stop=1, length=size(U_opt, 1))

    if mode == "surface"
        return surface(
            x, t, U_opt;
            c = :grays,
            linewidth = 0.2,
            legend = false,
            xlabel = "x", ylabel = "t", zlabel = "U",
            fillalpha = 0.8,
            title = "U(t,x) with contours",
            contour = true,
            levels = levels,
            size = (1200, 1200)
        )
    elseif mode == "tslice"
        return plot(
            x, U_opt';
            color = :black,
            legend = false,
            xlabel = "x", ylabel = "U",
            lw = 0.5,
            alpha = 0.8,
            title = "t-direction side view with full contours",
            size = (1200, 1200)
        )
    elseif mode == "xslice"
        return plot(
            t, U_opt;
            color = :black,
            legend = false,
            xlabel = "t", ylabel = "U",
            lw = 0.5,
            alpha = 0.8,
            title = "x-direction side view with full contours",
            size = (1200, 1200)
        )
    else
        error("Invalid mode. Use \"surface\", \"tslice\", or \"xslice\".")
    end
end

plot_U(U_opt; mode="surface", levels = 20)
#haha
#有没有好的convergence test
#在julia里改一下gradient
#preconditioner除了identity matrix以外怎么改
