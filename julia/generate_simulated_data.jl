using Distributions;

function make_xy(x, y, x_mu, y_mu, bx_left, bx_right, by_down, by_up, lr_lr, lr_ud, nsteps)
    xy = zeros(nsteps, 2)
    for k in 1:nsteps
        x_sample = rand(Normal(x_mu, 1.))
        x = x + lr_lr * (x_sample - x)
        y_sample = rand(Normal(y_mu, 1.))
        y = y + lr_ud * (y_sample - y)
        if x <= bx_right && x >= bx_left
            xy[k, 1] = x
        else
            xy[k, 1] = xy[k-1, 1]
        end
        if y <= by_up && y >= by_down
            xy[k, 2] = y
        else 
            xy[k, 2] = xy[k-1, 2]
        end
    end
    return xy
end


