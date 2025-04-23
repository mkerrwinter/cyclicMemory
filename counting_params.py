widths = [4, 8, 32, 64]
depth = 3
input = 4
output = 1

for width in widths:
    N_params = 0
    for i in range(depth):
        if i == 0:
            N_params += (input * width)
        elif i == depth - 1:
            N_params += (width * output)
        else:
            N_params += (width * width)

    print('Width: {}, Depth: {}, N_params: {}'.format(width, depth, N_params))
