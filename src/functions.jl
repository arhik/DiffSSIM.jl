using Images
using CUDA
using Flux
using Statistics

function kernelWindow(windowSize, σ)
    kernel = zeros(windowSize, windowSize)
    dist = C -> sqrt(sum((ceil.(windowSize./2) .- C.I).^2))
    for idx in CartesianIndices((1:windowSize, 1:windowSize))
        kernel[idx] = exp(-(dist(idx)))/sqrt(2*σ^2)
    end
    return (kernel/sum(kernel))
end

using TestImages

using ImageQualityIndexes

img = testimage("coffee") .|> float32

x = reshape(permutedims(channelview(img) .|> Float32, (2, 3, 1)), size(img)..., 3, 1) |> gpu
y = rand(Float32, size(img)..., 3, 1) |> gpu

windowSize = 11
nChannels = size(x, 3)
kernel = repeat(
    reshape(kernelWindow(windowSize, 1.5) .|> Float32, (windowSize, windowSize, 1, 1)),
        inner=(1, 1, 1, nChannels)
    ) |> gpu
cdims = DenseConvDims(
    size(x), 
    size(kernel), 
    stride=(1, 1), 
    padding=div(windowSize, 2), 
    dilation=(1, 1), 
    flipkernel=false, 
    groups=nChannels           
)

C1 = 0.01f0^2
C2 = 0.03f0^2

function ssimScore(x, y)
    μx = conv(x, kernel, cdims)
    μy = conv(y, kernel, cdims)

    μx2 = μx.^2
    μy2 = μy.^2
    μxy = μx.*μy

    σ2x = conv(x.^2, kernel, cdims) .- μx2
    σ2y = conv(y.^2, kernel, cdims) .- μy2
    σxy = conv(x.*y, kernel, cdims) .- μxy

    lp = (2.0f0.*μxy .+ C1)./(μx2 .+ μy2 .+ C1)
    cp = (2.0f0.*σxy .+ C2)./(σ2x .+ σ2y .+ C2)

    ssimMap = lp.*cp
    return mean(ssimMap)
end

score = ssimScore(x, y)

ssimLoss(x, y) = -ssimScore(x, y)

using ImageView

gui = imshow_gui((512, 512))
canvas = gui["canvas"]

while score < 0.99999
    score = ssimScore(x, y)
    @info score
    grads = gradient(ssimLoss, x, y)
    y .-= 100.0f0*grads[2] #lr is strange ... need to check grads
    yimg = colorview(RGB{N0f8},
        permutedims(
            reshape(clamp.(y |> cpu, 0.0, 1.0), size(img)..., nChannels),
            (3, 1, 2),
        ) .|> n0f8
    )
    imshow!(canvas, yimg)
end
