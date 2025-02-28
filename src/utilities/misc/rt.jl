# This function is useful for visualizing matrices with Makie.
function rt(A)
    return reverse(transpose(A), dims = 2)
end