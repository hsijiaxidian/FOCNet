function modelSize = get_model_size(net)
    paramSizes = cellfun(@size, {net.params.value}, 'UniformOutput', false) ;
    modelSize = 0;
    for i = 1:numel(paramSizes)
        modelSize = modelSize + prod(paramSizes{i}) * 4;
    end
end

