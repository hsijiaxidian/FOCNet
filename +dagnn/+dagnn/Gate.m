classdef Gate < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
      function outputs = forward(obj, inputs, params)
          outputs{1} = inputs{1};
          outputs{1} = outputs{1} + inputs{2}*params{1};
      end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = derOutputs{1};
            derInputs{2} = params{1}*derOutputs{1};
            derParams{1} =sum(sum(sum(sum(derOutputs{1}.*inputs{end},3))),4);
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = Gate(varargin)
      obj.load(varargin) ;
    end
  end
end

