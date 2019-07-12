classdef Wsum < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
      function outputs = forward(obj, inputs, params)
          obj.numInputs = numel(inputs) ;
          outputs{1} = inputs{1} * params{1}(1);
          for k = 2:obj.numInputs
              outputs{1} = outputs{1} + params{1}(k)*inputs{k} ;
          end
      end

      function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          for k = 1:obj.numInputs
              derInputs{k} = params{1}(k)*derOutputs{1} ;
              der(k) =sum(sum(sum(sum(derOutputs{1}.*inputs{k},3))),4);
          end
          derParams{1} = der;
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

    function obj = WSum(varargin)
      obj.load(varargin) ;
    end
  end
end

