classdef PLM < dagnn.ElementWise
  %PLM DagNN sum layer
  %   The Power Low Memory layer takes the weighted sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
    Memory_w = single([0.2000,0.0800,0.0480,0.0336,0.0255,0.0204,0.0169,0.0144,0.0125,0.0110]);
    nmliz = 1;
  end

  methods
      function outputs = forward(obj, inputs, params)
          obj.numInputs = numel(inputs) ;
          obj.Memory_w = obj.Memory_w(obj.numInputs-1:-1:1);
          obj.nmliz = sum(obj.Memory_w);
          outputs{1} = inputs{end};
          for k = 1:obj.numInputs-1
              outputs{1} = outputs{1} + obj.Memory_w(k)*inputs{k}/obj.nmliz;
          end
      end

      function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          for k = 1:obj.numInputs-1
              derInputs{k} = obj.Memory_w(k)*derOutputs{1}/obj.nmliz;
          end
          derInputs{obj.numInputs} = derOutputs{1};
          derParams = {};
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

    function obj = PLM(varargin)
      obj.load(varargin) ;
    end
  end
end

