function s = pmem(x)
% copy from matconvnet
    if isnan(x),       s = 'NaN' ;
        elseif x < 1024^1, s = sprintf('%.0fB', x) ;
        elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
        elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
        else               s = sprintf('%.0fGB', x / 1024^3) ;
    end
end