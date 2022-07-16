function [ c ] = fixedcolorbar(axeshandle, varargin)
%fixedcolorbar creates a colorbar but doesnt move the plot

originalSize = get(axeshandle, 'Position'); 
c = colorbar; 
set(axeshandle, 'Position', originalSize); 


for i=1:2:numel(varargin)
    try
        set(c, varargin{i}, varargin{i+1});
    catch
        error([varargin{i} ' is not a valid property for fixedcolorbar function.']);
    end
    
    
end

end

