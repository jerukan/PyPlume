function [cmap]=buildcmap(colors, N)
% [cmap]=buildcmap(colors)
%
% This function can be used to build your own custom colormaps. Imagine if
% you want to display rainfall distribution map. You want a colormap which
% ideally brings rainfall in mind, which is not achiveved by colormaps such
% as winter, cool or jet and such. A gradient of white to blue will do the
% task, but you might also use a more complex gradient (such as
% white+blue+red or colors='wbr'). This function can be use to build any
% colormap using main colors rgbcmyk. In image processing, w (white) can be
% used as the first color so that in the output, the background (usually
% with 0 values) appears white. In the example of rainfall map, 'wb' will
% produce a rainfall density map where the background (if its DN values are
% 0) will appear as white.
%
% Inputs:
%  colors: string (char) of color codes, any sequence of rgbcmywk
%  representing different colors (such as 'b' for blue) is acceptable. If a
%  gradient of white to blue is needed, colors would be 'wb'; a rainbow of
%  white+blue+red+green would be 'wbrg'.
%
% Example:
%  [cmap]=buildcmap('wygbr');
% %try the output cmap:
% im=imread('cameraman.tif');
% imshow(im), colorbar
% colormap(cmap) %will use the output colormap
%
% First version: 14 Feb. 2013
% sohrabinia.m@gmail.com
%--------------------------------------------------------------------------

if nargin<1
    colors='wrgbcmyk';
%     return
end

if ~exist('N')
    N = 300;
end

if ~ischar(colors)
    
    
    if isnumeric(colors)
        dms = size(colors);
        if sum(dms==3)
            disp('Using numeric list of colors');
            di = find(dms==3);
            if di==1
                colors = colors';
            end
        end
    else
        
        error(['Error! colors must be a variable of type char with '...
        'color-names, such as ''r'', ''g'', etc., '...
        '[OR NOW can be a numeric list] '...
        'type ''help buildcmap'' for more info']);
    
    end
end

ncolors=length(colors)-1;
% ncolors=length(colors);



% bins=round(N/ncolors);
bins=floor(N/ncolors);

% diff1=255-bins*ncolors;

vec=zeros(N,3);
if ischar(colors)
    switch colors(1)
        case 'w'
            vec(1,:)=1;
        case 'r'
            vec(1,:)=[1 0 0];
        case 'g'
            vec(1,:)=[0 1 0];
        case 'b'
            vec(1,:)=[0 0 1];
        case 'c'
            vec(1,:)=[0 1 1];
        case 'm'
            vec(1,:)=[1 0 1];
        case 'y'
            vec(1,:)=[1 1 0];
        case 'k'
            vec(1,:)=[0 0 0];
        case 't'
            vec(1,:)=1;
    end
else
    vec = colors;
end
%{
for i=1:ncolors
 beG=(i-1)*bins+1;
 enD=i*bins+1; %beG,enD
%     beG=(i-1)*bins+1;
%     enD=i*bins;
 switch colors(i+1)
     case 'w'
         vec(beG:enD,1)=linspace(vec(beG,1),1,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),1,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),1,bins+1)';%colors(i+1),beG,enD,
     case 'r'
         vec(beG:enD,1)=linspace(vec(beG,1),1,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),0,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),0,bins+1)';%colors(i+1),beG,enD
     case 'g'
         vec(beG:enD,1)=linspace(vec(beG,1),0,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),1,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),0,bins+1)';%colors(i+1),beG,enD
     case 'b'         
         vec(beG:enD,1)=linspace(vec(beG,1),0,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),0,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),1,bins+1)';%colors(i+1),beG,enD
     case 'c'
         vec(beG:enD,1)=linspace(vec(beG,1),0,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),1,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),1,bins+1)';%colors(i+1),beG,enD
     case 'm'
         vec(beG:enD,1)=linspace(vec(beG,1),1,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),0,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),1,bins+1)';
     case 'y'
         vec(beG:enD,1)=linspace(vec(beG,1),1,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),1,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),0,bins+1)';
     case 'k'
         vec(beG:enD,1)=linspace(vec(beG,1),0,bins+1)';
         vec(beG:enD,2)=linspace(vec(beG,2),0,bins+1)';
         vec(beG:enD,3)=linspace(vec(beG,3),0,bins+1)';
 end
end
%}

for i=1:ncolors
%  beG=(i-1)*bins+1;
%  enD=i*bins+1; %beG,enD
    if i==1
        beG=(i-1)*bins+1;
    else
        beG=(i-1)*bins;
    end
    
    enD=i*bins;
    thesebins = enD-beG+1;
    
%  if vec(beG,:) == [0 0 0]
%      disp('mo')
%  end

%  switch colors(i+1)
%      case 'w'
%          vec(beG:enD,1)=linspace(vec(beG,1),1,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),1,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),1,thesebins)';%colors(i+1),beG,enD,
%      case 'r'
%          vec(beG:enD,1)=linspace(vec(beG,1),1,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),0,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),0,thesebins)';%colors(i+1),beG,enD
%      case 'g'
%          vec(beG:enD,1)=linspace(vec(beG,1),0,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),1,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),0,thesebins)';%colors(i+1),beG,enD
%      case 'b'         
%          vec(beG:enD,1)=linspace(vec(beG,1),0,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),0,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),1,thesebins)';%colors(i+1),beG,enD
%      case 'c'
%          vec(beG:enD,1)=linspace(vec(beG,1),0,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),1,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),1,thesebins)';%colors(i+1),beG,enD
%      case 'm'
%          vec(beG:enD,1)=linspace(vec(beG,1),1,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),0,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),1,thesebins)';
%      case 'y'
%          vec(beG:enD,1)=linspace(vec(beG,1),1,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),1,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),0,thesebins)';
%      case 'k'
%          vec(beG:enD,1)=linspace(vec(beG,1),0,thesebins)';
%          vec(beG:enD,2)=linspace(vec(beG,2),0,thesebins)';
%          vec(beG:enD,3)=linspace(vec(beG,3),0,thesebins)';
%  end

    if ischar(colors)
         ncol = ones(1,3);
         switch colors(i+1)
            case 'w'
                ncol(1,:)=1;
            case 'r'
                ncol(1,:)=[1 0 0];
            case 'g'
                ncol(1,:)=[0 1 0];
            case 'b'
                ncol(1,:)=[0 0 1];
            case 'c'
                ncol(1,:)=[0 1 1];
            case 'm'
                ncol(1,:)=[1 0 1];
            case 'y'
                ncol(1,:)=[1 1 0];
            case 'k'
                ncol(1,:)=[0 0 0];
            case 't'
                ncol(1,:)=1;
        end
    else
        ncol = colors(i+1,:);
    end

    vec(beG:enD,1)=linspace(vec(beG,1),ncol(1),thesebins)';
    vec(beG:enD,2)=linspace(vec(beG,2),ncol(2),thesebins)';
    vec(beG:enD,3)=linspace(vec(beG,3),ncol(3),thesebins)';

 
end

if strcmp(colors(1),'t')
    decayingalpha = ones(N,1);
    decayrate = [0:N-1]./N;
    alpha = decayingalpha.*decayrate;
    
    vec = [vec,alpha];
    
end



cmap=vec(1:bins*ncolors,:);
end %end of buildcmap