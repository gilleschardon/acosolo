function array = square_array(aperture, N, center, axis)
    
    c = linspace(-aperture/2, aperture/2, N);
    
    if axis == 'x'
        [xg, yg, zg] = meshgrid(c + center(1), c + center(2), center(3));
    elseif axis == 'y'
        [xg, yg, zg] = meshgrid(c + center(1), center(2), c + center(3));
    elseif axis == 'z'
        [xg, yg, zg] = meshgrid(c + center(1), c + center(2), center(3)); 
    end 
array = [xg(:) yg(:) zg(:)];

end