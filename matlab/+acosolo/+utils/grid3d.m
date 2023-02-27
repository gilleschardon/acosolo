function grid, dims = grid3D(lb, ub, step):
    xx = lb(1):step:ub(1);
    yy = lb(2):step:ub(2);
    zz = lb(3):step:ub(3);

    dims = [length(xx), length(yy), length(zz)];

    Xg, Yg, Zg = meshgrid(xx, yy, zz);
    
    grid = [Xg(:) Yg(:) Zg(:)];
end
