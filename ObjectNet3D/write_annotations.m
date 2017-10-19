function write_annotations

opt = globals();

ids = textread(fullfile(opt.path_objectnet3d, 'Image_sets/trainval.txt'), '%s');
N = numel(ids);

for i = 1:N
    % load annotation
    filename = sprintf('%s/Annotations/%s.mat', opt.path_objectnet3d, ids{i});
    object = load(filename);
    record = object.record;
    objects = record.objects;
    n = numel(objects);
    
    % write to file
    fid = fopen(sprintf('Labels/%s.txt', ids{i}), 'w');
    
    for j = 1:n
        cls = objects(j).class;
        bbox = objects(j).bbox;
        if isempty(objects(j).viewpoint) == 0
            viewpoint = objects(j).viewpoint;
            if isfield(viewpoint, 'azimuth') == 0 || isempty(viewpoint.azimuth) == 1
                a = viewpoint.azimuth_coarse;
            else
                a = viewpoint.azimuth;
            end
            if isfield(viewpoint, 'elevation') == 0 || isempty(viewpoint.elevation) == 1
                e = viewpoint.elevation_coarse;
            else
                e = viewpoint.elevation;
            end            
            theta = viewpoint.theta;
            
            a = a * pi / 180;
            e = e * pi / 180;
            theta = theta * pi / 180;

            fprintf(fid, '%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n', cls, bbox(1), ...
                bbox(2), bbox(3), bbox(4), a, e, theta);
        else
            fprintf('no viewpoint\n');
            fprintf(fid, '%s %.2f %.2f %.2f %.2f\n', cls, bbox(1), ...
                bbox(2), bbox(3), bbox(4));
        end
    end
    
    fclose(fid);
end