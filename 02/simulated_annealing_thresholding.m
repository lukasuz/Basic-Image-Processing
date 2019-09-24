function state = simulated_annealing_thresholding(img_path, gt_path, intervals)
    I = imread(img_path);
    gt = imread(gt_path);
    
	img = rgb2hsv(I);
    %img = (double(I) / 255);
    
    temp = 10000;
    alpha = 0.99; % temp decay;
    beta = 1; % step size
    
    init_temp = temp;
    init_beta = beta;
    
    state = gen_rnd_state(intervals);
    state_seg = gen_segmentation_from_state(state, img);
    state_err = calc_err(gt, state_seg);
    while temp > 1
        temp = temp * alpha;
        beta = init_beta - 0.9 * (init_beta -(init_beta * temp/init_temp));
        
        neighbour = gen_neighbour(state, beta);     
        neighbour_seg = gen_segmentation_from_state(neighbour, img);
        neighbour_err = calc_err(gt, neighbour_seg);
        
        delta_energy = neighbour_err - state_err;
        
        disp(strcat("Temp:", num2str(temp), ...
            "  Energy:", num2str(delta_energy), ...
            "  state_err:", num2str(state_err), ...
            "  beta:", num2str(beta)));
        % "fake" simulated annelealing, does not incorporate energy.
        % Resembles more a greedy descent with some random factor.
        if delta_energy < 0 || 0.2 * temp/init_temp > rand()
            state = neighbour;
            state_err = neighbour_err;
            
        end
    end
   state
end

function segmentation = gen_segmentation_from_state(state, img)
    segmentation = zeros(size(img,1), size(img,2), 1);
    for i=1:size(state,1)/6
        k = i * 6 - 6;
        new_segmentation = ...
            (img(:,:,1) >= state(k+1) & img(:,:,1) <= state(k+2)) & ...
            (img(:,:,2) >= state(k+3) & img(:,:,2) <= state(k+4)) & ...
            (img(:,:,3) >= state(k+5) & img(:,:,3) <= state(k+6));
        segmentation = segmentation + new_segmentation;
    end
end

function err = calc_err(gt, segmentation)
    err = sum(sum(abs(segmentation - gt))) / numel(gt);
end

function neighbour = gen_neighbour(state, beta)
    only_one = 0;
    if ~only_one % modify whole state
        step_vec = gen_rnd_state(size(state, 1)/6); % generate step vec
        step_vec = step_vec - 0.5; % center
        step_vec = step_vec * beta; % "step size"
        neighbour = state + step_vec; % create neighbour
        neighbour = min(max(neighbour,0),1);
    else % modify only one axis
        step_vec = zeros(size(state));
        step = (rand() - 0.5) * beta;
        rnd_indx = ceil(rand() * size(state, 1));
        step_vec(rnd_indx) = step;
        neighbour = state - step_vec;
        neighbour = min(max(neighbour,0),1);
    end
end

function state = gen_rnd_state(intervals)
    state = rand(6*intervals, 1);
end