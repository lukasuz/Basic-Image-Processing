function [state, state_err] = simulated_annealing(img_path, gt_path, amount_kernels)
    I = imread(img_path);
    gt = imread(gt_path);
    
    img = (double(I) / 255) - 0.5;
    temp = 500;
    alpha = 0.99; % temp decay;
    beta = 2; % step size
    state = gen_rnd_state(amount_kernels);
    init_random_step = 0.10;
    
    init_temp = temp;
    init_beta = beta;
    
    state_seg = gen_segmentation_from_state(state, img);
    state_err = calc_err(gt, state_seg);
    while temp > 1
        temp = temp * alpha;
        beta = init_beta - 0.9 * (init_beta -(init_beta * temp/init_temp));
        
        neighbour = gen_neighbour(state, beta, amount_kernels);
        neighbour_seg = gen_segmentation_from_state(neighbour, img);
        neighbour_err = calc_err(gt, neighbour_seg);
        delta_energy = neighbour_err - state_err;
        
        random_acceptance_chance = init_random_step * temp/init_temp;
        disp(strcat("Temp:", num2str(temp), ...
            "  Energy:", num2str(delta_energy), ...
            "  state_err:", num2str(state_err), ...
            "  beta:", num2str(beta), ...
            "  RAC:", num2str(random_acceptance_chance)));
        
        % this is not working
        % worse_acceptance_prob = exp((-delta_energy * init_temp)/temp);
        % rand_num = rand();
        if delta_energy < 0 || random_acceptance_chance >= rand()
            state = neighbour;
            state_err = neighbour_err;
        end
    end
    state
end

function segmentation = gen_segmentation_from_state(state, img)
    segmentation = zeros(size(img,1), size(img,2), 1);
    for i=1:size(state,2)
        segmentation = segmentation + gen_segmentation_from_kernel(state(:,i), img);
    end
    segmentation = min(segmentation, 1);
end

function segmentation = gen_segmentation_from_kernel(kernel, img)
    % Depth-wise convolution.
    % I should have used matrix multiplication, but I was tired.
    segmentation = zeros(size(img,1), size(img,2), 1);
    for i=1:size(img,1)
        for j=1:size(img,2)
            segmentation(i,j) = img(i,j,1) * kernel(1) + ...
                img(i,j,2) * kernel(2) + img(i,j,3) * kernel(3);
        end
    end
    segmentation = sigmoid(segmentation);
    segmentation = round(segmentation);
end

function output = sigmoid(input)
    % Simple mapping of function on matrix,
    % technically not allowed, but I could have used
    % two for loops as well. I hope it is okay.
    output = arrayfun(@(x) 1 / (1 + exp(1)^(-x)), input);
end

function err = calc_err(gt, segmentation)
    err = sum(sum(abs(segmentation - gt))) / numel(gt);
end

function neighbour = gen_neighbour(state, beta, amount_kernels)
    step_vec = gen_rnd_state(amount_kernels); % generate step vec
    step_vec = step_vec * beta; % "step size"
    neighbour = state - step_vec; % create neighbour
    % neighbour = min(max(neighbour,0),1);
end

function state = gen_rnd_state(amount_kernels)
    state = rand(3, amount_kernels);
    state = state -0.5;
end