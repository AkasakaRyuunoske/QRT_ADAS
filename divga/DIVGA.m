% Program configuration
programs = {'qrt'};
program_config.qrt = struct('N', 497, 'M', 350, 'max_iterations', 334);

%for each program
%costs is an array of size N
%coverage is an array of size N
%revealed_failures is an array of size N (of binary values)
function total_fitness = computeFitness(population, costs, coverage, revealed_failures)
    [m, n] = size(population);
    fitness_values = zeros(m, 1);

    for i = 1:m
        solution = population(i, :);
        total_cost = sum(solution .* costs)/sum(costs);

        total_coverage = -sum(solution .* coverage)/sum(coverage);

        total_failures = -sum(solution .* revealed_failures) / sum(revealed_failures ~= 0);
        fitness_values(i) = total_cost + total_coverage + total_failures;
    end

    total_fitness = sum(fitness_values);
end


function f = fitnessFunction(x, costs, coverage, revealed_failures)
    total_cost = sum(x .* costs)/sum(costs);

    total_coverage = -sum(x .* coverage)/sum(coverage);

    nonzero_failures = revealed_failures(revealed_failures ~= 0);
    total_failures = -sum(x .* revealed_failures)/length(nonzero_failures);

    f = [total_cost, total_coverage, total_failures];
end


for prog = programs
    program = prog{1};
    cfg = program_config.(program);
    N = cfg.N;
    M = cfg.M;
    max_iterations = cfg.max_iterations;

    fprintf('\n=== Running for %s ===\n', program);
    
    % Example data for test cases
    
    %get costs
    costs = str2double(strsplit(fileread([program '_costs.txt']), ','));
    
    coverage = str2double(strsplit(fileread([program '_div_scores.txt']), ','));
    
    %get revealed_failures
    revealed_failures = str2double(strsplit(fileread([program '_collisions.txt']), ','));
    
    % Define the fitness function handle
    fitnessFcn = @(x) fitnessFunction(x, costs, coverage, revealed_failures);
    
    % Array to store Pareto fronts
    pareto_fronts = cell(1, 10);
    
    % Array to store execution times
    execution_times = zeros(1, 10);
    
    % Step 1: Find a valid Hadamard matrix size
    H_size = 4;
    while H_size < max(N, M) + 1
        H_size = H_size * 2;
    end
    
    % Generate an Hadamard matrix
    H = hadamard(H_size);
    
    % Step 2: Sort the rows of the matrix H in ascending order
    H_sorted = sortrows(H);
    
    % Step 3: Delete the first row of the matrix H
    H_sorted(1, :) = [];
    
    % Step 4: Select the first M rows and N columns of the matrix H
    L = H_sorted(1:M, 1:N);
    
    for rep = 1:10
        fprintf('Repetition %d\n', rep);
        % Step 5: Convert L into a binary matrix Lm
        % Convert 1 to 1 and -1 to 0
        starting_population = (L + 1) / 2;
        
        last_generations = {};
        last_mean_fvalue = 0;
        
        average_change = 0;
        first_loop = 1;
        subsequent_diff = 0;
    
        routine_execution_times = [];
        
        for i = 1:max_iterations
            fprintf('Repetition iteration %d\n', rep);
            fprintf('Optimization iteration %d\n', i);
            fprintf('Subsequent diffs %d\n', subsequent_diff);
            % Set up options for gamultiobj
            options = optimoptions('gamultiobj', ...
                'InitialPopulationMatrix', starting_population, ...
                'PopulationSize', M, ...
                'MaxGenerations', 2, ...
                'CrossoverFcn', {@crossoverscattered}, ...
                'MutationFcn', {@mutationuniform, 1/N}, ...
                'DistanceMeasureFcn', {@distancecrowding,'phenotype'});
        
            tic;
            % Run gamultiobj
            [x, fval, exitflag, output, population] = gamultiobj(fitnessFcn, N, [], [], [], [], [], [], options);
            % Stop timing and save execution time
            routine_execution_times = [routine_execution_times,toc * 1000];  % Convert time to milliseconds
    
            % Store the entire population of the last generation
            last_generations{end+1} = population;
            starting_population = population;
            
            % Calculate the sum of fitness values for each individual in the last generation
            sum_fvals = sum(fval, 2); % Sum across each row
            mean_fvalue = -(mean(sum_fvals));
            
            if first_loop ~= 1
                difference = abs(mean_fvalue - last_mean_fvalue);
                threshold = 0.05 * max(mean_fvalue, last_mean_fvalue); % 5% of the maximum value
                if difference < threshold
                    subsequent_diff = subsequent_diff + 1; % Increment count variable
                end
            end
    
            first_loop = 0;
        
            if subsequent_diff == 50
                % Initialize front array
                pareto_front = cell(size(x, 1), 1);
            
                % Iterate over each row of x
                for row = 1:size(x, 1)
                    % Find the indices where x(row, :) is equal to 1
                    indices = find(x(row, :) == 1);
                    % Subtract 1 from each index to start from 0
                    indices = indices - 1;
                    % Store the indices in the corresponding row of front
                    pareto_front{row} = indices;
                end
            
                % Store Pareto front
                pareto_fronts{rep} = pareto_front;
                break;
            end
            
            % Store the sum of fitness values in the last_fvalues array
            last_mean_fvalue = mean_fvalue;
    
            % Set up options for gamultiobj
            options = optimoptions('gamultiobj', ...
                'InitialPopulationMatrix', starting_population, ...
                'PopulationSize', M, ...
                'MaxGenerations', 2, ...
                'CrossoverFcn', {@crossoverscattered}, ...
                'MutationFcn', {@mutationuniform, 1/N}, ...
                'DistanceMeasureFcn', {@distancecrowding,'phenotype'});
        
            tic;
            % Run gamultiobj
            [x, fval, exitflag, output, population] = gamultiobj(fitnessFcn, N, [], [], [], [], [], [], options);
            routine_execution_times = [routine_execution_times,toc * 1000];  % Convert time to milliseconds
            
            starting_population = population;
            
            % Calculate the sum of fitness values for each individual in the last generation
            sum_fvals = sum(fval, 2); % Sum across each row
            mean_fvalue = -(mean(sum_fvals));
            
            if first_loop ~= 1
                difference = abs(mean_fvalue - last_mean_fvalue);
                threshold = 0.05 * max(mean_fvalue, last_mean_fvalue); % 5% of the maximum value
                if difference < threshold
                    subsequent_diff = subsequent_diff + 1; % Increment count variable
                end
            end
        
            if subsequent_diff == 50
                % Initialize front array
                pareto_front = cell(size(x, 1), 1);
            
                % Iterate over each row of x
                for row = 1:size(x, 1)
                    % Find the indices where x(row, :) is equal to 1
                    indices = find(x(row, :) == 1);
                    % Subtract 1 from each index to start from 0
                    indices = indices - 1;
                    % Store the indices in the corresponding row of front
                    pareto_front{row} = indices;
                end
            
                % Store Pareto front
                pareto_fronts{rep} = pareto_front;
                break;
            end
            
            % Store the sum of fitness values in the last_fvalues array
            last_mean_fvalue = mean_fvalue;
    
            % Set up options for gamultiobj
            options = optimoptions('gamultiobj', ...
                'InitialPopulationMatrix', starting_population, ...
                'PopulationSize', M, ...
                'MaxGenerations', 2, ...
                'CrossoverFcn', {@crossoverscattered}, ...
                'MutationFcn', {@mutationuniform, 1/N}, ...
                'DistanceMeasureFcn', {@distancecrowding,'phenotype'});
        
            tic;
            % Run gamultiobj
            [x, fval, exitflag, output, population] = gamultiobj(fitnessFcn, N, [], [], [], [], [], [], options);
            routine_execution_times = [routine_execution_times,toc * 1000];  % Convert time to milliseconds
            
            % Store the entire population of the last generation
            last_generations{end+1} = population;
            starting_population = population;
            
            % Calculate the sum of fitness values for each individual in the last generation
            sum_fvals = sum(fval, 2); % Sum across each row
            mean_fvalue = -(mean(sum_fvals));
            
            if first_loop ~= 1
                difference = abs(mean_fvalue - last_mean_fvalue);
                threshold = 0.05 * max(mean_fvalue, last_mean_fvalue); % 5% of the maximum value
                if difference < threshold
                    subsequent_diff = subsequent_diff + 1; % Increment count variable
                end
            end
    
            first_loop = 0;
        
            if subsequent_diff == 50
                % Initialize front array
                pareto_front = cell(size(x, 1), 1);
            
                % Iterate over each row of x
                for row = 1:size(x, 1)
                    % Find the indices where x(row, :) is equal to 1
                    indices = find(x(row, :) == 1);
                    % Subtract 1 from each index to start from 0
                    indices = indices - 1;
                    % Store the indices in the corresponding row of front
                    pareto_front{row} = indices;
                end
            
                % Store Pareto front
                pareto_fronts{rep} = pareto_front;
                break;
            end
            
            % Store the sum of fitness values in the last_fvalues array
            last_mean_fvalue = mean_fvalue;
        
            if i == ceil(max_iterations / 2)
                % Initialize front array
                pareto_front = cell(size(x, 1), 1);
            
                % Iterate over each row of x
                for row = 1:size(x, 1)
                    % Find the indices where x(row, :) is equal to 1
                    indices = find(x(row, :) == 1);
                    % Subtract 1 from each index to start from 0
                    indices = indices - 1;
                    % Store the indices in the corresponding row of front
                    pareto_front{row} = indices;
                end
            
                % Store Pareto front
                pareto_fronts{rep} = pareto_front;
                break;
            end
    
            % Set up options for gamultiobj
            options = optimoptions('gamultiobj', ...
                'InitialPopulationMatrix', last_generations{1,1}, ...
                'PopulationSize', M, ...
                'MaxGenerations', 1, ...
                'CrossoverFcn', {@crossoverscattered}, ...
                'MutationFcn', {@mutationuniform, 1/N}, ...
                'DistanceMeasureFcn', {@distancecrowding,'phenotype'});
    
            % Run gamultiobj
            [x] = gamultiobj(fitnessFcn, N, [], [], [], [], [], [], options);
            x_t = x;
            x_t_rows = size(x,1);
    
            % Set up options for gamultiobj
            options = optimoptions('gamultiobj', ...
                'InitialPopulationMatrix', last_generations{1,2}, ...
                'PopulationSize', M, ...
                'MaxGenerations', 1, ...
                'CrossoverFcn', {@crossoverscattered}, ...
                'MutationFcn', {@mutationuniform, 1/N}, ...
                'DistanceMeasureFcn', {@distancecrowding,'phenotype'});
    
            % Run gamultiobj
            [x, fval, exitflag, output, population] = gamultiobj(fitnessFcn, N, [], [], [], [], [], [], options);
            x_t2 = x;
            x_t2_rows = size(x,1);
            
            if x_t2_rows <= x_t_rows        
                % Get the top 50% of individuals in x
                num_individuals = size(x_t2, 1);
                top_half_index = 1:round(num_individuals/2);
                p_t2 = x_t2(top_half_index, :);
                p_t = x_t(top_half_index, :);
            end
            if x_t2_rows > x_t_rows        
                % Get the top 50% of individuals in x
                num_individuals = size(x_t, 1);
                top_half_index = 1:round(num_individuals/2);
                p_t2 = x_t2(top_half_index, :);
                p_t = x_t(top_half_index, :);
            end
    
            [U,S,V] = svd(p_t);
            S_t = S;
            V_t = V;
    
            [U,S,V] = svd(p_t2);
            U_t2 = U;
            S_t2 = S;
            V_t2 = V;
    
            V = V_t2 - V_t;
            S = S_t2 - S_t;
    
            % Initialize matrix for V_orth
            V_orth = zeros(size(V));
            
            % Iterate over each column of V
            for j = 1:size(V, 2)
                % Step 1: Compute the corresponding vector v_i_orth with elements in reverse order
                v_i = V(:, j);
                v_i_orth = flip(v_i);
                
                % Step 2: Generate a random number between 0 and 1
                n = rand;
                
                % Step 3: Flip the sign of the first or second half of v_i_orth based on the random number
                if n > 0.5
                    v_i_orth(1:end/2) = -v_i_orth(1:end/2);
                else
                    v_i_orth(end/2+1:end) = -v_i_orth(end/2+1:end);
                end
                
                % Step 4: Randomly modify one element of the unmodified half of v_i_orth based on the value of n
                if n > 0.5
                    unmodified_half_start = 1;
                    unmodified_half_end = floor(size(V, 1) / 2);
                else
                    unmodified_half_start = floor(size(V, 1) / 2) + 1;
                    unmodified_half_end = size(V, 1);
                end
                unmodified_half = randi([unmodified_half_start, unmodified_half_end]);
                v_i_orth(unmodified_half) = rand; % Randomly modify one element
                
                % Assign v_i_orth to corresponding column of V_orth
                V_orth(:, j) = v_i_orth;
            end
    
            % Calculate (S_t2 + S)
            S_sum = S_t2 + S;
            
            % Calculate (V_t2 + V_orth)
            V_sum = V_t2 + V_orth;
            
            % Calculate transposed(V_t2 + V_orth)
            V_sum_transpose = V_sum';
            
            % Calculate U_t2 * (S_t2 + S) * transposed(V_t2 + V_orth)
            P_orth = U_t2 * S_sum * V_sum_transpose;

            % Initialize matrix for P_final
            P_binary = zeros(size(P_orth));

            % Apply the condition: P_final(i,j) is 0 if P_orth(i,j) < 0.5, 1 otherwise
            P_binary(P_orth >= 0.5) = 1;

            % Calculate fitness values for each individual in the population
            population = last_generations{1, 2};
            %total_fitness = computeFitness(population, costs, coverage, revealed_failures);

            fitness_values = zeros(size(population, 1), 1);
            for z = 1:size(population, 1)
                fitness_values(z) = computeFitness(population(z, :), costs, coverage, revealed_failures);
            end

            % Find indices of the worst individuals (rows)
            [~, sorted_indices] = sort(fitness_values);
            worst_indices = sorted_indices(1:size(P_binary, 1)); % Select worst k individuals

            % Replace worst individuals with corresponding individuals from P_binary
            updated_population = population;
            updated_population(worst_indices, :) = P_binary;

            starting_population = updated_population;
            last_generations = {};

        end

        execution_times(rep) = sum(routine_execution_times);
    end

    % Show each element
    disp('Execution times:');
    for i = 1:length(execution_times)
        fprintf('Element %d: %.2f\n', i, execution_times(i));
    end
    
    % Calculate and show the standard deviation
    exec_times_std_dev = std(execution_times);
    fprintf('Standard Deviation: %.2f\n', exec_times_std_dev);
    
    % Calculate mean execution time
    mean_execution_time = mean(execution_times);
    
    % Display mean execution time
    fprintf('DIVGA mean execution time (ms): %f\n', mean_execution_time);
    
    % Save Pareto fronts and mean execution time to JSON file
    json_data = struct();
    for i = 1:10
        field_name = sprintf('%s_pareto_front_%d', program, i-1);
        json_data.(field_name) = pareto_fronts{i};
    end
    json_data.execution_times = execution_times
    json_data.DIVGA_mean_execution_time_ms = mean_execution_time;
    json_data.std_dev = exec_times_std_dev;
    
    json_file = sprintf('../results/divga/%s_pareto_fronts_divga.json', program);
    json_str = jsonencode(json_data);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    disp('JSON file saved successfully.');
end
