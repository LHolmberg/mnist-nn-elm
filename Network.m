classdef Network
    properties
        test_data
        train_data
        
        train_imgs
        train_lbls
        test_imgs
        test_lbls
        
        numOfNeurons = 4000;
    end
    methods
        function obj = Network
            obj.train_data = readmatrix('mnist_train.csv');
            obj.test_data = readmatrix('mnist_test.csv');
            
            obj.train_imgs = obj.train_data(:, 2:end)'./255;
            obj.train_lbls = obj.vectorToMat(obj.train_data(:, 1)');
            obj.test_imgs = obj.test_data(:, 2:end)'./255;
            obj.test_lbls = obj.vectorToMat(obj.test_data(:, 1)');
        end
        
        function aFn = ActivationFn(obj, x, name)
            if name == "tanh"
                aFn = tanh(x);
            elseif name == "sigmoid"
                aFn = 1./(1+exp(-x));
            end
        end
        
        function y = vectorToMat(obj, vec) % Tar 60000x1 vek. och gör den till 60000x10 (0...9) mat.
            y = zeros(10, length(vec));

            for i = 1 : length(vec)
                y(vec(i) + 1, i) = 1.0;
            end
        end
        
        function Run(obj)
            %% Lös ekvationen och hitta + sätt nya värdena
            W = rand(obj.numOfNeurons, size(obj.train_imgs, 1)) * 2 - 1;

            y = obj.ActivationFn(W*obj.train_imgs, "sigmoid");
            W2 = y' \obj.train_lbls';
            y2 =(y' * W2)';

            y_t = obj.ActivationFn(W * obj.test_imgs, "sigmoid");
            y2_t = (y_t' * W2)';
            
            %% Kolla hur många rätt modellen gav (på tränings datan)
            correctlyPredicted_train = 0;
            for i = 1 : size(obj.train_lbls, 2)
                [~, pred] = max(obj.train_lbls(:, i));
                [~, act] = max(y2(:, i));
                if act == pred
                    correctlyPredicted_train = correctlyPredicted_train + 1;
                end
            end
            accuracy_train = correctlyPredicted_train / size(obj.train_lbls, 2) * 100;
            
            %% Kolla hur många rätt modellen gav (på test datan)
            correctlyPredicted_test = 0;
            for i = 1 : size(obj.test_lbls, 2)
                [~, pred] = max(obj.test_lbls(:, i));
                [~, act] = max(y2_t(:, i));
                if act == pred
                    correctlyPredicted_test = correctlyPredicted_test + 1;
                end
            end
            accuracy_test = correctlyPredicted_test / size(obj.test_lbls, 2) * 100;
            
            %% Visa informationen på ett tydligt sätt
            disp("#################################################################");
            disp([newline, 'Results: | Training accuracy: ', num2str(accuracy_train), '% | Testing accuracy: ', num2str(accuracy_test), '% |', newline])
            disp("#################################################################");
        end
    end
end