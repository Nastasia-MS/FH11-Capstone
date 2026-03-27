%% Test Waveform Generator
% Visual test for waveform_generator.m
% Tests PAM, QAM, PSK, FSK modulation schemes with graphical output

clear all; close all; clc;

% Test Parameters
test_cases = {
    % Name, N, fs, Tsymb, fc, M, modulation, alpha, span, pulse_shape
    'PAM-4',     256, 8e6, 1e-6, 1e6, 4,   'PAM',  0.35, 8, 'rrc';
    'QAM-16',    256, 8e6, 1e-6, 1e6, 16,  'QAM',  0.35, 8, 'rrc';
    'PSK-8',     256, 8e6, 1e-6, 1e6, 8,   'PSK',  0.35, 8, 'rrc';
    'FSK-2',     256, 8e6, 1e-6, 1e6, 2,   'FSK',  0.35, 8, 'rrc';
};

% Create figure with subplots
figure('Position', [100, 100, 1200, 900]);

% Run tests and plot
for i = 1:size(test_cases, 1)
    test_name = test_cases{i, 1};
    N = test_cases{i, 2};
    fs = test_cases{i, 3};
    Tsymb = test_cases{i, 4};
    fc = test_cases{i, 5};
    M = test_cases{i, 6};
    modulation = test_cases{i, 7};
    alpha = test_cases{i, 8};
    span = test_cases{i, 9};
    pulse_shape = test_cases{i, 10};
    
    try
        % Call waveform generator
        waveform = waveform_generator(N, fs, Tsymb, fc, M, modulation, ...
            'alpha', alpha, 'span', span, 'pulse_shape', pulse_shape);
        
        % Create time vector (microseconds)
        t = (0:length(waveform)-1) / fs * 1e6;
        
        % Plot waveform
        subplot(2, 2, i);
        plot(t, waveform, 'LineWidth', 1.5);
        grid on;
        xlabel('Time (\mu s)');
        ylabel('Amplitude');
        title(sprintf('%s (N=%d, M=%d)', test_name, N, M));
        
        % Print statistics to console
        fprintf('Test %d: %s\n', i, test_name);
        fprintf('  Output length: %d samples\n', length(waveform));
        fprintf('  Duration: %.2f µs\n', t(end));
        fprintf('  Power (avg): %.4f\n', mean(abs(waveform).^2));
        fprintf('  Peak amplitude: %.4f\n', max(abs(waveform)));
        fprintf('\n');
        
    catch ME
        % Plot error message
        subplot(2, 2, i);
        text(0.5, 0.5, sprintf('Error:\n%s', ME.message), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 10, 'Color', 'red');
        axis off;
        
        fprintf('Test %d: %s - FAILED\n', i, test_name);
        fprintf('  Error: %s\n\n', ME.message);
    end
end
