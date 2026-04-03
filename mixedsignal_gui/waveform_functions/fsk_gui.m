function fsk_pb = fsk_gui(output_len, fs, Tsymb, fc, M, freq_sep)
    % fsk_gui - Generate M-ary FSK modulated signal
    %
    % Inputs:
    %   output_len   - Length of output vector (total samples)
    %   fs          - Sample Rate (Hz)
    %   Tsymb       - Symbol Period (s)
    %   fc          - Carrier Frequency (Hz)
    %   M           - Modulation order (number of frequencies)
    %   freq_sep    - Frequency separation between tones (Hz)
    %
    % Output:
    %   fsk_pb      - FSK modulated passband signal
    %
    % ***** IMPORTANT *****
    % constraints (errors not handled):
    % - M must be a power of 2
    % - fs*Tsymb must be an integer
    % - fs*Tsymb/log2(M) must be an integer
    % - output_len/(fs*Tsymb) must be an integer
    % *********************

    % Set default frequency separation if not provided
    if nargin < 6
        freq_sep = 1/Tsymb;  % Default: minimum orthogonal spacing
    end
    
    samp_per_symb = fs*Tsymb;
    bit_per_symb = log2(M);
    samp_per_bit = samp_per_symb / bit_per_symb;

    % Create random bit sequence
    bit_seq = randi([0 1], output_len/samp_per_bit, 1);

    % Reshape bits into symbols (0 to M-1)
    data_symbols = bi2de(reshape(bit_seq, bit_per_symb, output_len/samp_per_symb).', 'left-msb');

    % Generate FSK signal
    fsk_pb = zeros(output_len, 1);
    
    % Time vector for one symbol
    t_symb = (0:samp_per_symb-1)' / fs;
    
    % Generate each symbol
    for i = 1:length(data_symbols)
        % Calculate frequency for this symbol
        % Frequencies centered around fc
        freq_offset = (data_symbols(i) - (M-1)/2) * freq_sep;
        f_symbol = fc + freq_offset;
        
        % Generate samples for this symbol
        start_idx = (i-1)*samp_per_symb + 1;
        end_idx = i*samp_per_symb;
        
        % Phase continuity: use cumulative phase
        if i == 1
            phase_offset = 0;
        else
            % Calculate phase at end of previous symbol
            prev_freq = fc + (data_symbols(i-1) - (M-1)/2) * freq_sep;
            phase_offset = mod(2*pi*prev_freq*samp_per_symb/fs + phase_offset, 2*pi);
        end
        
        % Generate FSK tone with phase continuity
        fsk_pb(start_idx:end_idx) = cos(2*pi*f_symbol*t_symb + phase_offset);
    end

end