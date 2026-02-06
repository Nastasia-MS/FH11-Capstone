function mqam_pb =  mqam_gui(output_len, fs, Tsymb, fc, M)
    % data_samp_count      % Number of data samples to generate
    % save_path            % Path where .npy is saved
    % output_len           % Length of each saved output vector
    % M                    % Modulation order (must be square number)
    % fs                   % Sample Rate (Hz)
    % Tsymb                % Symbol Period (s)
    % fc                   % Carrier Frequency (Hz)
    
    % ***** IMPORTANT *****
    % constraints (errors not handled)
    % sqrt(M) must be an integer
    % fs*Tsymb must be an integer
    % fs*Tsymb/log2(M) must be an integer
    % output_len/(fs*Tsymb) must be an integer
    % *********************


    
    samp_per_symb = fs*Tsymb;
    bit_per_symb = log2(M);
    samp_per_bit = samp_per_symb / bit_per_symb;

    % create random bit sequence
    bit_seq = randi([0 1], output_len/samp_per_bit, 1);

    % Reshape bits into symbols
    data_symbols = bi2de(reshape(bit_seq, bit_per_symb, output_len/samp_per_symb).', 'left-msb');

    % mQAM modulation (baseband)
    mqam_imp_train = qammod(data_symbols, M, 'UnitAveragePower', true);
    mqam_bb = repelem(mqam_imp_train, samp_per_symb);

    % set up carrier
    t = (0:output_len-1) / fs;
    carrier = cos(2*pi*fc*t).';
    
    % upconvert to passband
    mqam_pb = mqam_bb .* carrier;

end
