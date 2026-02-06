function bpsk(data_samp_count, save_path, output_len, fs, Tsymb, fc)
    % convert all inputs to double
    data_samp_count = double(data_samp_count);
    output_len      = double(output_len);
    fs              = double(fs);
    Tsymb           = double(Tsymb);
    fc              = double(fc);

    % create save path directory if it doesn't already exist
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end
    
    samp_per_symb = fs * Tsymb;
    for n = 1:data_samp_count
        % create random bit sequence
        bit_seq = randi([0 1], output_len/samp_per_symb, 1);

        % Reshape bits into symbols
        bpsk_imp_train = bit_seq*2 - 1;
        bpsk_bb = repelem(bpsk_imp_train, samp_per_symb);

        % set up carrier
        t = (0:output_len-1) / fs;  % t is now double
        carrier = cos(2*pi*fc*t).';  % works without error

        % upconvert to passband
        bpsk_pb = bpsk_bb .* carrier;

        % save file
        filename = fullfile(save_path, sprintf('bpsk_%d.npy', n));
        save(filename, 'bpsk_pb')
    end
end
