function sig = waveform_generator(output_len, fs, Tsymb, fc, M, modulation, varargin)
    % WAVEFORM_GENERATOR - Unified RF waveform generator with pulse shaping
    %
    % Generates baseband (complex IQ) or passband (real) waveforms.
    %
    % Inputs:
    %   output_len   - Length of output vector (total samples)
    %   fs           - Sample Rate (Hz)
    %   Tsymb        - Symbol Period (s)
    %   fc           - Carrier Frequency (Hz)
    %   M            - Modulation order
    %   modulation   - 'PAM', 'QAM', 'PSK', 'FSK', 'FHSS'
    %   varargin     - Optional name-value pairs:
    %                  'alpha' (default 0.35) - RRC roll-off
    %                  'span' (default 8) - Filter span in symbols
    %                  'pulse_shape' (default 'rrc') - 'rrc' or 'rect'
    %                  'output_type' (default 'passband') - 'baseband' or 'passband'
    %
    % Output:
    %   sig          - Waveform signal
    %                  baseband: complex-valued (QAM/PSK/FSK/FHSS), real (PAM)
    %                  passband: real-valued
    %
    % Examples:
    %   % QAM passband (default)
    %   sig = waveform_generator(98304, 48000, 0.001, 6000, 16, 'QAM');
    %
    %   % QAM baseband
    %   sig = waveform_generator(98304, 48000, 0.001, 6000, 16, 'QAM', ...
    %                            'output_type', 'baseband');
    %
    %   % PAM with rectangular pulses
    %   sig = waveform_generator(98304, 48000, 0.001, 6000, 4, 'PAM', ...
    %                            'pulse_shape', 'rect');
    %
    %   % PSK with custom roll-off
    %   sig = waveform_generator(98304, 48000, 0.001, 6000, 8, 'PSK', ...
    %                            'alpha', 0.5);

    %% Parse optional arguments
    p = inputParser;
    addParameter(p, 'alpha', 0.35, @isnumeric);      % RRC roll-off
    addParameter(p, 'span', 8, @isnumeric);          % Filter span
    addParameter(p, 'pulse_shape', 'rrc', @ischar);  % Pulse shape type
    addParameter(p, 'output_type', 'passband', @ischar);  % Output type
    parse(p, varargin{:});

    alpha = p.Results.alpha;
    span = p.Results.span;
    pulse_shape = p.Results.pulse_shape;
    output_type = p.Results.output_type;

    %% Common parameters
    sps = round(fs * Tsymb);  % Samples per symbol (must be integer)

    %% Design pulse shaping filter
    if strcmp(pulse_shape, 'rrc')
        h = rcosdesign(alpha, span, sps, 'sqrt');
        filter_delay = span * sps / 2;  % Group delay
    else
        % Rectangular pulse (backward compatibility)
        h = ones(sps, 1) / sqrt(sps);  % Normalized
        filter_delay = 0;
    end

    %% Calculate number of symbols needed
    % Account for filter delay on both sides
    total_samples_needed = output_len + 2 * filter_delay;
    num_symbols = ceil(total_samples_needed / sps);

    %% Handle FSK
    if strcmpi(modulation, 'FSK')
        if strcmpi(output_type, 'baseband')
            sig = generate_fsk_baseband(output_len, fs, Tsymb, M);
        else
            sig = generate_fsk_passband(output_len, fs, Tsymb, fc, M);
        end
        return;
    end

    %% Handle FHSS
    if strcmpi(modulation, 'FHSS')
        if strcmpi(output_type, 'baseband')
            sig = generate_fhss_baseband(output_len, fs, Tsymb, M);
        else
            sig = generate_fhss_passband(output_len, fs, Tsymb, fc, M);
        end
        return;
    end

    %% Generate symbols based on modulation type
    symbols = generate_symbols(num_symbols, M, modulation);

    %% Apply pulse shaping
    % upfirdn: upsample by sps, filter with h, downsample by 1
    sig_bb = upfirdn(symbols, h, sps, 1);

    %% Trim to exact output length
    % Remove filter delay from start
    start_idx = filter_delay + 1;
    end_idx = start_idx + output_len - 1;

    % Ensure we don't exceed signal length
    if end_idx > length(sig_bb)
        % Pad with zeros if needed
        sig_bb(end_idx) = 0;
    end

    sig_bb = sig_bb(start_idx:end_idx);

    %% Return baseband or upconvert to passband
    if strcmpi(output_type, 'baseband')
        sig = complex(sig_bb);  % Ensure complex dtype for all baseband (PAM is real-valued but needs complex type for Python-side branching)
    else
        sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
    end
end


%% ========================================================================
%  HELPER FUNCTIONS (Local to this file)
%% ========================================================================

function sig_bb = generate_fsk_baseband(output_len, fs, Tsymb, M)
    % GENERATE_FSK_BASEBAND - Generate complex baseband CPFSK signal
    %
    % Returns complex exponentials: exp(j*phase) with no carrier offset.

    sps = round(fs * Tsymb);
    freq_sep = 1 / Tsymb;

    num_symbols = ceil(output_len / sps);
    data = randi([0 M-1], num_symbols, 1);

    % Frequency offsets centered around zero (baseband)
    freq_offsets = (data - (M-1)/2) * freq_sep;

    sig_bb = zeros(output_len, 1);
    phase = 0;

    sample_idx = 1;
    for sym = 1:num_symbols
        f_offset = freq_offsets(sym);

        num_samp = min(sps, output_len - sample_idx + 1);
        if num_samp <= 0
            break;
        end

        for k = 1:num_samp
            sig_bb(sample_idx) = exp(1j * phase);
            phase = phase + 2*pi * f_offset / fs;
            sample_idx = sample_idx + 1;
        end

        phase = mod(phase, 2*pi);
    end
end


function sig_pb = generate_fsk_passband(output_len, fs, Tsymb, fc, M)
    % GENERATE_FSK_PASSBAND - Generate real-valued passband CPFSK signal

    sps = round(fs * Tsymb);  % Samples per symbol (must be integer)
    freq_sep = 1 / Tsymb;  % Minimum orthogonal frequency separation

    % Calculate number of symbols
    num_symbols = ceil(output_len / sps);

    % Generate random data symbols (0 to M-1)
    data = randi([0 M-1], num_symbols, 1);

    % Map data to frequency offsets centered around fc
    % e.g., for M=4: [-1.5, -0.5, 0.5, 1.5] * freq_sep
    freq_offsets = (data - (M-1)/2) * freq_sep;

    % Generate CPFSK signal with continuous phase
    sig_pb = zeros(output_len, 1);
    phase = 0;  % Running phase for continuity

    sample_idx = 1;
    for sym = 1:num_symbols
        % Instantaneous frequency for this symbol
        f_inst = fc + freq_offsets(sym);

        % Generate samples for this symbol
        num_samp = min(sps, output_len - sample_idx + 1);
        if num_samp <= 0
            break;
        end

        for k = 1:num_samp
            sig_pb(sample_idx) = cos(phase);
            phase = phase + 2*pi * f_inst / fs;
            sample_idx = sample_idx + 1;
        end

        % Keep phase bounded to avoid numerical issues
        phase = mod(phase, 2*pi);
    end
end


function sig_bb = generate_fhss_baseband(output_len, fs, Tsymb, M)
    % GENERATE_FHSS_BASEBAND - Generate complex baseband FHSS signal
    %
    % Returns complex exponentials with hopping frequency offsets around zero.

    sps = round(fs * Tsymb);
    channel_spacing = fs / (2 * M);

    num_hops = ceil(output_len / sps);
    hop_sequence = randi([0 M-1], num_hops, 1);

    % Frequency offsets centered around zero (baseband)
    freq_offsets = (hop_sequence - (M-1)/2) * channel_spacing;

    sig_bb = zeros(output_len, 1);
    phase = 0;

    sample_idx = 1;
    for hop = 1:num_hops
        f_offset = freq_offsets(hop);

        num_samp = min(sps, output_len - sample_idx + 1);
        if num_samp <= 0
            break;
        end

        for k = 1:num_samp
            sig_bb(sample_idx) = exp(1j * phase);
            phase = phase + 2*pi * f_offset / fs;
            sample_idx = sample_idx + 1;
        end

        phase = mod(phase, 2*pi);
    end
end


function sig_pb = generate_fhss_passband(output_len, fs, Tsymb, fc, M)
    % GENERATE_FHSS_PASSBAND - Generate real-valued passband FHSS signal

    sps = round(fs * Tsymb);  % Samples per hop

    % Hopping bandwidth - spread channels around fc
    % Channel spacing to avoid overlap
    channel_spacing = fs / (2 * M);  % Spread across half the bandwidth
    hop_bw = channel_spacing * (M - 1);

    % Calculate number of hops
    num_hops = ceil(output_len / sps);

    % Generate pseudo-random hopping sequence (0 to M-1)
    hop_sequence = randi([0 M-1], num_hops, 1);

    % Map hop indices to frequency offsets centered around fc
    freq_offsets = (hop_sequence - (M-1)/2) * channel_spacing;

    % Generate FHSS signal with continuous phase between hops
    sig_pb = zeros(output_len, 1);
    phase = 0;  % Running phase for continuity

    sample_idx = 1;
    for hop = 1:num_hops
        % Instantaneous frequency for this hop
        f_inst = fc + freq_offsets(hop);

        % Generate samples for this hop period
        num_samp = min(sps, output_len - sample_idx + 1);
        if num_samp <= 0
            break;
        end

        for k = 1:num_samp
            sig_pb(sample_idx) = cos(phase);
            phase = phase + 2*pi * f_inst / fs;
            sample_idx = sample_idx + 1;
        end

        % Keep phase bounded to avoid numerical issues
        phase = mod(phase, 2*pi);
    end
end


function symbols = generate_symbols(num_symbols, M, modulation)
    % GENERATE_SYMBOLS - Generate random modulated symbols
    %
    % Unified symbol generation for different modulation schemes

    % Generate random data
    data = randi([0 M-1], num_symbols, 1);

    switch upper(modulation)
        case 'PAM'
            % Pulse Amplitude Modulation
            symbols = pammod(data, M);
            % Normalize for unit average power
            symbols = symbols / sqrt(mean(abs(symbols).^2));

        case 'QAM'
            % Quadrature Amplitude Modulation
            symbols = qammod(data, M, 'UnitAveragePower', true);

        case 'PSK'
            % Phase Shift Keying
            symbols = pskmod(data, M, 0, 'gray');  % Gray coding
            % Already unit power

        otherwise
            error('Unsupported modulation type: %s', modulation);
    end
end


function sig_pb = upconvert_to_passband(sig_bb, fs, fc, output_len)
    % UPCONVERT_TO_PASSBAND - Convert complex baseband to real passband
    %
    % Uses quadrature upconversion: s(t) = I(t)*cos(wc*t) - Q(t)*sin(wc*t)
    % This is the standard method for all digital modulations

    % Ensure correct length
    sig_bb = sig_bb(1:output_len);

    % Extract I and Q components
    I = real(sig_bb);
    Q = imag(sig_bb);

    % Generate carriers
    t = (0:output_len-1)' / fs;
    carrier_I = cos(2*pi*fc*t);
    carrier_Q = -sin(2*pi*fc*t);

    % Quadrature upconversion
    sig_pb = I .* carrier_I + Q .* carrier_Q;
end


%% ========================================================================
%  BACKWARD COMPATIBILITY WRAPPER FUNCTIONS
%% ========================================================================
% These allow old code to still work, but they call the new unified function

function pam_pb = pam_gui(output_len, fs, Tsymb, fc, M, Var)
    % PAM_GUI - Legacy wrapper for backward compatibility
    % Calls waveform_generator with 'rect' pulse shape
    pam_pb = waveform_generator(output_len, fs, Tsymb, fc, M, 'PAM', ...
                                'pulse_shape', 'rect');
end


function mqam_pb = mqam_gui(output_len, fs, Tsymb, fc, M)
    % MQAM_GUI - Legacy wrapper for backward compatibility
    % Calls waveform_generator with 'rect' pulse shape
    mqam_pb = waveform_generator(output_len, fs, Tsymb, fc, M, 'QAM', ...
                                 'pulse_shape', 'rect');
end


function fsk_pb = fsk_gui(output_len, fs, Tsymb, fc, M, freq_sep)
    % FSK_GUI - Legacy wrapper for backward compatibility
    % Calls waveform_generator with 'rect' pulse shape
    if nargin < 6
        freq_sep = 1/Tsymb;  % Default frequency separation
    end
    fsk_pb = waveform_generator(output_len, fs, Tsymb, fc, M, 'FSK', ...
                                'pulse_shape', 'rect');
end