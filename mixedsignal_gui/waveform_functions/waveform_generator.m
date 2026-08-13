function [sig, symbols_out] = waveform_generator(output_len, fs, Tsymb, fc, M, modulation, varargin)
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
    %   modulation   - 'PAM', 'QAM', 'PSK', 'FSK', 'FHSS',
    %                  'LFM', 'Barker', 'FMCW', 'WiFi', 'LTE', '5G_NR'
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
    %   symbols_out  - Baseband symbols (empty for FSK/FHSS/radar/standards)
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

    symbols_out = [];

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
        symbols_out = [];
        return;
    end

    %% Handle FHSS
    if strcmpi(modulation, 'FHSS')
        if strcmpi(output_type, 'baseband')
            sig = generate_fhss_baseband(output_len, fs, Tsymb, M);
        else
            sig = generate_fhss_passband(output_len, fs, Tsymb, fc, M);
        end
        symbols_out = [];
        return;
    end

    %% ---- Radar waveforms (Phased Array System Toolbox) ----

    if strcmpi(modulation, 'LFM')
        sig = generate_lfm_signal(output_len, fs, Tsymb, fc, M, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, 'Barker')
        sig = generate_barker_signal(output_len, fs, Tsymb, fc, M, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, 'FMCW')
        sig = generate_fmcw_signal(output_len, fs, Tsymb, fc, M, output_type);
        symbols_out = [];
        return;
    end

    %% ---- Standards-based waveforms (WLAN / LTE / 5G Toolboxes) ----

    if strcmpi(modulation, 'WiFi')
        sig = generate_wifi_signal(output_len, fs, fc, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, 'LTE')
        sig = generate_lte_signal(output_len, fs, fc, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, '5G_NR')
        sig = generate_5gnr_signal(output_len, fs, fc, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, 'Zigbee')
        sig = generate_zigbee_signal(output_len, fs, fc, output_type);
        symbols_out = [];
        return;
    end

    if strcmpi(modulation, 'LoRa')
        sig = generate_lora_signal(output_len, fs, Tsymb, fc, M, output_type);
        symbols_out = [];
        return;
    end

    %% Generate symbols based on modulation type
    symbols = generate_symbols(num_symbols, M, modulation);
    symbols_out = symbols(:);

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
%  ADVANCED WAVEFORM HELPERS (Toolbox-based)
%% ========================================================================

function sig = generate_lfm_signal(output_len, fs, Tsymb, fc, M, output_type)
    % GENERATE_LFM_SIGNAL - Linear Frequency Modulated (chirp) pulse
    % Requires: Phased Array System Toolbox
    try
        sweep_bw = M * fs / 8;         % M scales the sweep bandwidth
        pulse_width = Tsymb * 20;       % longer pulse for visible chirp
        pulse_width = min(pulse_width, (output_len - 1) / fs);  % cap at signal length

        % Snap pulse_width so it contains a whole number of samples
        pulse_width = round(pulse_width * fs) / fs;

        % PRF must satisfy: fs / PRF is an integer
        pri_desired = pulse_width * 1.5;
        pri_samples = max(round(pri_desired * fs), round(pulse_width * fs) + 1);
        prf = fs / pri_samples;

        wav = phased.LinearFMWaveform( ...
            'SweepBandwidth', sweep_bw, ...
            'PulseWidth', pulse_width, ...
            'SampleRate', fs, ...
            'PRF', prf, ...
            'OutputFormat', 'Samples', ...
            'NumSamples', output_len);
        sig_bb = wav();

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('LFM generation failed (need Phased Array System Toolbox): %s', ME.message);
    end
end


function sig = generate_barker_signal(output_len, fs, Tsymb, fc, M, output_type)
    % GENERATE_BARKER_SIGNAL - Phase-coded pulse using Barker sequence
    % Requires: Phased Array System Toolbox
    % M selects num_chips: valid Barker lengths are 2,3,4,5,7,11,13
    try
        valid_chips = [2 3 4 5 7 11 13];
        [~, idx] = min(abs(valid_chips - M));
        num_chips = valid_chips(idx);

        chip_width = Tsymb;
        chip_width = min(chip_width, (output_len - 1) / (fs * num_chips));

        % Snap chip_width so it contains a whole number of samples (min 1)
        chip_width = max(round(chip_width * fs), 1) / fs;

        % PRF must satisfy: fs / PRF is an integer
        pulse_dur = chip_width * num_chips;
        pri_desired = pulse_dur * 1.5;
        pri_samples = max(round(pri_desired * fs), round(pulse_dur * fs) + 1);
        prf = fs / pri_samples;

        wav = phased.PhaseCodedWaveform( ...
            'Code', 'Barker', ...
            'NumChips', num_chips, ...
            'ChipWidth', chip_width, ...
            'SampleRate', fs, ...
            'PRF', prf, ...
            'OutputFormat', 'Samples', ...
            'NumSamples', output_len);
        sig_bb = wav();

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('Barker generation failed (need Phased Array System Toolbox): %s', ME.message);
    end
end


function sig = generate_fmcw_signal(output_len, fs, Tsymb, fc, M, output_type)
    % GENERATE_FMCW_SIGNAL - Frequency Modulated Continuous Wave
    % Requires: Phased Array System Toolbox
    try
        sweep_bw = M * fs / 8;
        sweep_time = Tsymb * 50;
        sweep_time = min(sweep_time, (output_len - 1) / fs);

        % Snap sweep_time so it contains a whole number of samples
        sweep_time = max(round(sweep_time * fs), 1) / fs;

        wav = phased.FMCWWaveform( ...
            'SweepBandwidth', sweep_bw, ...
            'SweepTime', sweep_time, ...
            'SampleRate', fs, ...
            'SweepDirection', 'Triangle', ...
            'OutputFormat', 'Samples', ...
            'NumSamples', output_len);
        sig_bb = wav();

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('FMCW generation failed (need Phased Array System Toolbox): %s', ME.message);
    end
end


function sig = generate_wifi_signal(output_len, fs, fc, output_type)
    % GENERATE_WIFI_SIGNAL - 802.11ax (WiFi 6) HE-SU packet
    % Requires: WLAN Toolbox
    try
        cfg = wlanHESUConfig;
        cfg.ChannelBandwidth = 'CBW20';     % 20 MHz
        cfg.MCS = 0;                         % BPSK, rate 1/2

        % Generate random data bits matching the required PSDU length
        psduLen = getPSDULength(cfg);        % bytes
        dataBits = randi([0 1], psduLen * 8, 1);

        % Generate at native rate, then resample to fs
        sig_bb_native = wlanWaveformGenerator(dataBits, cfg);
        native_fs = 20e6;  % CBW20 native rate

        % Resample to target fs
        if abs(fs - native_fs) > 1
            [P, Q] = rat(fs / native_fs, 1e-6);
            sig_bb = resample(sig_bb_native(:,1), P, Q);
        else
            sig_bb = sig_bb_native(:,1);
        end

        % Repeat to fill output_len if needed
        while length(sig_bb) < output_len
            sig_bb = [sig_bb; sig_bb]; %#ok<AGROW>
        end
        sig_bb = sig_bb(1:output_len);

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('WiFi generation failed (need WLAN Toolbox): %s', ME.message);
    end
end


function sig = generate_lte_signal(output_len, fs, fc, output_type)
    % GENERATE_LTE_SIGNAL - LTE downlink reference measurement channel
    % Requires: LTE Toolbox
    try
        % R.9 = 20 MHz BW, 100 resource blocks, 64-QAM, 1 antenna.  Its native
        % sampling rate is exactly 30.72 Msps, so the resample below is a no-op
        % at that rate.  R.0 (1.4 MHz) was used previously; at 30.72 Msps it
        % occupies only ~8% of the band, whereas real 20 MHz LTE/OFDM captures
        % occupy ~60%.
        bits = randi([0 1], 20000, 1);
        [sig_bb_native, ~, rmccfg] = lteRMCDLTool('R.9', bits);
        native_fs = rmccfg.SamplingRate;

        % Resample to target fs
        if abs(fs - native_fs) > 1
            [P, Q] = rat(fs / native_fs, 1e-6);
            sig_bb = resample(sig_bb_native(:,1), P, Q);
        else
            sig_bb = sig_bb_native(:,1);
        end

        % Fit to output_len, clipping at a random offset when the frame is
        % longer (see tile_or_clip_random: a fixed offset lands in the
        % deterministic preamble and makes every call identical).
        sig_bb = tile_or_clip_random(sig_bb, output_len);

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('LTE generation failed (need LTE Toolbox): %s', ME.message);
    end
end


function sig = generate_5gnr_signal(output_len, fs, fc, output_type)
    % GENERATE_5GNR_SIGNAL - 5G NR downlink waveform
    % Requires: 5G Toolbox
    try
        % 20 MHz FR1 carrier at 15 kHz SCS = 106 resource blocks, whose native
        % sampling rate is exactly 30.72 Msps (so the resample below is a no-op
        % at that rate).  Setting ChannelBandwidth alone is NOT sufficient: the
        % grid and bandwidth part keep their 52-RB (10 MHz) defaults, and the
        % PRBSet below would then exceed the BWP size and error out.  All three
        % must be widened together.
        cfg = nrDLCarrierConfig;
        cfg.ChannelBandwidth = 20;          % 20 MHz FR1
        cfg.NumSubframes = 10;              % 10 ms frame
        cfg.SCSCarriers{1}.NSizeGrid = 106;
        cfg.BandwidthParts{1}.NSizeBWP = 106;
        pdsch = cfg.PDSCH{1};
        pdsch.PRBSet = 0:105;               % fill the carrier
        cfg.PDSCH{1} = pdsch;

        [sig_bb_native, info] = nrWaveformGenerator(cfg);
        native_fs = info.ResourceGrids(1).Info.SampleRate;

        % Resample to target fs
        if abs(fs - native_fs) > 1
            [P, Q] = rat(fs / native_fs, 1e-6);
            sig_bb = resample(sig_bb_native(:,1), P, Q);
        else
            sig_bb = sig_bb_native(:,1);
        end

        % Fit to output_len, clipping at a random offset when the frame is
        % longer (see tile_or_clip_random: a fixed offset lands in the
        % deterministic preamble and makes every call identical).
        sig_bb = tile_or_clip_random(sig_bb, output_len);

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('5G NR generation failed (need 5G Toolbox): %s', ME.message);
    end
end


function sig = generate_zigbee_signal(output_len, fs, fc, output_type)
    % GENERATE_ZIGBEE_SIGNAL - IEEE 802.15.4 O-QPSK PHY (Zigbee)
    % Requires: Communications Toolbox (lrwpan package)
    %
    % The 2450 MHz O-QPSK PHY spreads each symbol to 2 Mchip/s with a
    % half-sine pulse, so the waveform occupies roughly 3 MHz regardless of
    % SamplesPerChip -- that is the standard's own bandwidth, not a tunable
    % parameter.  SamplesPerChip only sets the native rate we resample from.
    try
        samples_per_chip = 8;               % native 16 Msps at 2 Mchip/s
        cfg = lrwpanOQPSKConfig('Band', 2450, ...
                                'SamplesPerChip', samples_per_chip);

        % lrwpanWaveformGenerator requires DATA as a column vector whose
        % length is a multiple of 8 (whole PSDU octets).
        psdu_octets = 127;                  % max PSDU for this PHY
        bits = randi([0 1], psdu_octets * 8, 1);
        sig_bb_native = lrwpanWaveformGenerator(bits, cfg);
        native_fs = 2e6 * samples_per_chip;

        % Resample to target fs
        if abs(fs - native_fs) > 1
            [P, Q] = rat(fs / native_fs, 1e-6);
            sig_bb = resample(sig_bb_native(:,1), P, Q);
        else
            sig_bb = sig_bb_native(:,1);
        end

        % Fit to output_len, clipping at a random offset when the frame is
        % longer (see tile_or_clip_random: a fixed offset lands in the
        % deterministic preamble and makes every call identical).
        sig_bb = tile_or_clip_random(sig_bb, output_len);

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('Zigbee generation failed (need Communications Toolbox): %s', ...
              ME.message);
    end
end


function sig_bb = tile_or_clip_random(sig_bb, output_len)
    % TILE_OR_CLIP_RANDOM - fit a generated frame to output_len.
    %
    % Repeats the frame when it is shorter than requested, and otherwise takes a
    % clip at a RANDOM offset rather than always from sample 1.
    %
    % The fixed offset was a silent trap.  Standards frames begin with the same
    % deterministic preamble/sync/reference symbols every time, so clipping from
    % the start produced byte-identical output across calls even when the payload
    % bits were randomised: measured over 60 generated files, 5G NR and Zigbee
    % were 100% correlated to each other while only LoRa (which already took a
    % random offset) varied.  A classifier trained on that is memorising one
    % waveform per class, not learning the modulation.
    if numel(sig_bb) > output_len
        off = randi([0, numel(sig_bb) - output_len]);
        sig_bb = sig_bb(off + (1:output_len));
    else
        while numel(sig_bb) < output_len
            sig_bb = [sig_bb; sig_bb]; %#ok<AGROW>
        end
        sig_bb = sig_bb(1:output_len);
    end
end


function sig = generate_lora_signal(output_len, fs, Tsymb, fc, M, output_type)
    % GENERATE_LORA_SIGNAL - LoRa CSS (chirp spread spectrum) baseband packet
    % No toolbox required -- CSS is generated in closed form below.
    %
    % Adapted from the reference implementation in
    % wireless_waveforms/protocols/lora/lora.m, which follows the Semtech
    % SX127x packet structure.  Two deliberate departures, to match how every
    % other family in this file behaves:
    %
    %   1. Parameters follow this file's conventions rather than LoRa-specific
    %      name/value pairs.  M selects the spreading factor and Tsymb sets the
    %      CSS bandwidth, so the Waveform tab's existing controls drive it.
    %   2. The reference asserts that one complete packet fits in N and
    %      right-zero-pads it.  Here the packet is tiled or clipped to
    %      output_len like WiFi/LTE/5G NR/Zigbee, because the datasets this
    %      toolbox is compared against are continuous mid-capture clips with no
    %      silent tail.  A random clip offset inside the active region gives
    %      per-call variety; when the packet is shorter than output_len it is
    %      repeated rather than padded.
    %
    % M  -> spreading factor directly, clamped to the 7..12 LoRa range (the same
    %       convention Barker uses, where M selects a code length rather than a
    %       constellation order).  M values below 7 clamp to SF 7, the most
    %       common LoRa setting, so the tab's default M = 4 is still valid.
    % Tsymb -> oversampling factor, via sps = round(fs*Tsymb), which is the same
    %       quantity every other family in this file derives from Tsymb.  The CSS
    %       bandwidth is then BW = fs/sps, so sps = 8 at 30.72 Msps gives a
    %       3.84 MHz chirp and sps = 246 gives standard 125 kHz LoRa.
    %
    %       Deriving BW from sps rather than as 1/Tsymb matters: waveform_generator
    %       requires fs*Tsymb to be an integer (a pulse-shaping constraint), and
    %       setting BW = 1/Tsymb directly would force the LoRa bandwidth to be a
    %       divisor of fs, which has nothing to do with CSS and made useful
    %       bandwidths unreachable from the UI.
    try
        SF = max(7, min(12, round(M)));
        chips = 2^SF;

        sps = max(2, round(fs * Tsymb));   % oversampling factor per chip
        BW = fs / sps;                     % CSS bandwidth
        Ns = chips * sps;                  % samples per CSS symbol

        % Packet layout: preamble up-chirps, 2 sync up-chirps, 2.25 symbol SFD
        % of down-chirps, then payload chirps carrying random symbol values.
        preamble_syms = 8;
        n_sync_syms   = 2;
        sfd_full_syms = 2;
        sfd_quarter   = round(0.25 * Ns);

        % Enough payload symbols that the active packet covers output_len even
        % before tiling, so a clip can be taken without crossing the tail.
        min_payload = 8;
        need_syms = ceil((output_len + sfd_quarter) / Ns) + 1;
        n_payload_syms = max(min_payload, need_syms);

        payload_syms = randi([0, chips - 1], n_payload_syms, 1);

        % Build with analytic phase continuity across symbol boundaries.
        phi = 0;
        parts = cell(0);
        for s = 1:preamble_syms
            [c, phi] = lora_chirp(0, chips, Ns, BW, fs, phi, false);
            parts{end+1} = c; %#ok<AGROW>
        end
        for s = 1:n_sync_syms
            [c, phi] = lora_chirp(0, chips, Ns, BW, fs, phi, false);
            parts{end+1} = c; %#ok<AGROW>
        end
        for s = 1:sfd_full_syms
            [c, phi] = lora_chirp(0, chips, Ns, BW, fs, phi, true);
            parts{end+1} = c; %#ok<AGROW>
        end
        [c, phi] = lora_chirp(0, chips, Ns, BW, fs, phi, true, sfd_quarter);
        parts{end+1} = c;
        for s = 1:n_payload_syms
            [c, phi] = lora_chirp(payload_syms(s), chips, Ns, BW, fs, phi, false);
            parts{end+1} = c; %#ok<AGROW>
        end

        sig_bb = vertcat(parts{:});
        sig_bb = sig_bb(:);

        % Unit average power over the packet.
        pwr = mean(abs(sig_bb).^2);
        if pwr > 0
            sig_bb = sig_bb / sqrt(pwr);
        end

        % Clip or tile to output_len.  When the packet is longer, take a random
        % offset so repeated calls are not identical; the whole packet is
        % active signal, so any offset is valid.
        if numel(sig_bb) > output_len
            max_off = numel(sig_bb) - output_len;
            off = randi([0, max_off]);
            sig_bb = sig_bb(off + (1:output_len));
        else
            while numel(sig_bb) < output_len
                sig_bb = [sig_bb; sig_bb]; %#ok<AGROW>
            end
            sig_bb = sig_bb(1:output_len);
        end

        if strcmpi(output_type, 'baseband')
            sig = complex(sig_bb);
        else
            sig = upconvert_to_passband(sig_bb, fs, fc, output_len);
        end
    catch ME
        error('LoRa generation failed: %s', ME.message);
    end
end


function [c, phi_next] = lora_chirp(k, chips, Ns, BW, fs, phi_start, downchirp, cut)
    % LORA_CHIRP - one LoRa CSS symbol with analytically correct wrap.
    %
    % Centred instantaneous normalised frequency:
    %   f(n) = mod(k/chips + n/Ns, 1) - 0.5
    % which wraps at the generally non-integer time tw = (1 - k/chips)*Ns.
    % Integrating f analytically in two segments about that floating wrap point
    % keeps the phase continuous and avoids the per-symbol error that a
    % floor()-quantised wrap would accumulate.  phi_next is the analytic phase
    % at the end boundary, so continuity does not depend on angle() and its
    % 2*pi ambiguity.
    if nargin < 8
        cut = Ns;
    end

    scale = BW / fs;
    tw = (1 - k / chips) * Ns;

    n = (0 : Ns - 1).';
    before = n < tw;
    phase = zeros(Ns, 1);

    dn1 = n(before);
    phase(before) = 2*pi * scale * ((k/chips - 0.5) * dn1 + dn1.^2 / (2*Ns));

    phi_tw = 2*pi * scale * ((k/chips - 0.5) * tw + tw^2 / (2*Ns));

    dn2 = n(~before) - tw;
    phase(~before) = phi_tw + 2*pi * scale * (-0.5 * dn2 + dn2.^2 / (2*Ns));

    if downchirp
        phase = -phase;
    end

    c = exp(1j * (phase(1:cut) + phi_start));

    nb = double(cut);
    if nb <= tw
        phi_boundary = 2*pi * scale * ((k/chips - 0.5) * nb + nb^2 / (2*Ns));
    else
        dnb = nb - tw;
        phi_boundary = phi_tw + 2*pi * scale * (-0.5 * dnb + dnb^2 / (2*Ns));
    end
    if downchirp
        phi_boundary = -phi_boundary;
    end
    phi_next = phi_boundary + phi_start;
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
