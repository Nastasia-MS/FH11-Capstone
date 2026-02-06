function pam_pb = pam_gui(output_len, fs, Tsymb, fc, M, Var)
    % data_samp_count      % Number of data samples to generate
    % save_path            % Path where .npy is saved
    % output_len           % Length of each saved output vector
    % fs                   % Sample Rate (Hz)
    % Tsymb                % Symbol Period (s)
    % fc                   % Carrier Frequency (Hz)
    % M                    % PAM order (even integer >= 2)
    % Var                  % Target symbol variance (before pulse shaping)

    % ----- Defaults -----
    % if nargin < 1 || isempty(data_samp_count), data_samp_count = 10; end
    % if nargin < 3 || isempty(output_len),      output_len = 2000;   end
    % if nargin < 4 || isempty(fs),              fs = 48e3;           end
    % if nargin < 5 || isempty(Tsymb),           Tsymb = 1e-3;        end
    % if nargin < 6 || isempty(fc),              fc = 6e3;            end
    % if nargin < 7 || isempty(M),               M = 4;               end
    % if nargin < 8 || isempty(Var),             Var = 1;             end


    % ----- Integer samples-per-symbol & constraints -----
    sps = fs * Tsymb;
    assert(abs(sps - round(sps)) < 1e-9, 'fs*Tsymb must be an integer.');
    sps = round(sps);                % lock to integer
    assert(mod(output_len, sps) == 0, 'output_len/(fs*Tsymb) must be an integer.');
    assert(fc < fs/2, 'Carrier fc must be < fs/2 to avoid aliasing.');
    assert(mod(M,2) == 0 && M >= 2, 'M must be an even integer >= 2.');

    Nsym = output_len / sps;

    % PAM levels scaled to variance Var: E[a^2] = (M^2-1)/3 * step^2
    levels = (-(M-1):2:(M-1)) * sqrt(3*Var/(M^2 - 1));  % 1xM

    % ----- Symbols -----
    idx    = randi(M, Nsym, 1);      % Nsym x 1
    pam_sym = levels(idx);           % Nsym x 1, column

    % ----- Rectangular pulse shaping -----
    pam_bb = repelem(pam_sym, sps, 1);   % (Nsym*sps) x 1

    L = numel(pam_bb);
    t = (0:L-1).' / fs;                 % column
    carrier = cos(2*pi*fc*t);           % L x 1 column

    % Safety: force columns
    pam_bb = pam_bb(:);
    carrier = carrier(:);

    % ----- Upconvert -----
    pam_pb = pam_bb .* carrier;         % L x 1

    % ----- Save -----
    % filename = fullfile(save_path, sprintf('pam%u_%d.npy', M, n));
    % writeNPY(pam_pb, filename);
end
