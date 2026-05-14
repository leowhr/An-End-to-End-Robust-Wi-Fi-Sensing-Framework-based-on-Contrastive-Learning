%% run_prepare_widar3_csi.m
% Build unified Widar3.0 CSI tensors from raw .dat files.
% Output per sample: CSI_result with shape [1450, 90, 6] (complex).

clear;
clc;

%% User config
project_root = fileparts(mfilename('fullpath'));
input_dir = fullfile(project_root, 'CSI');
output_dir = fullfile(project_root, 'CSI_pro');

rx_count = 6;          % number of receivers
rx_ant_count = 3;      % antennas per receiver
subcarrier_count = 30; % subcarriers per antenna
target_T = 1450;       % fixed temporal length
resume_from_existing = true; % true: skip samples already saved in output_dir
progress_print_interval = 100; % print progress every N samples

%% Prepare paths
addpath(fullfile(project_root, 'csi_tool_box'));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Scan all .dat files and group by sample key
% Expected filename format:
% user_id-label-torso_loc-face_dir-rep_num-receiver_id-roomx.dat
dat_files = dir(fullfile(input_dir, '*.dat'));
if isempty(dat_files)
    error('No .dat files found in %s', input_dir);
end

sample_rx_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
sample_outname_map = containers.Map('KeyType', 'char', 'ValueType', 'char');
for i = 1:numel(dat_files)
    name = dat_files(i).name;
    tok = regexp(name, '^(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(r\d+)-(room\d+)\.dat$', 'tokens', 'once');
    if isempty(tok)
        continue;
    end

    user_id = tok{1};
    label = tok{2};
    torso_loc = tok{3};
    face_dir = tok{4};
    rep_num = tok{5};
    receiver_id = tok{6};
    room_id = tok{7};

    rx_tok = regexp(receiver_id, '^r(\d+)$', 'tokens', 'once');
    if isempty(rx_tok)
        continue;
    end
    rx_id = str2double(rx_tok{1});

    sample_key = sprintf('%s-%s-%s-%s-%s-%s', user_id, label, torso_loc, face_dir, rep_num, room_id);
    out_name = sprintf('%s-%s-%s-%s-%s-6-link-%s.mat', user_id, label, torso_loc, face_dir, rep_num, room_id);

    if ~isKey(sample_rx_map, sample_key)
        sample_rx_map(sample_key) = nan(1, rx_count);
        sample_outname_map(sample_key) = out_name;
    end

    rx_list = sample_rx_map(sample_key);
    if rx_id >= 1 && rx_id <= rx_count
        rx_list(rx_id) = i;
        sample_rx_map(sample_key) = rx_list;
    end
end

all_samples = sort(sample_rx_map.keys);
total_samples = numel(all_samples);
loop_timer = tic;

saved_new_files = {};
skipped_existing_files = {};
failed_files = {};
failed_msgs = {};

%% Process each grouped sample
for s = 1:numel(all_samples)
    sample_key = all_samples{s};
    rx_file_idx = sample_rx_map(sample_key);
    out_name = sample_outname_map(sample_key);

    elapsed_sec = toc(loop_timer);
    if s > 1
        eta_sec = elapsed_sec / (s - 1) * (total_samples - s + 1);
    else
        eta_sec = NaN;
    end
    if s == 1 || s == total_samples || mod(s, progress_print_interval) == 0
        fprintf('[Progress] %d/%d (%.2f%%%%), ETA: %s\n', s, total_samples, 100 * s / total_samples, format_eta(eta_sec));
    end

    if resume_from_existing && has_existing_output(output_dir, out_name)
        skipped_existing_files{end+1} = out_name;
        continue;
    end

    % [T, 90, 6] complex tensor
    CSI_result = complex(zeros(target_T, rx_ant_count * subcarrier_count, rx_count));

    for rx = 1:rx_count
        if isnan(rx_file_idx(rx))
            continue;
        end

        dat_name = dat_files(rx_file_idx(rx)).name;
        dat_path = fullfile(input_dir, dat_name);

        try
            [csi_raw, ~] = csi_get_all(dat_path); % [N, 90], complex

            if isempty(csi_raw)
                continue;
            end

            csi_proc = conj_mult_and_norm(csi_raw, rx_ant_count, subcarrier_count);
            csi_aligned = temporal_align_complex(csi_proc, target_T);
            CSI_result(:, :, rx) = csi_aligned;
        catch
        end
    end

    save_path = fullfile(output_dir, out_name);
    [save_ok, final_save_path, save_err_msg] = save_csi_result(save_path, CSI_result);
    if save_ok
        [~, fn, ext] = fileparts(final_save_path);
        saved_new_files{end+1} = [fn, ext];
    else
        failed_files{end+1} = out_name;
        failed_msgs{end+1} = save_err_msg;
    end
end

write_processing_summary(output_dir, numel(all_samples), saved_new_files, skipped_existing_files, failed_files, failed_msgs);

%% ===== Local functions =====
function csi_out = conj_mult_and_norm(csi_in, rx_ant_count, subcarrier_count)
% csi_in: [N, 90] complex
% csi_out: [N, 90] complex after conjugate multiplication and normalization

feat_per_rx = rx_ant_count * subcarrier_count;
if size(csi_in, 2) ~= feat_per_rx
    error('Expected %d features, got %d', feat_per_rx, size(csi_in, 2));
end

% Antenna-pair selection from Widar code: maximize mean/var ratio
amp = abs(csi_in);
csi_mean = mean(amp, 1);
csi_var = sqrt(var(amp, 0, 1));
csi_ratio = csi_mean ./ max(csi_var, eps);
ratio_mat = reshape(csi_ratio, [subcarrier_count, rx_ant_count]);
[~, ref_ant_idx] = max(mean(ratio_mat, 1));

% Build reference CSI per subcarrier and repeat to 90 dims
ref_range = (ref_ant_idx - 1) * subcarrier_count + 1 : ref_ant_idx * subcarrier_count;
csi_ref = repmat(csi_in(:, ref_range), 1, rx_ant_count);

% Conjugate multiplication for phase-offset correction
csi_cm = csi_in .* conj(csi_ref);

% Normalize amplitude and phase separately (per feature over time)
cm_amp = abs(csi_cm);
cm_phase = unwrap(angle(csi_cm), [], 1);

amp_scale = mean(cm_amp, 1);
cm_amp_norm = cm_amp ./ max(amp_scale, eps);

phase_mu = mean(cm_phase, 1);
phase_sigma = std(cm_phase, 0, 1);
cm_phase_norm = (cm_phase - phase_mu) ./ max(phase_sigma, eps);

csi_out = cm_amp_norm .* exp(1j * cm_phase_norm);
end

function x_aligned = temporal_align_complex(x, target_T)
% x: [N, C] complex -> x_aligned: [target_T, C] complex

N = size(x, 1);
C = size(x, 2);
if N <= 1
    x_aligned = repmat(x(1, :), target_T, 1);
    return;
end

old_t = linspace(0, 1, N);
new_t = linspace(0, 1, target_T);

x_real = real(x);
x_imag = imag(x);

x_real_aligned = interp1(old_t, x_real, new_t, 'linear', 'extrap');
x_imag_aligned = interp1(old_t, x_imag, new_t, 'linear', 'extrap');

x_aligned = complex(x_real_aligned, x_imag_aligned);
x_aligned = reshape(x_aligned, [target_T, C]);
end

function [ok, final_path, err_msg] = save_csi_result(save_path, CSI_result)
% Try to save to the requested filename first.
ok = false;
final_path = save_path;
err_msg = '';

try
    save(save_path, 'CSI_result', '-v7');
    ok = true;
    return;
catch err
    err_msg = err.message;
end

% If the target path is blocked, write to an alternative file and continue.
[folder, base, ext] = fileparts(save_path);
for idx = 1:20
    alt_path = fullfile(folder, sprintf('%s-savefail%d%s', base, idx, ext));
    try
        save(alt_path, 'CSI_result', '-v7');
        ok = true;
        final_path = alt_path;
        return;
    catch err
        err_msg = err.message;
    end
end
end

function exists_flag = has_existing_output(output_dir, out_name)
exists_flag = false;

exact_path = fullfile(output_dir, out_name);
if exist(exact_path, 'file') == 2
    exists_flag = true;
    return;
end

[~, base, ~] = fileparts(out_name);
alt_list = dir(fullfile(output_dir, [base, '-savefail*.mat']));
if ~isempty(alt_list)
    exists_flag = true;
end
end

function write_processing_summary(output_dir, total_samples, saved_new_files, skipped_existing_files, failed_files, failed_msgs)
summary_path = fullfile(output_dir, 'summary.txt');
success_count = numel(saved_new_files) + numel(skipped_existing_files);

fid = fopen(summary_path, 'wt');
if fid < 0
    return;
end

fprintf(fid, 'Widar3.0 Processing Summary\n');
fprintf(fid, '===========================\n');
fprintf(fid, 'Total grouped samples: %d\n', total_samples);
fprintf(fid, 'Successful files (saved + skipped existing): %d\n', success_count);
fprintf(fid, 'Saved new files: %d\n', numel(saved_new_files));
fprintf(fid, 'Skipped existing files: %d\n', numel(skipped_existing_files));
fprintf(fid, 'Failed files: %d\n\n', numel(failed_files));

% fprintf(fid, 'Saved New Files\n');
% fprintf(fid, '--------------\n');
% for i = 1:numel(saved_new_files)
%     fprintf(fid, '%s\n', saved_new_files{i});
% end

% fprintf(fid, '\nSkipped Existing Files\n');
% fprintf(fid, '---------------------\n');
% for i = 1:numel(skipped_existing_files)
%     fprintf(fid, '%s\n', skipped_existing_files{i});
% end

fprintf(fid, '\nFailed Files\n');
fprintf(fid, '------------\n');
for i = 1:numel(failed_files)
    fprintf(fid, '%s\t%s\n', failed_files{i}, failed_msgs{i});
end

fclose(fid);
end

function eta_str = format_eta(eta_sec)
if isnan(eta_sec) || isinf(eta_sec)
    eta_str = 'estimating...';
    return;
end

eta_sec = max(0, round(eta_sec));
hh = floor(eta_sec / 3600);
mm = floor(mod(eta_sec, 3600) / 60);
ss = mod(eta_sec, 60);
eta_str = sprintf('%02d:%02d:%02d', hh, mm, ss);
end
