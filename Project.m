% Load the data
load('825_2_PD_REST.mat');  % Adjust the path if necessary
data = EEG.data;  % Assuming EEG is the main structure and data is the field with EEG signals
for i=1:10
    % Select the 26th channel
channel_1 = data(i, :);

% Sample rate (Hz)
Fs = 500;

% Time vector for plotting
t = (0:length(channel_1)-1) / Fs;

% Plot the original signal in time domain
figure;
subplot(5,5,1);
plot(t, channel_1 / max(abs(channel_1))); % Normalize the original signal
title('Original Signal - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the original signal in frequency domain
n = length(channel_1);
original_fft = fft(channel_1);
f = Fs * (0:n-1) / n;  % Frequency range (Hz)
magnitude = abs(original_fft) / n;  % Magnitude of the FFT

subplot(5,5,2);
plot(f(1:floor(n/2)), magnitude(1:floor(n/2)) / max(magnitude(1:floor(n/2))));  % Normalize the magnitude
title('Original Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);  % Limit the plot to frequencies up to Nyquist frequency
grid on;

% Design a notch filter to remove 55 Hz noise
f0 = 55;  % Frequency to be removed (Hz)
Q = 100;  % Quality factor
wo = f0 / (Fs / 2);  % Normalized frequency
bw = wo / Q;
[b, a] = iirnotch(wo, bw);

b=double (b);
a=double(a);
channel_1=double(channel_1);
% Apply the notch filter to the signal
filtered_channel_1 = filtfilt(b, a, channel_1);

% Normalize the notch filtered signal
filtered_channel_1 = filtered_channel_1 / max(abs(filtered_channel_1));

% Plot the filtered signal in time domain
subplot(5,5,3);
plot(t, filtered_channel_1);
title('notch Filtered Signal - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the filtered signal in frequency domain
filtered_fft = fft(filtered_channel_1);
magnitude_filtered = abs(filtered_fft) / n;

subplot(5,5,4);
plot(f(1:floor(n/2)), magnitude_filtered(1:floor(n/2)) / max(magnitude_filtered(1:floor(n/2)))); % Normalize the magnitude
title('notch Filtered Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;

% Design low-pass filter for frequencies below 9 Hz
fc_lp = 30; % Cutoff frequency for low-pass filter (Hz)
[b_lp, a_lp] = butter(4, fc_lp / (Fs/2), 'low'); % Design low-pass filter

% Apply low-pass filter to the notch filtered signal
filtered_channel_lp = filtfilt(b_lp, a_lp, filtered_channel_1);

% Normalize the low-pass filtered signal
filtered_channel_lp = filtered_channel_lp / max(abs(filtered_channel_lp));

% Plot low-pass filtered signal in time and frequency domain
subplot(5,5,5);
plot(t, filtered_channel_lp);
title('Low-pass Signal- Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,6);
filtered_fft_lp = fft(filtered_channel_lp);
magnitude_filtered_lp = abs(filtered_fft_lp) / max(abs(filtered_fft_lp)); % Normalize the magnitude
plot(f(1:floor(n/2)), magnitude_filtered_lp(1:floor(n/2)));
title('Low-pass Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
ylim([0 1.2]); % Adjusted y-axis limit for better visualization
grid on;

% Design high-pass filter for frequencies above 20 Hz
fc_hp = 0.5; % Cutoff frequency for high-pass filter (Hz)
[b_hp, a_hp] = butter(4, fc_hp / (Fs/2), 'high'); % Design high-pass filter

% Apply high-pass filter to the notch filtered signal
filtered_channel_hp = filtfilt(b_hp, a_hp, filtered_channel_lp);

% Normalize the high-pass filtered signal
filtered_channel_hp = filtered_channel_hp / max(abs(filtered_channel_hp));

% Plot high-pass filtered signal in time and frequency domain
subplot(5,5,7);
plot(t, filtered_channel_hp);
title('High-pass Signal- Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,8);
filtered_fft_hp = fft(filtered_channel_hp);
magnitude_filtered_hp = abs(filtered_fft_hp) / n;
plot(f(1:floor(n/2)), magnitude_filtered_hp(1:floor(n/2)) / max(magnitude_filtered_hp(1:floor(n/2)))); % Normalize the magnitude
title('High-pass Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for delta band (0.5 - 4 Hz)
fc_delta = [0.5 4]; % Cutoff frequencies for delta band (Hz)
[b_bp_delta, a_bp_delta] = butter(4, fc_delta / (Fs/2), 'bandpass'); % Design band-pass filter for delta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_delta = filtfilt(b_bp_delta, a_bp_delta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_delta = filtered_channel_bp_delta / max(abs(filtered_channel_bp_delta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,9);
plot(t, filtered_channel_bp_delta);
title('Signal (Delta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,10);
filtered_fft_bp_delta = fft(filtered_channel_bp_delta);
magnitude_filtered_bp_delta = abs(filtered_fft_bp_delta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_delta(1:floor(n/2)) / max(magnitude_filtered_bp_delta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Delta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for theta band (4 - 7 Hz)
fc_theta = [4 7]; % Cutoff frequencies for theta band (Hz)
[b_bp_theta, a_bp_theta] = butter(4, fc_theta / (Fs/2), 'bandpass'); % Design band-pass filter for theta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_theta = filtfilt(b_bp_theta, a_bp_theta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_theta = filtered_channel_bp_theta / max(abs(filtered_channel_bp_theta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,11);
plot(t, filtered_channel_bp_theta);
title('Signal (Theta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,12);
filtered_fft_bp_theta = fft(filtered_channel_bp_theta);
magnitude_filtered_bp_theta = abs(filtered_fft_bp_theta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_theta(1:floor(n/2)) / max(magnitude_filtered_bp_theta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Theta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for alpha band (8 - 12 Hz)
fc_alpha = [8 12]; % Cutoff frequencies for alpha band (Hz)
[b_bp_alpha, a_bp_alpha] = butter(4, fc_alpha / (Fs/2), 'bandpass'); % Design band-pass filter for alpha band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_alpha = filtfilt(b_bp_alpha, a_bp_alpha, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_alpha = filtered_channel_bp_alpha / max(abs(filtered_channel_bp_alpha));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,13);
plot(t, filtered_channel_bp_alpha);
title('Signal (Alpha Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,14);
filtered_fft_bp_alpha = fft(filtered_channel_bp_alpha);
magnitude_filtered_bp_alpha = abs(filtered_fft_bp_alpha) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_alpha(1:floor(n/2)) / max(magnitude_filtered_bp_alpha(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Alpha Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for beta band (13 - 30 Hz)
fc_beta = [13 30]; % Cutoff frequencies for beta band (Hz)
[b_bp_beta, a_bp_beta] = butter(4, fc_beta / (Fs/2), 'bandpass'); % Design band-pass filter for beta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_beta = filtfilt(b_bp_beta, a_bp_beta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_beta = filtered_channel_bp_beta / max(abs(filtered_channel_bp_beta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,15);
plot(t, filtered_channel_bp_beta);
title('Signal (Beta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,16);
filtered_fft_bp_beta = fft(filtered_channel_bp_beta);
magnitude_filtered_bp_beta = abs(filtered_fft_bp_beta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_beta(1:floor(n/2)) / max(magnitude_filtered_bp_beta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Beta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for sigma band (12 - 16 Hz)
fc_sigma = [12 16]; % Cutoff frequencies for sigma band (Hz)
[b_bp_sigma, a_bp_sigma] = butter(4, fc_sigma / (Fs/2), 'bandpass'); % Design band-pass filter for sigma band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_sigma = filtfilt(b_bp_sigma, a_bp_sigma, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_sigma = filtered_channel_bp_sigma / max(abs(filtered_channel_bp_sigma));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,17);
plot(t, filtered_channel_bp_sigma);
title('Signal (Sigma Band) - Time Domain');% Load the data
load('825_2_PD_REST.mat');  % Adjust the path if necessary
data = EEG.data;  % Assuming EEG is the main structure and data is the field with EEG signals
for i=1:10
    % Select the 26th channel
channel_1 = data(i, :);

% Sample rate (Hz)
Fs = 500;

% Time vector for plotting
t = (0:length(channel_1)-1) / Fs;

% Plot the original signal in time domain
figure;
subplot(5,5,1);
plot(t, channel_1 / max(abs(channel_1))); % Normalize the original signal
title('Original Signal - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the original signal in frequency domain
n = length(channel_1);
original_fft = fft(channel_1);
f = Fs * (0:n-1) / n;  % Frequency range (Hz)
magnitude = abs(original_fft) / n;  % Magnitude of the FFT

subplot(5,5,2);
plot(f(1:floor(n/2)), magnitude(1:floor(n/2)) / max(magnitude(1:floor(n/2))));  % Normalize the magnitude
title('Original Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);  % Limit the plot to frequencies up to Nyquist frequency
grid on;

% Design a notch filter to remove 55 Hz noise
f0 = 55;  % Frequency to be removed (Hz)
Q = 100;  % Quality factor
wo = f0 / (Fs / 2);  % Normalized frequency
bw = wo / Q;
[b, a] = iirnotch(wo, bw);

b=double (b);
a=double(a);
channel_1=double(channel_1);
% Apply the notch filter to the signal
filtered_channel_1 = filtfilt(b, a, channel_1);

% Normalize the notch filtered signal
filtered_channel_1 = filtered_channel_1 / max(abs(filtered_channel_1));

% Plot the filtered signal in time domain
subplot(5,5,3);
plot(t, filtered_channel_1);
title('notch Filtered Signal - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the filtered signal in frequency domain
filtered_fft = fft(filtered_channel_1);
magnitude_filtered = abs(filtered_fft) / n;

subplot(5,5,4);
plot(f(1:floor(n/2)), magnitude_filtered(1:floor(n/2)) / max(magnitude_filtered(1:floor(n/2)))); % Normalize the magnitude
title('notch Filtered Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;

% Design low-pass filter for frequencies below 9 Hz
fc_lp = 30; % Cutoff frequency for low-pass filter (Hz)
[b_lp, a_lp] = butter(4, fc_lp / (Fs/2), 'low'); % Design low-pass filter

% Apply low-pass filter to the notch filtered signal
filtered_channel_lp = filtfilt(b_lp, a_lp, filtered_channel_1);

% Normalize the low-pass filtered signal
filtered_channel_lp = filtered_channel_lp / max(abs(filtered_channel_lp));

% Plot low-pass filtered signal in time and frequency domain
subplot(5,5,5);
plot(t, filtered_channel_lp);
title('Low-pass Signal- Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,6);
filtered_fft_lp = fft(filtered_channel_lp);
magnitude_filtered_lp = abs(filtered_fft_lp) / max(abs(filtered_fft_lp)); % Normalize the magnitude
plot(f(1:floor(n/2)), magnitude_filtered_lp(1:floor(n/2)));
title('Low-pass Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
ylim([0 1.2]); % Adjusted y-axis limit for better visualization
grid on;

% Design high-pass filter for frequencies above 20 Hz
fc_hp = 0.5; % Cutoff frequency for high-pass filter (Hz)
[b_hp, a_hp] = butter(4, fc_hp / (Fs/2), 'high'); % Design high-pass filter

% Apply high-pass filter to the notch filtered signal
filtered_channel_hp = filtfilt(b_hp, a_hp, filtered_channel_lp);

% Normalize the high-pass filtered signal
filtered_channel_hp = filtered_channel_hp / max(abs(filtered_channel_hp));

% Plot high-pass filtered signal in time and frequency domain
subplot(5,5,7);
plot(t, filtered_channel_hp);
title('High-pass Signal- Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,8);
filtered_fft_hp = fft(filtered_channel_hp);
magnitude_filtered_hp = abs(filtered_fft_hp) / n;
plot(f(1:floor(n/2)), magnitude_filtered_hp(1:floor(n/2)) / max(magnitude_filtered_hp(1:floor(n/2)))); % Normalize the magnitude
title('High-pass Signal - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for delta band (0.5 - 4 Hz)
fc_delta = [0.5 4]; % Cutoff frequencies for delta band (Hz)
[b_bp_delta, a_bp_delta] = butter(4, fc_delta / (Fs/2), 'bandpass'); % Design band-pass filter for delta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_delta = filtfilt(b_bp_delta, a_bp_delta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_delta = filtered_channel_bp_delta / max(abs(filtered_channel_bp_delta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,9);
plot(t, filtered_channel_bp_delta);
title('Signal (Delta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,10);
filtered_fft_bp_delta = fft(filtered_channel_bp_delta);
magnitude_filtered_bp_delta = abs(filtered_fft_bp_delta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_delta(1:floor(n/2)) / max(magnitude_filtered_bp_delta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Delta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for theta band (4 - 7 Hz)
fc_theta = [4 7]; % Cutoff frequencies for theta band (Hz)
[b_bp_theta, a_bp_theta] = butter(4, fc_theta / (Fs/2), 'bandpass'); % Design band-pass filter for theta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_theta = filtfilt(b_bp_theta, a_bp_theta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_theta = filtered_channel_bp_theta / max(abs(filtered_channel_bp_theta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,11);
plot(t, filtered_channel_bp_theta);
title('Signal (Theta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,12);
filtered_fft_bp_theta = fft(filtered_channel_bp_theta);
magnitude_filtered_bp_theta = abs(filtered_fft_bp_theta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_theta(1:floor(n/2)) / max(magnitude_filtered_bp_theta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Theta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for alpha band (8 - 12 Hz)
fc_alpha = [8 12]; % Cutoff frequencies for alpha band (Hz)
[b_bp_alpha, a_bp_alpha] = butter(4, fc_alpha / (Fs/2), 'bandpass'); % Design band-pass filter for alpha band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_alpha = filtfilt(b_bp_alpha, a_bp_alpha, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_alpha = filtered_channel_bp_alpha / max(abs(filtered_channel_bp_alpha));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,13);
plot(t, filtered_channel_bp_alpha);
title('Signal (Alpha Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,14);
filtered_fft_bp_alpha = fft(filtered_channel_bp_alpha);
magnitude_filtered_bp_alpha = abs(filtered_fft_bp_alpha) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_alpha(1:floor(n/2)) / max(magnitude_filtered_bp_alpha(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Alpha Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for beta band (13 - 30 Hz)
fc_beta = [13 30]; % Cutoff frequencies for beta band (Hz)
[b_bp_beta, a_bp_beta] = butter(4, fc_beta / (Fs/2), 'bandpass'); % Design band-pass filter for beta band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_beta = filtfilt(b_bp_beta, a_bp_beta, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_beta = filtered_channel_bp_beta / max(abs(filtered_channel_bp_beta));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,15);
plot(t, filtered_channel_bp_beta);
title('Signal (Beta Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,16);
filtered_fft_bp_beta = fft(filtered_channel_bp_beta);
magnitude_filtered_bp_beta = abs(filtered_fft_bp_beta) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_beta(1:floor(n/2)) / max(magnitude_filtered_bp_beta(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Beta Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;

% Design band-pass filter for sigma band (12 - 16 Hz)
fc_sigma = [12 16]; % Cutoff frequencies for sigma band (Hz)
[b_bp_sigma, a_bp_sigma] = butter(4, fc_sigma / (Fs/2), 'bandpass'); % Design band-pass filter for sigma band

% Apply band-pass filter to the notch filtered signal
filtered_channel_bp_sigma = filtfilt(b_bp_sigma, a_bp_sigma, filtered_channel_hp);

% Normalize the filtered signal
filtered_channel_bp_sigma = filtered_channel_bp_sigma / max(abs(filtered_channel_bp_sigma));

% Plot band-pass filtered signal in time and frequency domain
subplot(5,5,17);
plot(t, filtered_channel_bp_sigma);
title('Signal (Sigma Band) - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,18);
filtered_fft_bp_sigma = fft(filtered_channel_bp_sigma);
magnitude_filtered_bp_sigma = abs(filtered_fft_bp_sigma) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_sigma(1:floor(n/2)) / max(magnitude_filtered_bp_sigma(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Sigma Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;


end
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,5,18);
filtered_fft_bp_sigma = fft(filtered_channel_bp_sigma);
magnitude_filtered_bp_sigma = abs(filtered_fft_bp_sigma) / n;
plot(f(1:floor(n/2)), magnitude_filtered_bp_sigma(1:floor(n/2)) / max(magnitude_filtered_bp_sigma(1:floor(n/2)))); % Normalize the magnitude
title('Signal (Sigma Band) - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Magnitude (Normalized)');
xlim([0 Fs/2]);
grid on;


end