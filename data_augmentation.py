import numpy as np
from scipy.signal import (
    butter,
    istft,
    lfilter,
    stft,
)


class ECGAugmentation:
    """
    Controlled augmentation methods for one-dimensional ECG segments.

    Input signals are represented as a two-dimensional array with shape:

        (number_of_segments, signal_length)

    A one-dimensional signal is accepted and converted to a single-item
    batch. Every augmentation returns a float32 array and preserves the
    number and length of the input segments.
    """

    SUPPORTED_METHODS = (
        "gaussian",
        "amplitude",
        "timeshift",
        "baseline_wander",
        "time_warp",
        "cutout",
        "emg_noise",
        "istft_augment",
    )

    def __init__(self, seed=None):
        """
        Create an ECG augmentation utility.

        Args:
            seed (int, optional):
                Local random seed. When omitted, NumPy's global random
                state is used, allowing the framework-level random seed
                to control augmentation reproducibility.
        """
        if seed is not None:
            if isinstance(
                seed,
                (
                    bool,
                    np.bool_,
                ),
            ) or not isinstance(
                seed,
                (
                    int,
                    np.integer,
                ),
            ):
                raise ValueError(
                    "seed must be an integer or None."
                )

            seed = int(seed)

        self.seed = seed

        self._rng = (
            np.random.RandomState(seed)
            if seed is not None
            else None
        )

    def _get_rng(self):
        """
        Return either the local or global NumPy random generator.
        """
        if self._rng is not None:
            return self._rng

        return np.random

    def _ensure_2d(self, signals):
        """
        Convert input signals to a validated float32 batch.
        """
        signals = np.asarray(
            signals,
            dtype=np.float32,
        )

        if signals.ndim == 1:
            signals = signals[
                np.newaxis,
                :
            ]

        if signals.ndim != 2:
            raise ValueError(
                "ECG augmentation expects a one-dimensional "
                "signal or a two-dimensional signal batch."
            )

        return signals

    @staticmethod
    def _validate_non_negative(
        value,
        parameter_name,
    ):
        """
        Validate a finite numeric value greater than or equal to zero.
        """
        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError(
                f"{parameter_name} must be a non-negative number."
            )

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{parameter_name} must be a non-negative number."
            ) from error

        if (
            not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{parameter_name} must be a non-negative number."
            )

        return value

    @staticmethod
    def _validate_positive_integer(
        value,
        parameter_name,
        allow_zero=False,
    ):
        """
        Validate an integer parameter.
        """
        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ) or not isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):
            requirement = (
                "a non-negative integer"
                if allow_zero
                else "a positive integer"
            )

            raise ValueError(
                f"{parameter_name} must be {requirement}."
            )

        value = int(value)

        minimum_value = (
            0
            if allow_zero
            else 1
        )

        if value < minimum_value:
            requirement = (
                "a non-negative integer"
                if allow_zero
                else "a positive integer"
            )

            raise ValueError(
                f"{parameter_name} must be {requirement}."
            )

        return value

    def apply(
        self,
        signals,
        method,
        **kwargs,
    ):
        """
        Apply one named augmentation operation.

        Args:
            signals (np.ndarray):
                ECG signal or signal batch.
            method (str):
                Augmentation method name.
            **kwargs:
                Method-specific parameters.

        Returns:
            np.ndarray:
                Augmented float32 ECG batch.
        """
        if not isinstance(
            method,
            str,
        ):
            raise ValueError(
                "method must be a string."
            )

        normalized_method = (
            method
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "time_shift": "timeshift",
            "baseline": "baseline_wander",
            "warp": "time_warp",
            "emg": "emg_noise",
            "istft": "istft_augment",
        }

        normalized_method = aliases.get(
            normalized_method,
            normalized_method,
        )

        methods = {
            "gaussian": self.gaussian,
            "amplitude": self.amplitude,
            "timeshift": self.timeshift,
            "baseline_wander": self.baseline_wander,
            "time_warp": self.time_warp,
            "cutout": self.cutout,
            "emg_noise": self.emg_noise,
            "istft_augment": self.istft_augment,
        }

        if normalized_method not in methods:
            raise ValueError(
                "Unknown augmentation method "
                f"{method!r}. Supported methods are: "
                f"{', '.join(self.SUPPORTED_METHODS)}."
            )

        return methods[
            normalized_method
        ](
            signals,
            **kwargs,
        )

    def gaussian(
        self,
        beats,
        std=0.01,
        relative=True,
    ):
        """
        Add zero-mean Gaussian sensor noise.
        """
        beats = self._ensure_2d(
            beats
        )

        std = self._validate_non_negative(
            std,
            "std",
        )

        if not isinstance(
            relative,
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError(
                "relative must be Boolean."
            )

        if beats.size == 0:
            return beats.copy()

        rng = self._get_rng()

        if relative:
            signal_std = (
                beats.std(
                    axis=1,
                    keepdims=True,
                )
                + 1e-8
            )

            noise_scale = (
                std * signal_std
            )
        else:
            noise_scale = std

        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=beats.shape,
        ).astype(
            np.float32
        )

        augmented = (
            beats
            + noise * noise_scale
        )

        return augmented.astype(
            np.float32,
            copy=False,
        )

    def amplitude(
        self,
        beats,
        scale_range=(0.9, 1.1),
    ):
        """
        Apply random global amplitude scaling.
        """
        beats = self._ensure_2d(
            beats
        )

        if not isinstance(
            scale_range,
            (
                tuple,
                list,
            ),
        ) or len(scale_range) != 2:
            raise ValueError(
                "scale_range must contain exactly "
                "two numeric values."
            )

        try:
            low = float(
                scale_range[0]
            )

            high = float(
                scale_range[1]
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "scale_range must contain exactly "
                "two numeric values."
            ) from error

        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or low <= 0.0
            or high <= 0.0
            or low > high
        ):
            raise ValueError(
                "scale_range must satisfy "
                "0 < low <= high."
            )

        if beats.size == 0:
            return beats.copy()

        rng = self._get_rng()

        factors = rng.uniform(
            low,
            high,
            size=(
                beats.shape[0],
                1,
            ),
        ).astype(
            np.float32
        )

        return (
            beats * factors
        ).astype(
            np.float32,
            copy=False,
        )

    def timeshift(
        self,
        beats,
        max_shift=10,
    ):
        """
        Apply an independently sampled circular shift to each segment.
        """
        beats = self._ensure_2d(
            beats
        )

        max_shift = (
            self._validate_positive_integer(
                max_shift,
                "max_shift",
                allow_zero=True,
            )
        )

        if beats.size == 0:
            return beats.copy()

        if max_shift == 0:
            return beats.copy()

        rng = self._get_rng()

        augmented = np.empty_like(
            beats
        )

        for sample_index in range(
            len(beats)
        ):
            shift = rng.randint(
                -max_shift,
                max_shift + 1,
            )

            augmented[
                sample_index
            ] = np.roll(
                beats[sample_index],
                shift,
            )

        return augmented.astype(
            np.float32,
            copy=False,
        )

    def baseline_wander(
        self,
        beats,
        freq=0.3,
        amp=0.1,
        fs=250,
    ):
        """
        Add low-frequency sinusoidal baseline drift.
        """
        beats = self._ensure_2d(
            beats
        )

        freq = self._validate_non_negative(
            freq,
            "freq",
        )

        amp = self._validate_non_negative(
            amp,
            "amp",
        )

        fs = self._validate_non_negative(
            fs,
            "fs",
        )

        if fs <= 0.0:
            raise ValueError(
                "fs must be greater than zero."
            )

        if beats.size == 0:
            return beats.copy()

        rng = self._get_rng()

        time_axis = (
            np.arange(
                beats.shape[1],
                dtype=np.float32,
            )
            / fs
        )

        phase = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(
                beats.shape[0],
                1,
            ),
        )

        drift = (
            amp
            * np.sin(
                2.0
                * np.pi
                * freq
                * time_axis
                + phase
            )
        )

        return (
            beats + drift
        ).astype(
            np.float32,
            copy=False,
        )

    def time_warp(
        self,
        beats,
        sigma=0.2,
        num_knots=4,
    ):
        """
        Apply a smooth monotonic temporal deformation.

        Positive random interval multipliers are used so the warped time
        coordinates always remain strictly increasing.
        """
        beats = self._ensure_2d(
            beats
        )

        sigma = self._validate_non_negative(
            sigma,
            "sigma",
        )

        num_knots = (
            self._validate_positive_integer(
                num_knots,
                "num_knots",
            )
        )

        if beats.size == 0:
            return beats.copy()

        signal_length = beats.shape[1]

        if signal_length < 2:
            return beats.copy()

        rng = self._get_rng()

        augmented = np.empty_like(
            beats
        )

        original_time = np.arange(
            signal_length,
            dtype=np.float64,
        )

        control_time = np.linspace(
            0.0,
            float(signal_length - 1),
            num_knots + 2,
        )

        base_intervals = np.diff(
            control_time
        )

        for sample_index in range(
            len(beats)
        ):
            interval_multipliers = np.exp(
                rng.normal(
                    loc=0.0,
                    scale=sigma,
                    size=len(
                        base_intervals
                    ),
                )
            )

            warped_intervals = (
                base_intervals
                * interval_multipliers
            )

            warped_control_time = (
                np.concatenate(
                    [
                        np.asarray([0.0]),
                        np.cumsum(
                            warped_intervals
                        ),
                    ]
                )
            )

            warped_control_time *= (
                float(signal_length - 1)
                / warped_control_time[-1]
            )

            warped_time = np.interp(
                original_time,
                control_time,
                warped_control_time,
            )

            augmented[
                sample_index
            ] = np.interp(
                original_time,
                warped_time,
                beats[sample_index],
            )

        return augmented.astype(
            np.float32,
            copy=False,
        )

    def cutout(
        self,
        beats,
        num_holes=1,
        length=20,
    ):
        """
        Zero one or more continuous intervals in each segment.
        """
        beats = self._ensure_2d(
            beats
        )

        num_holes = (
            self._validate_positive_integer(
                num_holes,
                "num_holes",
                allow_zero=True,
            )
        )

        length = (
            self._validate_positive_integer(
                length,
                "length",
            )
        )

        if beats.size == 0:
            return beats.copy()

        signal_length = beats.shape[1]

        if length > signal_length:
            raise ValueError(
                "length cannot exceed the ECG signal length."
            )

        augmented = beats.copy()

        if num_holes == 0:
            return augmented

        rng = self._get_rng()

        for sample_index in range(
            len(augmented)
        ):
            for _ in range(
                num_holes
            ):
                start = rng.randint(
                    0,
                    signal_length
                    - length
                    + 1,
                )

                augmented[
                    sample_index,
                    start:
                    start + length,
                ] = 0.0

        return augmented.astype(
            np.float32,
            copy=False,
        )

    def emg_noise(
        self,
        beats,
        fs=250,
        std=0.05,
    ):
        """
        Add high-pass-filtered noise resembling EMG interference.
        """
        beats = self._ensure_2d(
            beats
        )

        fs = self._validate_non_negative(
            fs,
            "fs",
        )

        std = self._validate_non_negative(
            std,
            "std",
        )

        if fs <= 40.0:
            raise ValueError(
                "fs must be greater than 40 Hz for "
                "the 20 Hz EMG high-pass filter."
            )

        if beats.size == 0:
            return beats.copy()

        if std == 0.0:
            return beats.copy()

        rng = self._get_rng()

        noise = rng.standard_normal(
            beats.shape
        ).astype(
            np.float32
        )

        numerator, denominator = butter(
            4,
            20,
            btype="high",
            fs=fs,
        )

        coloured_noise = lfilter(
            numerator,
            denominator,
            noise,
            axis=1,
        )

        signal_std = (
            beats.std(
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

        noise_std = (
            coloured_noise.std(
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

        coloured_noise = (
            coloured_noise
            * (
                std
                * signal_std
                / noise_std
            )
        )

        return (
            beats + coloured_noise
        ).astype(
            np.float32,
            copy=False,
        )

    def istft_augment(
        self,
        beats,
        fs,
        window="hann",
        nperseg=128,
        noverlap=64,
        noise_std=0.05,
        log_power=True,
    ):
        """
        Perturb STFT magnitude while preserving phase.
        """
        beats = self._ensure_2d(
            beats
        )

        fs = self._validate_non_negative(
            fs,
            "fs",
        )

        if fs <= 0.0:
            raise ValueError(
                "fs must be greater than zero."
            )

        nperseg = (
            self._validate_positive_integer(
                nperseg,
                "nperseg",
            )
        )

        noverlap = (
            self._validate_positive_integer(
                noverlap,
                "noverlap",
                allow_zero=True,
            )
        )

        noise_std = (
            self._validate_non_negative(
                noise_std,
                "noise_std",
            )
        )

        if not isinstance(
            log_power,
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError(
                "log_power must be Boolean."
            )

        if noverlap >= nperseg:
            raise ValueError(
                "noverlap must be smaller than nperseg."
            )

        if beats.size == 0:
            return beats.copy()

        rng = self._get_rng()
        augmented = []

        for segment in beats:
            effective_nperseg = min(
                nperseg,
                len(segment),
            )

            if effective_nperseg < 2:
                augmented.append(
                    segment.copy()
                )
                continue

            effective_noverlap = min(
                noverlap,
                effective_nperseg - 1,
            )

            _, _, spectrum = stft(
                segment,
                fs=fs,
                window=window,
                nperseg=effective_nperseg,
                noverlap=effective_noverlap,
            )

            magnitude = np.abs(
                spectrum
            )

            phase = np.angle(
                spectrum
            )

            if log_power:
                log_magnitude = np.log(
                    magnitude + 1e-8
                )

                noise = rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=log_magnitude.shape,
                )

                augmented_magnitude = np.exp(
                    log_magnitude + noise
                )
            else:
                magnitude_scale = float(
                    magnitude.std()
                )

                noise = rng.normal(
                    loc=0.0,
                    scale=(
                        noise_std
                        * magnitude_scale
                    ),
                    size=magnitude.shape,
                )

                augmented_magnitude = np.clip(
                    magnitude + noise,
                    0.0,
                    None,
                )

            augmented_spectrum = (
                augmented_magnitude
                * np.exp(
                    1j * phase
                )
            )

            _, reconstructed = istft(
                augmented_spectrum,
                fs=fs,
                window=window,
                nperseg=effective_nperseg,
                noverlap=effective_noverlap,
            )

            if len(reconstructed) >= len(
                segment
            ):
                reconstructed = reconstructed[
                    :len(segment)
                ]
            else:
                reconstructed = np.pad(
                    reconstructed,
                    (
                        0,
                        len(segment)
                        - len(reconstructed),
                    ),
                )

            augmented.append(
                reconstructed
            )

        return np.stack(
            augmented
        ).astype(
            np.float32,
            copy=False,
        )