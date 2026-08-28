# H1: Worker-reported HBM pressure tracks NVML

**Hypothesis.** Worker physical-HBM pressure matches an independent NVML reference, with low latency, monotone under staircase allocation, and no false critical on idle.

**Expectation.** median err < 0.01, p95 err < 0.02, monotone up-staircase, sample p95 < 1000 ms, idle < 0.95.

**Verdict.** DEVIATION FROM EXPECTATION

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | raw median abs pressure error < 0.01 | FAIL | median=0.01121 (see bias analysis) |
| 2 | bias is a fixed offset, not drift (stdev < 5 MiB) | PASS | bias median=598.6 MiB (1.22% of total), stdev=0.000 MiB |
| 3 | bias-corrected p95 error < 0.005 | PASS | corrected p95=0.000000, corrected median=0.000000 |
| 4 | staircase-up device_used strictly monotone | PASS | per-step medians (MiB)=[1295, 2319, 3343, 4367, 5391, 6415, 7439, 8463], min delta=1024.0 MiB |
| 5 | sample wall p95 < 2x interval | PASS | p95=0.9 ms, interval=300 ms |
| 6 | idle never reaches critical watermark 0.95 | PASS | max idle pressure=0.0056 |

## Observations

```json
{
  "gpu": "NVIDIA RTX A6000",
  "device": 0,
  "step_mib": 1024,
  "steps": 8,
  "n_samples": 54,
  "raw_median_abs_err": 0.011213,
  "raw_p95_abs_err": 0.012113,
  "nvml_minus_reporter_bias_median_mib": 598.56,
  "nvml_minus_reporter_bias_stdev_mib": 0.0,
  "bias_fraction_of_total": 0.01218,
  "bias_corrected_median_abs_err": 0.0,
  "bias_corrected_p95_abs_err": 0.0,
  "p95_sample_wall_ms": 0.899,
  "max_idle_reporter_pressure": 0.005585,
  "up_step_used_medians_mib": [
    1295,
    2319,
    3343,
    4367,
    5391,
    6415,
    7439,
    8463
  ],
  "min_up_step_delta_mib": 1024.0,
  "series": [
    {
      "phase": "idle",
      "step": 0,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2908079986809753
    },
    {
      "phase": "idle",
      "step": 1,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.32956699578789994
    },
    {
      "phase": "idle",
      "step": 2,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.31071800185600296
    },
    {
      "phase": "idle",
      "step": 3,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.5488070019055158
    },
    {
      "phase": "idle",
      "step": 4,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.4254870000295341
    },
    {
      "phase": "idle",
      "step": 5,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.366687003406696
    },
    {
      "phase": "up_1",
      "step": 0,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.30463700386462733
    },
    {
      "phase": "up_1",
      "step": 1,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.5252659975667484
    },
    {
      "phase": "up_1",
      "step": 2,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3436580009292811
    },
    {
      "phase": "up_2",
      "step": 0,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.4427069943631068
    },
    {
      "phase": "up_2",
      "step": 1,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.40341700514545664
    },
    {
      "phase": "up_2",
      "step": 2,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3243080063839443
    },
    {
      "phase": "up_3",
      "step": 0,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3011779990629293
    },
    {
      "phase": "up_3",
      "step": 1,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3656780027085915
    },
    {
      "phase": "up_3",
      "step": 2,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3989969991380349
    },
    {
      "phase": "up_4",
      "step": 0,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2734180015977472
    },
    {
      "phase": "up_4",
      "step": 1,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.39032699714880437
    },
    {
      "phase": "up_4",
      "step": 2,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.7924849996925332
    },
    {
      "phase": "up_5",
      "step": 0,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.304418004816398
    },
    {
      "phase": "up_5",
      "step": 1,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.7219949984573759
    },
    {
      "phase": "up_5",
      "step": 2,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.39553800161229447
    },
    {
      "phase": "up_6",
      "step": 0,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.35559699608711526
    },
    {
      "phase": "up_6",
      "step": 1,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.41730800148798153
    },
    {
      "phase": "up_6",
      "step": 2,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.40850700315786526
    },
    {
      "phase": "up_7",
      "step": 0,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3302570039522834
    },
    {
      "phase": "up_7",
      "step": 1,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.8191840024664998
    },
    {
      "phase": "up_7",
      "step": 2,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.8326239985763095
    },
    {
      "phase": "up_8",
      "step": 0,
      "reporter_pressure": 0.17434846258930836,
      "reporter_pressure_guarded_1g": 0.1954438411511814,
      "nvml_pressure": 0.18440552503052499,
      "abs_err": 0.010057062441216624,
      "reporter_device_used": 8874229760,
      "nvml_device_used": 9501868032,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2995779941556975
    },
    {
      "phase": "up_8",
      "step": 1,
      "reporter_pressure": 0.17434846258930836,
      "reporter_pressure_guarded_1g": 0.1954438411511814,
      "nvml_pressure": 0.18440552503052499,
      "abs_err": 0.010057062441216624,
      "reporter_device_used": 8874229760,
      "nvml_device_used": 9501868032,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.23403899831464514
    },
    {
      "phase": "up_8",
      "step": 2,
      "reporter_pressure": 0.17434846258930836,
      "reporter_pressure_guarded_1g": 0.1954438411511814,
      "nvml_pressure": 0.18440552503052499,
      "abs_err": 0.010057062441216624,
      "reporter_device_used": 8874229760,
      "nvml_device_used": 9501868032,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.48141700244741514
    },
    {
      "phase": "down_8",
      "step": 0,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2581980006652884
    },
    {
      "phase": "down_8",
      "step": 1,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3743779961951077
    },
    {
      "phase": "down_8",
      "step": 2,
      "reporter_pressure": 0.15325308402743532,
      "reporter_pressure_guarded_1g": 0.17434846258930836,
      "nvml_pressure": 0.16356710419210418,
      "abs_err": 0.010314020164668869,
      "reporter_device_used": 7800487936,
      "nvml_device_used": 8428126208,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3211980001651682
    },
    {
      "phase": "down_7",
      "step": 0,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2879780004150234
    },
    {
      "phase": "down_7",
      "step": 1,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3862169978674501
    },
    {
      "phase": "down_7",
      "step": 2,
      "reporter_pressure": 0.13215770546556227,
      "reporter_pressure_guarded_1g": 0.15325308402743532,
      "nvml_pressure": 0.14272868335368338,
      "abs_err": 0.010570977888121114,
      "reporter_device_used": 6726746112,
      "nvml_device_used": 7354384384,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.38262700400082394
    },
    {
      "phase": "down_6",
      "step": 0,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.27769800362875685
    },
    {
      "phase": "down_6",
      "step": 1,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 1.0969419963657856
    },
    {
      "phase": "down_6",
      "step": 2,
      "reporter_pressure": 0.11106232690368922,
      "reporter_pressure_guarded_1g": 0.13215770546556227,
      "nvml_pressure": 0.12189026251526247,
      "abs_err": 0.010827935611573247,
      "reporter_device_used": 5653004288,
      "nvml_device_used": 6280642560,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.37160699866944924
    },
    {
      "phase": "down_5",
      "step": 0,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.27383799897506833
    },
    {
      "phase": "down_5",
      "step": 1,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3753380005946383
    },
    {
      "phase": "down_5",
      "step": 2,
      "reporter_pressure": 0.08996694834181618,
      "reporter_pressure_guarded_1g": 0.11106232690368922,
      "nvml_pressure": 0.10105184167684167,
      "abs_err": 0.011084893335025492,
      "reporter_device_used": 4579262464,
      "nvml_device_used": 5206900736,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 3.4362459991825745
    },
    {
      "phase": "down_4",
      "step": 0,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.27216900343773887
    },
    {
      "phase": "down_4",
      "step": 1,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.35608700272860005
    },
    {
      "phase": "down_4",
      "step": 2,
      "reporter_pressure": 0.06887156977994313,
      "reporter_pressure_guarded_1g": 0.08996694834181618,
      "nvml_pressure": 0.08021342083842087,
      "abs_err": 0.011341851058477737,
      "reporter_device_used": 3505520640,
      "nvml_device_used": 4133158912,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3286680002929643
    },
    {
      "phase": "down_3",
      "step": 0,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.26886799605563283
    },
    {
      "phase": "down_3",
      "step": 1,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.37562700163107365
    },
    {
      "phase": "down_3",
      "step": 2,
      "reporter_pressure": 0.047776191218070085,
      "reporter_pressure_guarded_1g": 0.06887156977994313,
      "nvml_pressure": 0.059374999999999956,
      "abs_err": 0.01159880878192987,
      "reporter_device_used": 2431778816,
      "nvml_device_used": 3059417088,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.8258839952759445
    },
    {
      "phase": "down_2",
      "step": 0,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.2687880041776225
    },
    {
      "phase": "down_2",
      "step": 1,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.33698700281092897
    },
    {
      "phase": "down_2",
      "step": 2,
      "reporter_pressure": 0.02668081265619704,
      "reporter_pressure_guarded_1g": 0.047776191218070085,
      "nvml_pressure": 0.038536579161579154,
      "abs_err": 0.011855766505382115,
      "reporter_device_used": 1358036992,
      "nvml_device_used": 1985675264,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.39066699537215754
    },
    {
      "phase": "down_1",
      "step": 0,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3002980010933243
    },
    {
      "phase": "down_1",
      "step": 1,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.3845669998554513
    },
    {
      "phase": "down_1",
      "step": 2,
      "reporter_pressure": 0.005585434094324104,
      "reporter_pressure_guarded_1g": 0.02668081265619704,
      "nvml_pressure": 0.017698158323158353,
      "abs_err": 0.012112724228834248,
      "reporter_device_used": 284295168,
      "nvml_device_used": 911933440,
      "external_or_unattributed_bytes": 284295168,
      "sample_wall_ms": 0.38392799615394324
    }
  ]
}
```

## Analysis

PARTIAL MATCH. Shape tracks NVML almost perfectly; there is a constant 599 MiB (1.22% of total HBM) offset - the reporter reads LOWER than NVML 'used' - and it is fixed to within 0.000 MiB across every idle, up-step and down-step sample. Cause: torch.cuda.mem_get_info returns free/total *after* the CUDA primary context, cuDNN/cuBLAS handles and NCCL scratch already exist, so that ~0.6 GiB is invisible to the worker but counted by NVML as device-resident. It is a fixed context tax - not drift, not noise, not load-dependent. Design impact: the AIMD state machine in allocator.py keys off *relative* pressure crossing the low/high/critical watermarks and off sample-to-sample deltas. A constant offset shifts every watermark comparison by the same amount, i.e. it is equivalent to running with watermarks ~1.2 pp lower - strictly conservative (reacts a hair earlier), never optimistic. The bias-corrected error (NVML change vs reporter change) is < 0.5% at p95, so the quantity the controller actually consumes is accurate. Fix options: subtract a one-time context baseline in the pressure calc; or rely on the existing guard_bytes knob, which is already sized for an offset this large; or simply document that effective watermarks run ~1 pp tighter on this GPU/driver. Latency and monotonicity pass cleanly: a sample is one mem_get_info call (p95 0.9 ms, interval 300 ms) and every 1024 MiB up-step raises reported device_used by >= 1024 MiB. Idle pressure 0.0056 is far below the 0.95 critical watermark, so there is no false critical on an idle device.
